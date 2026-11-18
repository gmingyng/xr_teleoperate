# G1-D / OpenPI

独立的机器人侧客户端：采集双臂**实际关节反馈经 FK 得到的绝对末端位姿**及头部、左腕、右腕三张图像，通过 WebSocket 请求策略，再将相对 EEF 动作经 IK 转为 G1D DDS 控制目标。运行时不导入 `unitree_robotskills`，无需 XR 设备、Vuer 或 Meshcat。

业务代码和 G1D URDF 已放在本目录内，不导入原项目的 `teleop`、`teleimager` 或 robotskills。整体复制 `unitree_openpi/` 后仍可通过具名脚本启动；NumPy、SciPy、Pinocchio、OpenCV、PyZMQ、websockets、msgpack 和 Unitree SDK 仍需按 `requirements.txt` 安装。相机发布服务和策略服务器为独立服务。

## 目录与提取来源

- `robot_ctrl/dds.py`：从 robotskills 绘画后端提取，保留接管、状态时效、限速、故障保持；增加动作超时和动态夹爪。
- `robot_ctrl/kinematics.py`：从绘画运动学实现改写，支持完整 G1D URDF、连续轮关节、可配置腰部映射、双臂 IK、FK 与重力前馈。
- `robot_ctrl/agv.py`、`chassis.py`：提取导航项目的 AGV RPC API 和带超时的速度发布；EEF 主程序不发送底盘或升降指令。
- `robot_ctrl/controller.py`：相对 EEF 动作、工作范围、关节增量和实测到达检查。
- `utils/`：独立配置、相机持续采集、坐标变换、控制权锁、MessagePack/JSON 编解码和 WebSocket 客户端。
- `scripts/run_g1d_policy.py`：机器人客户端入口；`scripts/serve_mock_policy.py`：模拟策略服务入口。
- `assets/g1d/`：固定版本的官方 G1D URDF 及原许可证，数值运动学不需要 mesh。
- `tests/`、`mock_policy_server.py`：离线测试和模拟策略服务器。

来源及许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。没有移动或修改 robotskills 原文件。

## 安装

真机使用 Linux，推荐单独的 Python 3.10–3.12 环境，避免与原框架的 NumPy/Pinocchio 版本混装。在项目根目录执行：

```bash
conda create -n unitree_openpi python=3.11 -y
conda activate unitree_openpi
pip install -r unitree_openpi/requirements.txt
```

如果 SDK 的 CycloneDDS 需要单独构建，使用 SDK 官方安装流程设置 `CYCLONEDDS_HOME`。模拟模式只需要 NumPy、SciPy、websockets 和 msgpack；JSON 图像与真实相机额外需要 OpenCV、PyZMQ。

## 先运行离线闭环

分别在两个终端、项目根目录运行：

```bash
python unitree_openpi/scripts/serve_mock_policy.py --port 8000
python unitree_openpi/scripts/run_g1d_policy.py --mock --max-steps 10 --prompt "pick up the object"
```

默认模拟服务器返回零增量。`--delta-x 0.001` 可测试左末端每次向基座 X 方向移动 1 mm。模拟运动学是测试映射，不代表 G1D 几何。

```bash
python -m unittest discover -s unitree_openpi/tests -v
```

## 真机配置与运行

复制 `config/g1d.json` 为 `config/local.json`，修改 DDS 网卡、相机地址和策略 URL，核对设备的电机槽位、腰部模型映射、PD 增益、工具变换及工作范围。示例参数不是现场标定结果。

```bash
# 只订阅状态和相机，仍与策略通信，不创建 DDS 命令发布器
python unitree_openpi/scripts/run_g1d_policy.py --config unitree_openpi/config/local.json --prompt "pick up the object"

# 接管并执行策略返回的相对 EEF 动作
python unitree_openpi/scripts/run_g1d_policy.py --config unitree_openpi/config/local.json --execute --prompt "pick up the object"
```

相机服务需提前开启三路单消息 JPEG/ZMQ 流。原框架 TeleImager 默认关闭右腕，需启用 `right_wrist_camera.enable_zmq` 并配置设备。默认头部输入为左右拼接双目，取左半；单目请设置 `head_view: "full"`。图像转换为 HWC、uint8、RGB，并保持比例填充到 224×224；分辨率、头部裁剪需与训练数据一致。

G1D DDS 映射来自绘画后端：双臂 15–28，内置 Dex1 为 31/33，腰部 12=yaw、13=pitch，14 不启用。腰部状态向量为 `[yaw,0,pitch]`。启动接管时保留当前双臂、腰部、夹爪姿态，其他电机不启用。

默认 EEF 基座为 **上身基座 `torso_link`**，不是世界坐标或 AGV 地图坐标。FK 与动作始终使用同一个 `model.reference_frame`，随观测明确发送 `frame_id`。如改为 `AGV_link`，必须将 `reference_positions` 中两个升降关节设为实测值，并保证升降位置不变；本程序尚未订阅升降高度或 SLAM 位姿。默认上身基座的 EEF 几何不依赖升降值。腰部反馈通过 `waist_joint_map` 更新，用于全模型重力计算；名称、方向和零点必须匹配设备版本。

## 状态、相对动作与 WebSocket 协议

默认协议遵循 [OpenPI WebSocket 客户端](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py)：机器人连接策略服务器，先接收一条 MessagePack metadata，然后循环发送 observation、接收 `{"actions": ndarray}`。NumPy 编码兼容 OpenPI 的 `msgpack_numpy`，不使用 pickle。`OPENPI_API_KEY` 可提供认证。

每次 observation 的默认结构：

```python
{
    "state": float32[12],  # 左 xyz+rpy，然后右 xyz+rpy；均为绝对值
    "images": {
        "head": uint8[224,224,3],
        "left_wrist": uint8[224,224,3],
        "right_wrist": uint8[224,224,3],
    },
    "prompt": "task instruction",
    "_unitree": {
        "sequence": 1, "frame_id": "torso_link",
        "joint_positions": [...], "joint_velocities": [...],
        "waist_q3": [...], "grip_q2": [...],
        "cameras": {...}, "last_action": {...},
        # 时间、表示方式和动作坐标系等信息
    },
}
```

默认 `actions` 为 `(T,12)`：

```text
[left_dx, left_dy, left_dz, left_drx, left_dry, left_drz,
 right_dx,right_dy,right_dz,right_drx,right_dry,right_drz]
```

平移单位米，旋转单位弧度。**观测中的姿态是 XYZ 欧拉角，动作中的旋转是旋转向量**，不直接相加欧拉角。默认 `action_frame: "base"`：

```text
p_target = p_measured + delta_p
R_target = Exp(delta_rotation_vector) @ R_measured
```

也支持 `action_frame: "tool"`，对应 `p_target = p + R @ delta_p`、`R_target = R @ Exp(delta_rotation_vector)`。每个动作基于执行开始时的实测 FK 位姿，属于逐步增量，不是整个 chunk 都相对于最初 observation 的偏移。

默认 `include_grippers: false`，夹爪保持启动状态。需要夹爪时设为 true，并填写现场标定的 `open_q2` 和 `closed_q2`；状态、动作变为 14 维，各臂六维 EEF 后跟一个夹爪值。夹爪状态和目标都是绝对归一化值：0=closed、1=open，EEF 仍为相对动作。禁止使用未标定的电机单位作为归一化值。

`state_key`、`prompt_key`、`image_keys` 和 `image_container_key` 可映射训练的观测字段；`image_container_key: null` 表示将图像字段放在顶层。**协议兼容不意味着任意 OpenPI checkpoint 都能驱动 G1D**：必须使用按上述维度、图像预处理和相对动作约定训练的模型，或在服务端添加对应适配器。服务端不得再重复执行 delta 化；输出必须是解归一化后的物理单位。

自定义 JSON 对端可设置 `protocol: "json"`，仍需先发送 JSON metadata，再按一问一答通信。状态为数字列表，图像为含 `encoding: "jpeg_base64"` 的对象。模拟服务器增加 `--protocol json` 可联调。

## 执行和反馈

默认每次推理执行 chunk 的第一个动作后重新采样（`execute_steps: 1`）。每个执行目标等待关节反馈达到容差，之后在下一次 observation 的 `_unitree.last_action` 返回 `reached` 或错误；不额外向官方服务器插入 ACK 消息。可配置执行更多步，但每步仍以最新反馈为基准，且整个 chunk 必须在观测有效期内完成。它不是严格固定周期的高频 chunk 播放器。`command_hz` 是上层循环频率上限，DDS 独立以 250 Hz 发布并限制关节速度。

收到回复后检查观测年龄、全 chunk 格式、平移与旋转限幅；每步检查工作范围、IK 误差、关节跳变、求解期间的反馈漂移及动作期限。默认每步平移≤3 cm、旋转向量模≤0.15 rad，超过限制会拒绝，不静默裁剪。SDK 回调过期、控制模式变化或跟踪故障会进入终止保持。

网络超时、坏动作、相机过期或断连会取消待执行目标，在下一次连接使用新观测，旧回复不重放。底层独立动作超时会保持最后已发布目标，避免推理卡住时继续追踪旧目标。退出不会自动回零；关闭 DDS 后不保证持续带电保持。

本实现没有全身碰撞检测、场景避障或视觉目标识别。实测 FK 状态仍受模型、工具标定和编码器误差影响。相机时间为**主机接收时间**，不是曝光时间；仅检查接收年龄和三路接收时间差，不声明硬件同步。

### Kinematics backend

FK and gravity use Pinocchio. IK uses Pinocchio symbolic FK with CasADi/IPOPT, joint bounds and measured-state initialization. Install CasADi and a Pinocchio build exposing `pinocchio.casadi`; installing `pin` alone does not guarantee those bindings. There is no silent SciPy IK fallback. The full G1D model, measured waist configuration, reference frame and tool transforms are preserved.

### 双臂电机调试脚本

默认 `config/g1d.json` 增益与 `teleop/robot_control/robot_arm.py` 的
`G1_29_Arm_Internal_Dex1_Controller` 对齐：肩/肘 Kp=80、Kd=3，腕部
Kp=40、Kd=1.5，腰部 Kp=300、Kd=3，内置夹爪 Kp=5、Kd=0.05。
这是增益数值对齐；G1D 仍使用自己的两轴腰部槽位映射，不复制 G1 人形机器人的全身映射。

```bash
# 离线检查（不连接机器人）
python unitree_openpi/scripts/reset_dual_arms.py --mock
python unitree_openpi/scripts/swing_dual_arms.py --mock

# 真机；eth0 替换为机器人通信网卡
python unitree_openpi/scripts/reset_dual_arms.py --execute --interface eth0 --duration 5
python unitree_openpi/scripts/swing_dual_arms.py --execute --interface eth0 --amplitude-deg 3 --period 6 --cycles 3
```

复位沿用 teleop 的零关节角定义（14 维全零，不是末端坐标归零），从实测关节角做平滑插值，
根据配置速度上限自动延长时间。摆动围绕启动姿态，仅改变左右肩俯仰关节，默认 ±3°、周期 6 秒、3 个周期，
采用 sin³ 波形使开始和结束速度为零，结束回到启动姿态；幅度最大 10°。
两者保持其余手臂关节、腰部和夹爪目标，使用现有 G1DDSBackend 的 250 Hz LowCmd、
CRC、状态超时、跟踪检查和限速，以及 Pinocchio 重力前馈。无需相机、WebSocket 或 IK/CasADi。
真机需要 Pinocchio 和 Unitree SDK；模拟模式使用假模型。

正常完成后默认持续发送保持命令，Ctrl+C 释放 DDS；`--release-on-finish` 可选择完成即退出。
退出后不再提供主动保持。异常或中断时取消待执行目标并关闭连接，不自动回零。
同一台机器人不要同时运行 teleop、robotskills 控制服务或其他 LowCmd 发布者；本包锁仅协调使用相同锁文件的进程。
脚本检查关节限位，但不做碰撞检测。真机动作需在双臂运动空间清空的情况下执行。
