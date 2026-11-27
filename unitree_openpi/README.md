# G1-D / OpenPI

机器人客户端通过 DDS 读取双臂反馈，通过 ZMQ 接收相机画面，再通过 WebSocket 向云端策略请求动作。设备与协议配置见 [g1d.json](config/g1d.json)，代码来源见 [第三方声明](THIRD_PARTY_NOTICES.md)。


## G1-D 遥操作数采（PICO 手柄 + 内置 Dex1）

在机器人终端运行。使用此前采集货架数据的部署目录；其中的采集入口支持手柄控制和录制。先结束正在控制双臂的推理或 reset 保持程序，再启动遥操作。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tv
cd /home/unitree/zhoumy/xr_teleoperate/teleop

# 按本批演示修改目录名和任务文本。下面以罐头扫码为例。
TASK_NAME="scan_demo_v1"
TASK_GOAL="pick milk, scan, then place"

PYTHONPATH=.. python teleop_hand_and_arm.py \
  --input-mode controller --control-source controller --display-mode pass-through \
  --arm G1_29 --ee dex1_internal --base-type legs \
  --network-interface eth0 --img-server-ip 192.168.123.164 --xr-deadman-s 0.3 \
  --record --barcode-scanner --frequency 30 --arm-reset-speed 1.5 \
  --task-dir "$HOME/g1d_dataset" --task-name "$TASK_NAME" \
  --task-goal "$TASK_GOAL" --task-desc "Fixed-base G1-D can scanning with internal Dex1 grippers." \
  --task-steps "Grasp the bottle, lift it, present and rotate it, then hold after a scan, then place it." \
  --headless

python3 -u /home/unitree/thor_barcode.py --device /dev/input/event6
```

启用录制且 XR 输入有效时，可以先按 **B+Y** 开始录制，再接管手臂。单臂接管和回零期间会持续记录同一条轨迹。单击并松开 **Y/B** 分别接管左/右臂，单击并松开 **X/A** 分别让左/右臂回到关节零位并保持。组合按键需有按住时间重叠：**B+Y** 开始或保存录制；录制期间 **A+X** 丢弃并暂停，未录制时退出。暂停后可用 **Y** 或 **B** 重新接管。原始轨迹保存在 `~/g1d_dataset/<TASK_NAME>/episode_*/`，任务描述和状态动作记录在各轨迹的 `data.json` 中。

THOR USB 扫码数采：在上方的数采命令增加 `--barcode-scanner`。在 THOR 执行 `python3 thor_barcode.py`，脚本源在机器人 `~/zhoumy/xr_teleoperate/unitree_openpi/scripts/thor_barcode.py`。默认读取 /dev/input/event6，发送到 192.168.123.164:48166。每条轨迹第一次扫码显示 accepted，重复显示 already_scanned，录制外显示 outside_episode。每帧 _unitree.scan_trigger 是单帧事件，scan_success 从该帧保持到结束。顶层 _unitree.scan_event 保存条码和机器人接收时间。扫码不切换手臂或录制状态。

扫码任务可在机器人的第二个终端启动声光提示。扫码脚本独立运行，成功事件目前只打印到终端；录制的轨迹不会自动包含 `scan_success` 标签。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tv
cd ~/zhoumy/xr_teleoperate
python unitree_openpi/scripts/test_scan.py
```

## 运行

在仓库根目录执行。真机使用 Linux、Python 3.10–3.12 和 Unitree SDK2；EEF 逆解需要带 `pinocchio.casadi` 绑定的 Pinocchio 和 CasADi/IPOPT。先启动机器人的三路相机发布服务。

```bash
python -m pip install -r unitree_openpi/requirements.txt
# 4090 / openpi
python -m examples.g1d.main --config pi05_g1d --checkpoint ~/openpi/checkpoints/pi05_g1d/shelf_pp_test/20000
# 机器人：上传真实观测并校验一轮回复
python unitree_openpi/scripts/run_g1d_client.py --url ws://192.168.66.200:8000 --max-steps 1 --prompt "pick and place the object on the shelf"
# 机器人：持续执行云端动作
python unitree_openpi/scripts/run_g1d_client.py --url ws://192.168.66.200:8000 --execute --prompt "pick and place the object on the shelf"
# Read-only test: print predictions without acquiring actuators.
python unitree_openpi/scripts/test_g1d_client.py --url ws://192.168.66.200:8000 --max-steps 11
```


## WebSocket 数据协议

机器人主动连接 `policy.url`（`ws://` 或 `wss://`）。默认 `protocol=openpi`，每条消息是一帧二进制 MessagePack，NumPy 数组使用 OpenPI 编码。配置了 `OPENPI_API_KEY` 时，握手带 `Authorization: Api-Key <key>`。

The joint client requires G1-D absolute joint metadata. State and each action use [left arm 7, raw left gripper, right arm 7, raw right gripper]. Both clients execute cloud targets only with --execute on hardware. The EEF formats below describe the legacy adapter.

G1-D 入口默认每个 chunk 最多执行 20 条动作（`--max-execute 20`）。启用 `--rtc` 时，可执行窗口还剩 10 条动作就发出下一个请求（`--rtc-prefix-len 10`）。模型返回的长度小于上限时，按实际返回长度计算。

1. **云端 → 机器人**：连接后先发 metadata 字典，允许空字典 `{}`。
2. **机器人 → 云端**：发送一条 observation，默认字段如下。
3. **云端 → 机器人**：返回一条带 `actions` 的字典。执行结果随下一条 observation 的 `_unitree.last_action` 返回。

| 观测字段 | 默认内容 |
|---|---|
| `observation/state` | `float32[14]`，左 7 后右 7：每臂 `[x,y,z,rx,ry,rz,g]`。位置（m）和旋转向量（rad）均为相对 `torso_link` 的绝对位姿，g 为实测归一化夹爪开度 |
| `observation/images/high`、`observation/images/left_wrist`、`observation/images/right_wrist` | 三个顶层字符串键，各为 `uint8[224,224,3]` RGB、HWC 图像。high 默认取头部双目左半幅 |
| `prompt` | 任务字符串，默认空串 |
| `_unitree` | `sequence`、`frame_id`、`received_at`、`state_age_s`、`state_representation`、`action_representation`、`joint_action_representation`、`action_frame`、每路接收时间和序号 `cameras`、`last_action` |
| `_unitree` 内的实测反馈 | `joint_positions[14]`（rad）、`joint_velocities[14]`（rad/s），均左 7 后右 7。`waist_q3=[yaw,0,pitch]`，`grip_q2` 为左右夹爪原始电机位置 |

| 云端回复（示意） | 动作含义 |
|---|---|
| `{"actions": eef}` 或 `{"actions":{"eef": eef}}` | `eef[H,14]`，每臂 `[dx,dy,dz,drx,dry,drz,g_target]`。g_target 为绝对归一化夹爪目标（0 闭、1 开）。平移增量用 m，旋转向量用 rad。每步相对执行前的实测位姿，默认在基座系施加，可配置为工具系 |
| `{"actions":{"joint_positions": q}}` | `q[H,14]`，绝对关节角（rad），左 7 后右 7，每臂依次为肩 pitch/roll/yaw、肘、腕 roll/pitch/yaw |
| `{"actions":{"joint_positions": q,"eef": eef}}` | 两者步数相同。逐步用 FK 检查两种目标一致后执行关节目标 |

旧版 EEF 适配器也接受表示单步动作的一维数组。其默认 `1 ≤ H ≤ 64`，每轮执行第一步后采集新观测。回复可带顶层 `observation_id`，其值须等于本次 `_unitree.sequence`。默认观测有效期 2 秒。

夹爪通道索引为 6、13（从零计数）。默认内置 Dex1 电机映射为闭合 0、打开 2.3 rad，可用 `closed_q2` / `open_q2` 按实机标定修改。可选 `protocol=json` 使用文本帧、数字列表和 JPEG 图像对象（`encoding="jpeg_base64"`、`color_space="RGB"`、`shape`、`data`）。

## 可选：笔记本预览与调焦

在机器人运行四画面预览，再在笔记本打开 SSH 隧道。头部左右目取自同一条双目流，可在实际工作距离分别旋转两侧镜头对焦环，对照画面清晰度。

```bash
cd ~/zhoumy/xr_teleoperate/unitree_robotskills/unitree_gpt6/g1d_gomoku
PYTHONPATH=src python -m hardware.camera_web \
  --config config/hardware.example.yaml --host 192.168.123.164 \
  --cameras cam_left_high cam_right_high cam_right_wrist cam_left_wrist \
  --web-host 127.0.0.1 --web-port 8081
```

笔记本执行 `ssh -N -L 8081:127.0.0.1:8081 unitree@192.168.66.47`，浏览器打开 `http://127.0.0.1:8081`。

## G1-D 直接录制为 LeRobot v3

普通遥操在原命令加 `--record --record-format lerobot_v3`，使用内置 Dex1 和四路相机。DAgger 在执行命令中加 `--execute --record-format lerobot_v3`。扫码任务同时使用 `--barcode-scanner`。省略格式参数使用原始 JSON 模式。

输出在原任务目录名后加 `_lerobot_v3`。B+Y 提交或开启普通示范，录制中 A+X 丢弃。DAgger 每次松开全部 Grip 提交一个人工纠错段。退出后等待写入进程结束再拷贝或训练。控制终端使用 tv，写入进程默认使用 ~/miniconda3/envs/unitree_lerobot/bin/python，可通过 LEROBOT_PYTHON 指定解释器。队列满会提示并丢弃当前不完整段。

磁盘 state 为 [scan_success, L7, R7, Lgrip, Rgrip]，action 为 16 维绝对电机目标。每帧另存 episode_s、scan_trigger、barcode_ascii（零填充 ASCII）和左右 clutch。
