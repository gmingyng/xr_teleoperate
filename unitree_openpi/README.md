# G1-D / OpenPI

机器人客户端通过 DDS 读取双臂反馈，通过 ZMQ 接收相机画面，再通过 WebSocket 向云端策略请求动作。设备与协议配置见 [g1d.json](config/g1d.json)，代码来源见 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 运行

在仓库根目录执行。真机使用 Linux、Python 3.10–3.12 和 Unitree SDK2；EEF 逆解需要带 `pinocchio.casadi` 绑定的 Pinocchio 和 CasADi/IPOPT。先启动机器人的三路相机发布服务。

```bash
python -m pip install -r unitree_openpi/requirements.txt
# 云端联调：启动模拟策略；正式运行时换成实际策略服务
python unitree_openpi/scripts/serve_mock_policy.py --host 192.168.66.200 --port 8000
# 机器人：上传真实观测并校验一轮回复
python unitree_openpi/scripts/run_g1d_client.py --url ws://192.168.66.200:8000 --max-steps 1
# 机器人：持续执行云端动作
python unitree_openpi/scripts/run_g1d_client.py --url ws://192.168.66.200:8000 --execute
# 机器人：打印云端动作，双臂只向本地自然下垂位移动
python unitree_openpi/scripts/test_g1d_client.py --url ws://192.168.66.200:8000 --execute
```

## WebSocket 数据协议

机器人主动连接 `policy.url`（`ws://` 或 `wss://`）。默认 `protocol=openpi`，每条消息是一帧二进制 MessagePack，NumPy 数组使用 OpenPI 编码。配置了 `OPENPI_API_KEY` 时，握手带 `Authorization: Api-Key <key>`。

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

单步也接受一维数组。默认 `1 ≤ H ≤ 64`，每轮只执行第一步，再采集新观测。回复可带顶层 `observation_id`，其值须等于本次 `_unitree.sequence`。默认观测有效期 2 秒。

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

笔记本执行 `ssh -N -L 8081:127.0.0.1:8081 unitree@ROBOT_IP`，浏览器打开 `http://127.0.0.1:8081`。
