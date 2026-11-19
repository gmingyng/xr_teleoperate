# G1-D / OpenPI

机器人客户端通过 DDS 读取双臂反馈，通过 ZMQ 接收相机画面，再通过 WebSocket 向云端策略请求动作。设备与协议配置见 [g1d.json](config/g1d.json)，代码来源见 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 运行

在仓库根目录执行。真机使用 Linux、Python 3.10–3.12 和 Unitree SDK2；客户端用 Pinocchio 验证工作范围并计算重力补偿。先启动机器人的三路相机发布服务。

```bash
python -m pip install -r unitree_openpi/requirements.txt
# 云端联调：启动模拟策略；正式运行时换成实际策略服务
python unitree_openpi/scripts/serve_mock_policy.py --host 192.168.66.200 --port 8000
# 机器人：上传真实观测并校验一轮回复
python unitree_openpi/scripts/run_g1d_client.py --server cloud --max-steps 1
# 机器人：持续执行云端动作
python unitree_openpi/scripts/run_g1d_client.py --server cloud --execute
# 机器人：打印云端动作，双臂只向本地自然下垂位移动
python unitree_openpi/scripts/test_g1d_client.py --server cloud --execute
```

两个机器人客户端均支持 `--server cloud`（现有云端 `192.168.66.200:8000`）或 `--server thor`（背包 Thor `192.168.123.166:8000`）。自定义服务可传 `--url`；省略两项时读取配置的 `policy.url`。

Thor 联调：通过 `ssh unitree@192.168.123.166` 登录，在 Thor 项目根目录运行 `python unitree_openpi/scripts/serve_mock_policy.py --host 192.168.123.166 --port 8000`。随后在机器人运行 `python unitree_openpi/scripts/run_g1d_client.py --server thor --max-steps 1` 验证一次往返。

## WebSocket 数据协议

机器人主动连接 `policy.url`（`ws://` 或 `wss://`）。默认 `protocol=openpi`，每条消息是一帧二进制 MessagePack，NumPy 数组使用 OpenPI 编码。配置了 `OPENPI_API_KEY` 时，握手带 `Authorization: Api-Key <key>`。

1. **云端 → 机器人**：连接后先发 metadata 字典，允许空字典 `{}`。
2. **机器人 → 云端**：发送一条 observation，默认字段如下。
3. **云端 → 机器人**：返回一条带 `actions` 的字典。执行结果随下一条 observation 的 `_unitree.last_action` 返回。

| 观测字段 | 默认内容 |
|---|---|
| `observation/state` | `float32[16]`，左右臂各 `[肩 pitch, 肩 roll, 肩 yaw, 肘, 腕 roll, 腕 pitch, 腕 yaw, 夹爪]`。七个关节为绝对角度（rad），夹爪为绝对电机位置 |
| `observation/images/high`、`observation/images/left_wrist`、`observation/images/right_wrist` | 三个顶层字符串键，各为 `uint8[224,224,3]` RGB、HWC 图像。high 默认取头部双目左半幅 |
| `prompt` | 任务字符串，默认空串 |
| `_unitree` | `sequence`、`received_at`、`state_age_s`、`state_representation`、`action_representation`、每路接收时间和序号 `cameras`、`last_action` |
| `_unitree` 内的实测反馈 | `joint_positions[14]`（rad）、`joint_velocities[14]`（rad/s），均左 7 后右 7。`waist_q3=[yaw,0,pitch]`，`grip_q2` 为左右夹爪原始电机位置 |

| 云端回复（示意） | 动作含义 |
|---|---|
| `{"actions": q}` | `q[H,16]`，布局与 state 相同。每行是绝对目标，云端负责 EEF/增量转换，机器人直接执行关节目标。关闭 `include_grippers` 时收发均为 14 维纯关节角。 |

单步也接受一维数组。默认 `1 ≤ H ≤ 64`，每轮只执行第一步，再采集新观测。回复可带顶层 `observation_id`，其值须等于本次 `_unitree.sequence`。默认观测有效期 2 秒。

夹爪通道索引为 7、15（从零计数），收发绝对电机位置（rad）。内置 Dex1 默认命令范围为 0～5.4 rad；可通过 `policy.gripper_command_limits` 配置为 `[[左最小, 右最小], [左最大, 右最大]]`。swing 单独使用 `closed_q2` 和 `open_q2`（默认 0 和 2.3 rad）。可选 `protocol=json` 使用文本帧、数字列表和 JPEG 图像对象（`encoding="jpeg_base64"`、`color_space="RGB"`、`shape`、`data`）。

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
