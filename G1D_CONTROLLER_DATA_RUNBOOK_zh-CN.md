# G1-D 纯 Controller 上半身遥操与数采 RUNBOOK

本文档对应 `g1d-adaptation` 分支当前已完成实机验证的流程：

- PICO Controller 控制双臂和内置 Dex1 夹爪；
- Controller 完成启动、录制切换和安全退出；
- 一次程序运行可连续录制多个 episode；
- 保存 Unitree 原始 JSON/JPEG 数据；
- 转换为 LeRobot v3 数据集并回读验证。

当前范围不包含腿部移动、基于头显/脚环的全身跟随，也不包含深度、触觉和音频。
原始 JSON schema `1.1.0` 已增加 IMU、35 路电机动态、实际下发 LowCmd、XR/Controller
输入、时间戳和相机接收质量。不要使用 `--ee dex3`，G1-D 的内置夹爪应使用
`--ee dex1_internal`。

> 部署状态：Controller 遥操、原始数据保存和 LeRobot v3 转换已经完成实机验证。
> schema `1.1.0` 的丰富原始 JSON 扩充当前只在本地代码中通过测试，尚未部署到机器人；
> 机器人仍运行上一版稳定代码，LeRobot 输出格式不受影响。

## 1. 固定配置

| 项目 | 值 |
| --- | --- |
| 机器人计算机 | `unitree@192.168.66.47` |
| 图像服务器 | `192.168.123.164` |
| 遥操目录 | `/home/unitree/zhoumy/xr_teleoperate/teleop` |
| 遥操环境 | `tv` |
| 转换目录 | `/home/unitree/zhoumy/xr_teleoperate` |
| 转换环境 | `unitree_lerobot` |
| 手臂模型 | `G1_29` |
| 末端执行器 | `dex1_internal` |
| 下半身模式 | `legs`（本流程不下发移动命令） |
| 控制频率 | 30 Hz |
| PICO 页面 | `https://192.168.66.47:8012?ws=wss://192.168.66.47:8012` |

## 2. 主要文件与数据架构

### 2.1 遥操运行代码

机器人上的遥操代码根目录为：

```text
/home/unitree/zhoumy/xr_teleoperate/
├── assets/                                # G1/G1-D、手臂和末端模型，IK 初始化需要
├── scripts/
│   ├── g1d_preflight.sh                   # 启动前只读检查，建议保留
│   └── probe_dex3_state.py                # 早期 Dex3 排查脚本，当前 Dex1 流程不需要
└── teleop/
    ├── teleop_hand_and_arm.py             # 主程序：初始化、控制循环、录制生命周期
    ├── utils/
    │   ├── robot_config.py                # 机器人型号和关节配置
    │   ├── output_gate.py                 # XR 断流/重新使能的硬件输出安全门
    │   ├── controller_controls.py         # A、A+X、B+Y、Grip 的边沿触发和锁定逻辑
    │   ├── episode_writer.py              # 多 episode、JPEG 和 data.json 异步落盘
    │   └── recording_schema.py            # schema 1.1 原始遥测；当前仅本地待部署
    ├── televuer/
    │   ├── televuer.py                    # WebXR/Vuer 事件和共享数据
    │   └── tv_wrapper.py                  # OpenXR 坐标到机器人坐标的转换
    ├── robot_control/
    │   ├── robot_arm.py                   # G1_29 双臂、腰 yaw 和内置 Dex1 DDS 控制
    │   ├── robot_arm_ik.py                # 双臂正逆运动学
    │   └── mobile_control.py              # 移动底盘/升降接口；legs 模式不下发移动
    └── teleimager/src/teleimager/
        └── image_client.py                # 接收头部和腕部图像
```

不要为了精简当前任务而删除整个 `assets`、`robot_control`、`televuer` 或
`teleimager` 目录。即使只运行 `G1_29 + dex1_internal + legs`，其中仍有启动时导入和
模型加载依赖。

### 2.2 相机服务

相机采集是独立的 systemd 服务，不在遥操主进程中启动：

```text
/etc/systemd/system/teleimager.service
/home/unitree/unitree_eai_environment/service/teleimager/
├── cam_config_server.yaml                 # 服务端三路 UVC 配置
└── src/teleimager/image_server.py         # 图像服务端

/home/unitree/.config/xr_teleoperate/
├── cert.pem                               # HTTPS/WSS 证书
└── key.pem                                # 私钥，禁止复制到数据集或提交 Git
```

遥操从图像服务器取得配置后生成客户端缓存：

```text
/home/unitree/.cache/xr_teleoperate/cam_config_client.yaml
/home/unitree/.cache/xr_teleoperate/g1_29_model_cache.pkl
```

这两个缓存可重新生成，但正常运行时应保留；删除模型缓存会导致下一次启动重新构建模型。

### 2.3 LeRobot 转换代码

```text
/home/unitree/zhoumy/xr_teleoperate/
├── data_processing/    # 转换器、机器人字段定义和数据编辑工具
└── lerobot/src/        # 随项目固定的 LeRobot 库源码
```

转换运行在 Conda 环境 `unitree_lerobot` 中。不要只复制转换脚本而省略同目录的项目源码，
否则可能与系统安装的 LeRobot 版本不一致。

### 2.4 数据、转换结果与备份

推荐正式数据布局：

```text
/home/unitree/g1d_dataset/
└── <task_name>/
    ├── episode_0000/
    │   ├── data.json
    │   └── colors/*.jpg
    ├── episode_0001/
    └── ...

/home/unitree/.cache/huggingface/lerobot/
└── <repo_id>/                              # Parquet、MP4 和 meta
```

源码回滚备份与数据备份是两件事。源码备份不能代替 episode 备份：

```text
/home/unitree/backups/                       # 历史源码部署备份
/home/unitree/xr_teleoperate_backups/        # 关键运行文件备份
```

正式原始数据写入：

```text
/home/unitree/g1d_dataset/pick_place_controller_dex1/episode_NNNN
```

该目录应保留并单独备份。名称带 `smoke` 的 LeRobot 本地数据集属于以前的转换验证，
不是正式数据的转换结果。

### 2.5 已验证测试产物的清理边界

以下目录是早期冒烟数据、转换视图或部署 staging；确认不再需要保留一份“已知可用”的
样例后可以清理，但不要和正式目录一起做通配删除：

```text
/home/unitree/g1d_dataset_smoke/             # 5 个旧测试 episode，约 1.3 GB
/home/unitree/g1d_dataset_controller_only/   # 指向 smoke 数据的符号链接视图
/home/unitree/g1d-safety-stage-20261118/      # 已部署完成的临时 staging
/home/unitree/.cache/huggingface/lerobot/local/g1d_upper_body_dex1_smoke_v1
/home/unitree/.cache/huggingface/lerobot/local/g1d_controller_upper_body_dex1_smoke_v1
```

以下内容虽然不是运行时主路径，但建议继续保留作为回滚点：

```text
/home/unitree/backups/g1d-teleop-20261116-201130
/home/unitree/xr_teleoperate_backups/20261118_raw_json_predeploy_8a192d7.tar.gz
```

清理前必须再次用 `pgrep` 确认没有遥操或转换进程，并逐项解析目标的绝对路径。本文只记录
分类，不授权自动删除任何文件。

## 3. 启动前准备

### 3.1 人员和机械安全

1. 机器人旁必须有人值守急停，首次动作时扶稳机器人。
2. 清空双臂、夹爪和身体周围的碰撞物。
3. 确认机器人处于稳定站立状态，电量足够。
4. PICO 和两个 Controller 电量足够，Controller 能正常追踪。
5. Controller 放在舒适的初始位置后再启动跟随。

### 3.2 相机检查与调焦

头部是一个 `1280x480` 双目 UVC 流，保存时拆为两幅 `640x480` 图像；两只腕部
相机各输出一幅 `640x480` 图像。

采集正式数据前必须检查四路画面：

1. 清洁镜片，确认没有保护膜和外罩遮挡。
2. 在约 0.8 至 1.5 米处放置文字或棋盘格。
3. 若头部画面整体虚焦，分别缓慢旋转左右镜头的调焦旋钮，直到文字边缘清晰。
4. 左右镜头分别调焦，不要以其中一幅代替另一幅。
5. 固定调焦旋钮，再录制一个短 episode 检查 JPEG 原图。

提高 WebRTC 或 AV1 码率不能修复光学失焦。正式采集前必须先解决物理焦距问题。

### 3.3 登录并检查服务

从工作站登录：

```bash
ssh unitree@192.168.66.47
```

机器人上检查图像服务和是否存在旧遥操进程：

```bash
systemctl is-active teleimager
pgrep -af '[t]eleop_hand_and_arm.py' || true
```

预期图像服务输出 `active`。如果已有遥操进程，先确认它是否仍被使用；不要同时
启动两个写机器人控制 Topic 的进程。

检查图像服务器端口：

```bash
python - <<'PY'
import socket

for port in (60000, 55555, 55556, 55557):
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("192.168.123.164", port))
        print(f"[ok] 192.168.123.164:{port}")
    except OSError as exc:
        print(f"[blocked] 192.168.123.164:{port}: {exc}")
    finally:
        sock.close()
PY
```

## 4. 启动纯 Controller 遥操与录制

G1-D 没有 G1 的“运动模式/Debug 模式”切换；不需要按 `L2+R2`，程序直接使用
`rt/lowcmd`。启动前只需确认机器人安全、DDS 网卡为 `eth0`，且没有其他程序同时发送低层命令。

先设置本次数据目录、统一任务名和当前采集批次的目标。正式数据不要继续写入
`g1d_dataset_smoke`：

```bash
export G1D_DATA_ROOT=/home/unitree/g1d_dataset
export G1D_TASK_NAME=shelf_pick_bag
export G1D_TASK_GOAL='Pick up and lift the green bag from the shelf.'

mkdir -p "$G1D_DATA_ROOT"
```

`G1D_TASK_NAME` 表示整个数据集，所有圆柱和长方体保持相同。`G1D_TASK_GOAL` 表示
当前这次程序运行中指定抓取的物体；同一次运行录制的所有 episode 使用同一个 goal。

启动命令：

```bash
cd /home/unitree/zhoumy/xr_teleoperate/teleop
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate tv

PYTHONPYCACHEPREFIX=/tmp/g1d-pycache \
PYTHONPATH=.. \
python teleop_hand_and_arm.py \
  --input-mode controller \
  --control-source controller \
  --display-mode immersive \
  --arm G1_29 \
  --ee dex1_internal \
  --base-type legs \
  --network-interface eth0 \
  --img-server-ip 192.168.123.164 \
  --xr-deadman-s 0.3 \
  --record \
  --task-dir "$G1D_DATA_ROOT" \
  --task-name "$G1D_TASK_NAME" \
  --task-goal "$G1D_TASK_GOAL" \
  --task-desc "Fixed-base G1-D shelf picking with controller teleoperation and internal Dex1 grippers." \
  --task-steps "Approach; grasp; lift clear of shelf; move to target; place; release; retreat." \
  --headless
```

更换指定物体时，用 `A+X` 退出程序，只修改 goal 后重新执行同一启动命令，例如：

```bash
export G1D_TASK_GOAL='Pick the blue cuboid from the shelf and place it at the fixed target.'
```

不要修改 `G1D_TASK_NAME`。新一批会继续写入同一任务目录并自动增加 episode 编号；
每个 `data.json` 保存自己的 goal，转换器会逐 episode 将它写成 LeRobot task。

不要添加 `--dual-trigger-dex1-mode`，它与 Grip 保持锁冲突。

程序初始化完成后，在 PICO 浏览器打开：

```text
https://192.168.66.47:8012?ws=wss://192.168.66.47:8012
```

确认 PICO 中图像与 Controller 追踪均正常。未按启动键之前，机器人不应跟随。

## 5. Controller 映射

| 操作 | 功能 |
| --- | --- |
| `A` 短按 | 启动或 XR 断流后的重新使能 |
| 单独 `X` | 无动作 |
| `A+X` 同时按下 | 彻底退出程序；若正在录制则丢弃当前 episode |
| `B+Y` 短按 | 开始录制；再次按下结束并保存 |
| 左/右 Trigger | 松开 Trigger 使对应夹爪打开，按下 Trigger 使对应夹爪闭合 |
| 左/右 Grip 短按 | 独立切换对应 Dex1 的保持锁定/解除锁定 |

Grip 是切换开关，不需要一直按住。锁定后，对应 Trigger 不再改变夹爪目标，双臂
仍可继续运动。

XR 数据超过 `0.3` 秒没有更新时，硬件输出会关闭。XR 恢复后必须重新按 `A`，
机器人不会自行恢复运动。

程序启动并完成控制器初始化后，不会自动移动手臂，直接等待 `A`。按 `X` 退出遥操后，可再次按
`A` 进入遥操。单独 `X` 无动作；若按 `A+X` 时仍在录制，当前未完成的 episode 会被直接
丢弃；只有再次按 `B+Y` 正常结束的 episode 才会保存。

## 6. 连续录制多个 episode

一次程序运行可以录制多个 episode：

1. 按 `A` 启动跟随。
2. 按 `B+Y` 开始录制。
3. 完成动作后再次按 `B+Y`。
4. 等待日志出现 `Episode saved successfully`。
5. 再按 `B+Y` 开始下一段。

目录自动递增：

```text
$G1D_DATA_ROOT/$G1D_TASK_NAME/episode_0000
$G1D_DATA_ROOT/$G1D_TASK_NAME/episode_0001
$G1D_DATA_ROOT/$G1D_TASK_NAME/episode_0002
...
```

同一任务目录可以包含多个目标批次；例如前几个 episode 的 `text.goal` 可以是红色
圆柱，后几个可以是蓝色长方体。转换后 LeRobot 的 `tasks.parquet` 会保留这些不同
任务文本，训练时模型才能根据指令选择目标。

写入器拒绝复用已有 episode 目录，不会覆盖旧数据。上一段仍在异步保存时如果立即
再次按 `B+Y`，本次开始请求可能被丢弃；等待保存成功日志后再按一次。

录制过程中按 `A+X` 会丢弃当前 episode 并彻底退出。预期日志包含：

```text
A+X pressed during recording; discarding the active episode
A+X pressed; graceful process shutdown requested
```

## 7. 原始数据内容与校验

每个 episode 主要包含 `data.json` 和四路 JPEG：

```text
episode_NNNN/
├── data.json
└── colors/
```

当前机器人录制的是 `legacy` JSON；转换后仍是 16 维 state、16 维 action 和四路视频。
在同一个终端中先完成环境与路径设置：

```bash
cd /home/unitree/zhoumy/xr_teleoperate
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate unitree_lerobot

export G1D_DATA_ROOT=/home/unitree/g1d_dataset
export G1D_TASK_NAME=shelf_pick_place_mixed_v1
export LEROBOT_REPO_ID=local/g1d_dataset/shelf_pick_place_mixed_v1

export UNITREE_LEROBOT_ROOT=/home/unitree/zhoumy/xr_teleoperate
export PYTHONPATH="$UNITREE_LEROBOT_ROOT:$UNITREE_LEROBOT_ROOT/lerobot/src:${PYTHONPATH:-}"
```

`PYTHONPATH` 必须保留：当前 Conda 环境的 editable 安装仍指向已经不存在的旧目录。
随后快速校验全部原始 episode：

```bash
python - <<'PY'
from pathlib import Path
import json
import cv2
import os

root = Path(os.environ.get("G1D_DATA_ROOT", "/home/unitree/g1d_dataset"))
task = os.environ.get("G1D_TASK_NAME", "pick_place_controller_dex1")
episodes = sorted((root / task).glob("episode_*"))
assert episodes, f"No episodes found under {root / task}"

for episode in episodes:
    doc = json.loads((episode / "data.json").read_text(encoding="utf-8"))
    frames = doc.get("data", [])
    assert frames, f"No frames in {episode}"
    camera_keys = sorted(frames[0].get("colors", {}))
    assert len(camera_keys) == 4, (episode, camera_keys)

    for index, frame in enumerate(frames):
        states, actions = frame["states"], frame["actions"]
        state_dims = [len(states[k]["qpos"]) for k in
                      ("left_arm", "right_arm", "left_ee", "right_ee")]
        action_dims = [len(actions[k]["qpos"]) for k in
                       ("left_arm", "right_arm", "left_ee", "right_ee")]
        assert state_dims == [7, 7, 1, 1], (episode, index, state_dims)
        assert action_dims == [7, 7, 1, 1], (episode, index, action_dims)
        assert sorted(frame["colors"]) == camera_keys, (episode, index)
        for relative_path in frame["colors"].values():
            assert (episode / relative_path).is_file(), episode / relative_path

    first_image = episode / frames[0]["colors"][camera_keys[0]]
    assert cv2.imread(str(first_image)) is not None, first_image
    print(episode.name, "frames=", len(frames),
          "duration_s=", round(len(frames) / 30, 2),
          "cameras=", camera_keys)

print("[ok] all raw episodes are structurally valid")
PY
```

## 8. 转换为 LeRobot v3

先确认两个 Python 包都从当前源码目录导入：

```bash
python - <<'PY'
import lerobot
from data_processing.constants import ROBOT_CONFIGS
print("lerobot:", lerobot.__file__)
assert "Unitree_G1_Dex1" in ROBOT_CONFIGS
print("[ok] converter imports are ready")
PY
```

转换只针对一个任务目录，输入和输出一一对应。同名 LeRobot 输出会被覆盖，
原始 JSON/JPEG 不会被修改。这里的 `repo-id` 会对应到：

```text
/home/unitree/.cache/huggingface/lerobot/local/g1d_dataset/shelf_pick_place_mixed_v1
```

检查目标后执行转换：

```bash
export RAW_TASK_DIR="$G1D_DATA_ROOT/$G1D_TASK_NAME"
export DATASET_ROOT="/home/unitree/.cache/huggingface/lerobot/$LEROBOT_REPO_ID"
python data_processing/convert_unitree_json_to_lerobot.py \
  --raw-dir "$RAW_TASK_DIR" \
  --repo-id "$LEROBOT_REPO_ID" \
  --robot-type Unitree_G1_Dex1 \
  --no-push-to-hub \
  --mode video
```

`torchcodec` 不可用并回退到 `pyav` 是当前 ARM 平台上的已知警告，不代表转换失败。

## 9. 查看和回读 LeRobot 数据

转换结束后直接回读。仍在同一终端中执行：

```bash
find "$DATASET_ROOT" -maxdepth 5 -type f | sort
python - <<'PY'
import os
import json
import torch
import torch._dynamo.utils
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

repo_id = os.environ["LEROBOT_REPO_ID"]
root = HF_LEROBOT_HOME / repo_id
dataset = LeRobotDataset(repo_id=repo_id, root=root)
sample = dataset[0]

raw_task = Path(os.environ["G1D_DATA_ROOT"]) / os.environ["G1D_TASK_NAME"]
raw_episodes = sorted(raw_task.glob("episode_*/data.json"))
expected_frames = sum(
    len(json.loads(path.read_text(encoding="utf-8"))["data"])
    for path in raw_episodes
)

image_keys = sorted(
    key for key in sample if key.startswith("observation.images.")
)

print("root:", root)
print("frames:", len(dataset))
print("episodes:", dataset.num_episodes)
print("state shape:", tuple(sample["observation.state"].shape))
print("action shape:", tuple(sample["action"].shape))
for key in image_keys:
    print(key, tuple(sample[key].shape))

assert tuple(sample["observation.state"].shape) == (16,)
assert tuple(sample["action"].shape) == (16,)
assert len(image_keys) == 4
assert len(dataset) == expected_frames
assert dataset.num_episodes == len(raw_episodes)
print("[ok] LeRobot conversion and reload passed")
PY
```

最终必须看到：

```text
[ok] LeRobot conversion and reload passed
```

如果再次出现 `No module named 'lerobot'`，说明换了终端或遗漏了第 7 节的
`PYTHONPATH` 设置；重新执行第 7 节环境设置即可。

## 10. 在 Windows 工作站查看和备份

下载转换后的视频：

```powershell
New-Item -ItemType Directory -Force `
  D:\Code\beaver\g1d_data\lerobot | Out-Null

scp -r `
  unitree@192.168.66.47:/home/unitree/.cache/huggingface/lerobot/local/g1d_pick_place_controller_dex1_v1/videos `
  D:\Code\beaver\g1d_data\lerobot\
```

下载原始 episode：

```powershell
New-Item -ItemType Directory -Force `
  D:\Code\beaver\g1d_data\raw | Out-Null

scp -r `
  unitree@192.168.66.47:/home/unitree/g1d_dataset/pick_place_controller_dex1 `
  D:\Code\beaver\g1d_data\raw\
```

先检查四路 MP4 的画面方向、焦点、遮挡和同步，再把数据用于训练。转换成功不等于
原始数据已备份；确认工作站或存储服务器上的副本可读之前，不要删除机器人上的
原始 JSON/JPEG。

## 11. 常见故障

### `Dex3_1_Controller Failed to subscribe dds`

G1-D 内置的是 Dex1，不要使用 `--ee dex3`。使用：

```text
--ee dex1_internal
```

### 缓存或相机配置 Permission denied

当前版本应写入：

```text
/home/unitree/.cache/xr_teleoperate/
```

不要让普通用户进程向源码目录写模型缓存或相机配置。

### XR 断流后机器人不恢复

这是安全设计。确认 PICO 和 Controller 追踪恢复后，短按 `A` 重新使能。

### 头部图像模糊但腕部清晰

先分别旋转两个头部镜头的调焦旋钮进行物理调焦。码率和后处理无法恢复失焦细节。

### 转换后找不到新 episode

确认 `--raw-dir` 的目录层级是 `任务目录/episode_NNNN`。同一个 `--repo-id`
再次转换时会覆盖旧的 LeRobot 输出；转换器不会修改原始 JSON/JPEG。
