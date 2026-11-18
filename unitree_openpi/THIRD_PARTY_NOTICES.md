# Sources and licenses

This package extracts reusable device infrastructure from the local
`unitree_robotskills` checkout, licensed under Apache-2.0 (see `LICENSE`). It does
not import that checkout at runtime. Original workflows and task-specific gates
are not part of this controller.

Local source checkout commit: `bc60264a660432d5541adc9286161725c1f82a9f`.

| Destination | Original source | Adaptation |
| --- | --- | --- |
| `robot_ctrl/dds.py` | `unitree_gpt6/g1d_draw/src/g1.py` | Independent imports, dynamic grippers, target cancellation, command watchdog, receive timestamps |
| `robot_ctrl/kinematics.py` | `unitree_gpt6/g1d_draw/src/kinematics.py` | Full G1-D model, continuous wheels, measured two-axis waist, configurable reference, dual-arm IK |
| `robot_ctrl/agv.py` | `unitree_gpt6/g1d_r1d_navigation/src/navi_agent/services/g1_agv_client.py` | Finite input validation and unsupported lateral velocity rejection |
| `utils/config.py`, `utils/transforms.py` | `unitree_gpt6/g1d_draw/src/config.py`, `kinematics.py` | General bridge configuration and EEF conventions |
| `utils/cameras.py` | `unitree_gpt6/g1d_draw/examples/capture_images.py`; `g1d_gomoku/src/hardware/camera_web.py` | Persistent subscriptions, RGB processing, age/skew validation |
| `utils/ownership.py` | `unitree_gpt6/g1d_draw/src/storage.py` | Standalone controller lease, including Windows mock development |

`assets/g1d/g1_d.urdf` is from Unitree's `unitree_ros` repository under BSD-3-Clause.
The exact commit and original path are in `assets/g1d/SOURCE.json`, and the original
license is retained in `assets/g1d/LICENSE`. No meshes are bundled; FK/IK and RNEA
load the numerical model without rendering geometry.

The OpenPI NumPy MessagePack wire format and WebSocket handshake follow the public
[OpenPI client](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client/src/openpi_client).
The codec here is an independent implementation, rejects object/structured dtypes,
and never loads pickle. Model observation keys and action semantics remain the
deployment's responsibility.

The Unitree SDK is an external dependency, pinned to a commit in `requirements.txt`;
its vendored copy in robotskills is not duplicated here.
