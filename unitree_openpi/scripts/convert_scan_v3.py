"""Convert one G1-D scan task's episode_*/data.json files to LeRobot v3."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_lerobot.utils.constants import ROBOT_CONFIGS


def vector(row, key):
    parts = [row[key][side]["qpos"] for side in ("left_arm", "right_arm", "left_ee", "right_ee")]
    if [len(part) for part in parts] != [7, 7, 1, 1]:
        raise ValueError(f"Unexpected {key} joint layout")
    return np.concatenate(parts).astype(np.float32)


def convert(raw, output):
    files = sorted(raw.glob("episode_*/data.json"))
    if not files or output.exists():
        raise ValueError("Expected recorded episodes and a new output directory")
    dataset = None
    motors = list(ROBOT_CONFIGS["Unitree_G1_Dex1"].motors)
    expected = {"cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"}
    for path in files:
        doc = json.loads(path.read_text())
        mapping = doc["info"]["camera_mapping"]
        fps = doc["info"]["image"]["fps"]
        if int(fps) != fps:
            raise ValueError("Expected integer frame rate")
        fps = int(fps)
        if set(mapping.values()) != expected or not doc["data"] or not doc["text"]["goal"].strip():
            raise ValueError(f"Invalid episode or camera mapping: {path}")
        seen = False
        for row in doc["data"]:
            flag = row["_unitree"]["scan_success"]
            if type(flag) is not bool or (seen and not flag):
                raise ValueError(f"Invalid latched flag in {path}, frame {row['idx']}")
            seen = flag
            frame = {"observation.state": np.r_[np.float32(flag), vector(row, "states")],
                     "action": vector(row, "actions"), "task": doc["text"]["goal"]}
            for src, name in mapping.items():
                image = cv2.imread(str(path.parent / row["colors"][src]))
                if image is None:
                    raise ValueError(f"Unreadable image at {path}, frame {row['idx']}, {src}")
                frame["observation.images." + name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if dataset is None:
                features = {"observation.state": {"dtype": "float32", "shape": (17,), "names": ["scan_success", *motors]},
                            "action": {"dtype": "float32", "shape": (16,), "names": motors}}
                for key in frame:
                    if key.startswith("observation.images."):
                        features[key] = {"dtype": "video", "shape": frame[key].shape, "names": ["height","width","channel"]}
                dataset = LeRobotDataset.create(repo_id=output.name, root=output, fps=fps, robot_type="Unitree_G1_Dex1",
                                                features=features, image_writer_threads=4)
            if fps != dataset.fps:
                raise ValueError("Mixed frame rates")
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.raw, args.output)
