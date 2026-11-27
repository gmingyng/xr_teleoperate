"""Bounded frame handoff from tv to one LeRobot v3 writer in its own environment."""
import fcntl
import os
from pathlib import Path
import pickle
from queue import SimpleQueue
import subprocess
import sys
from threading import BoundedSemaphore, Thread
import time

import numpy as np


class V3Writer:
    def __init__(self, task_dir, task_goal, frequency, extra_info, *, scanner=None, **unused):
        self.scanner = scanner
        self.goal = task_goal
        self.mapping = extra_info["camera_mapping"]
        if set(self.mapping.values()) != {"cam_left_high","cam_right_high","cam_left_wrist","cam_right_wrist"}:
            raise ValueError("Expected four G1-D cameras")
        if frequency != int(frequency) or frequency <= 0:
            raise ValueError("LeRobot fps must be a positive integer")
        root = Path(task_dir).expanduser().absolute()
        root.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(root.parent / ("."+root.name+".writer.lock"), "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=f"{repo}:{repo}/lerobot/src",
                   HF_HUB_OFFLINE="1", OPENBLAS_NUM_THREADS="1")
        python = os.environ.get("LEROBOT_PYTHON", str(Path.home()/"miniconda3/envs/unitree_lerobot/bin/python"))
        self.proc = subprocess.Popen([python,__file__,str(root),str(int(frequency))], env=env,
                                     stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        try:
            if self.proc.stdout.readline() != b"READY\n":
                raise RuntimeError("LeRobot writer failed to initialize; see stderr")
        except BaseException:
            self.proc.stdin.close()
            self.proc.wait()
            self.lock.close()
            raise
        self.queue = SimpleQueue()
        self.slots = BoundedSemaphore(32)
        self.start = None
        self.failed = None
        self.dropped = False
        self.thread = Thread(target=self._send, daemon=True)
        self.thread.start()

    def _send(self):
        try:
            while True:
                cmd,payload = self.queue.get()
                pickle.dump((cmd,payload),self.proc.stdin,protocol=5)
                self.proc.stdin.flush()
                if cmd=="frame":
                    self.slots.release()
                if cmd=="close":
                    break
        except Exception as exc:
            self.failed = exc
        finally:
            self.proc.stdin.close()

    def is_ready(self):
        if self.failed or self.proc.poll() is not None:
            raise RuntimeError("LeRobot writer exited; inspect stderr") from self.failed
        return self.start is None

    def create_episode(self, episode_id=None):
        if episode_id is not None:
            raise ValueError("v3 assigns consecutive episode indices")
        if not self.is_ready():
            return False
        self.start=time.monotonic_ns()
        self.dropped=False
        if self.scanner:
            self.scanner.begin(self.start)
        return True

    def get_episode_elapsed_s(self, ns):
        return (ns-self.start)/1e9

    def add_item(self, colors, states, actions, timestamps, operator, unitree=None, **unused):
        self.is_ready()
        if self.start is None or self.dropped:
            return
        if not self.slots.acquire(blocking=False):
            print("LeRobot queue full: discarding this incomplete episode",file=sys.stderr,flush=True)
            self.dropped=True
            self.queue.put(("discard",None))
            return
        ns=timestamps["monotonic_ns"]
        scan=self.scanner.frame(ns) if self.scanner else (unitree or {})
        flag=bool(scan.get("scan_success",False))
        def vector(values):
            return np.concatenate([values[k]["qpos"] for k in ("left_arm","right_arm","left_ee","right_ee")]).astype(np.float32)
        barcode=(scan.get("barcode") or "").encode("ascii")
        frame={"observation.state":np.r_[np.float32(flag),vector(states)],"action":vector(actions),
               "task":self.goal,"episode_s":np.array([timestamps["episode_s"]],np.float64),
               "scan_trigger":np.array([scan.get("scan_trigger",False)],np.uint8),
               "barcode_ascii":np.frombuffer(barcode.ljust(14,b"\0"),np.uint8).copy(),
               "clutch":np.array([operator.get(k,False) for k in ("left_clutch","right_clutch")],np.uint8)}
        for src,name in self.mapping.items():
            frame["observation.images."+name]=np.ascontiguousarray(colors[src][:,:,::-1])
        self.queue.put(("frame",frame))

    def _finish(self,command):
        if self.start is None:
            return False
        summary=self.scanner.finish() if self.scanner else None
        if not self.dropped:
            self.queue.put((command,summary))
        self.start=None
        return True

    def save_episode(self):
        return self._finish("save")

    def discard_episode(self):
        return self._finish("discard")

    def close(self):
        try:
            self.save_episode()
            self.queue.put(("close",None))
            self.thread.join()
            if self.proc.wait() or self.failed:
                raise RuntimeError("v3 writer failed; inspect stderr") from self.failed
        finally:
            self.lock.close()


def worker(root,fps):
    import json
    import cv2  # Load OpenCV in the encoder environment before torchvision.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from data_processing.constants import ROBOT_CONFIGS
    motors=list(ROBOT_CONFIGS["Unitree_G1_Dex1"].motors)
    from tempfile import TemporaryDirectory
    staging = None
    ds=None
    if root.exists():
        info=json.loads((root/"meta/info.json").read_text())
        if info["codebase_version"]!="v3.0" or info["fps"]!=fps or "barcode_ascii" not in info["features"]:
            raise ValueError("Output is not a compatible G1-D v3 recording")
        ds=LeRobotDataset(root.name,root=root,video_backend="pyav")
    # Reserve stdout for the startup handshake; library output after it goes to stderr.
    print("READY",flush=True)
    sys.stdout=sys.stderr
    count=0
    while True:
        try:
            command,frame=pickle.load(sys.stdin.buffer)
        except EOFError:
            command,frame="close",None
        if command=="frame":
            if ds is None:
                features={"observation.state":{"dtype":"float32","shape":(17,),"names":["scan_success",*motors]},
                          "action":{"dtype":"float32","shape":(16,),"names":motors}}
                for k in ("episode_s","scan_trigger","barcode_ascii","clutch"):
                    features[k]={"dtype":str(frame[k].dtype),"shape":frame[k].shape,"names":None}
                for k in frame:
                    if k.startswith("observation.images."):
                        features[k]={"dtype":"video","shape":frame[k].shape,"names":["height","width","channel"]}
                staging = TemporaryDirectory(prefix="."+root.name+"-", dir=root.parent)
                ds=LeRobotDataset.create(root.name,root=Path(staging.name)/root.name,fps=fps,features=features,robot_type="Unitree_G1_Dex1")
            ds.add_frame(frame)
            count+=1
        elif command=="save" and count:
            index=ds.meta.total_episodes
            ds.save_episode()
            if staging is not None:
                ds.finalize()
                (Path(staging.name)/root.name).rename(root)
                staging.cleanup()
                staging = None
                ds = LeRobotDataset(root.name,root=root,video_backend="pyav")
            if frame and frame.get("scan_event"):
                with (root/"meta/scan_events.jsonl").open("a") as f:
                    print(json.dumps({"episode_index":index,**frame}),file=f)
            count=0
        elif command in ("discard","close") and count:
            ds.clear_episode_buffer(delete_images=True)
            count=0
        if command=="close":
            if ds is not None:
                ds.finalize()
            if staging is not None:
                staging.cleanup()
            return


if __name__=="__main__":
    worker(Path(sys.argv[1]),int(sys.argv[2]))
