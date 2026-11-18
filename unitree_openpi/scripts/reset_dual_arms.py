"""G1-D dual-arm reset entry point."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.robot_ctrl.joint_motion import main

if __name__ == "__main__":
    raise SystemExit(main("reset"))
