"""Print cloud actions and drive only the local hanging posture with --execute."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.app import main


if __name__ == "__main__":
    raise SystemExit(main(hanging_test=True))
