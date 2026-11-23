"""Inspect absolute joint predictions. Pass --execute to apply them."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.scripts._client import launch


def main(argv=None):
    return launch(argv, prompt="test", control_waist=False)


if __name__ == "__main__":
    raise SystemExit(main())
