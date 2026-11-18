"""Send G1-D observations and execute cloud policy actions."""

from pathlib import Path
import sys

# Direct execution also works after copying this package outside the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.app import main


if __name__ == "__main__":
    raise SystemExit(main())
