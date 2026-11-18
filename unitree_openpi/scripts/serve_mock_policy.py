"""Start a local mock policy server for the G1-D WebSocket client."""

from pathlib import Path
import sys

# Direct execution also works after copying this package outside the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unitree_openpi.mock_policy_server import main


if __name__ == "__main__":
    raise SystemExit(main())
