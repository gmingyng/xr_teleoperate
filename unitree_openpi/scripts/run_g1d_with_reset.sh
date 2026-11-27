#!/usr/bin/env bash
# Run from the robot's activated tv environment.
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
usage() {
    echo "Usage: $0 [--execute | --mock] [run_g1d_client.py options]"
    echo "Default: real robot (--execute). Use --mock for simulation."
    echo "Hold reset; Enter starts inference. Ctrl+C during inference returns to reset; Ctrl+C while holding exits."
}
mode=--execute
case ${1:-} in
    -h|--help) usage; exec python "$scripts/run_g1d_client.py" --help ;;
    --execute|--mock) mode=$1; shift ;;
esac
args=(
    --url ws://192.168.66.200:8000
    --barcode-scanner
    --rtc
    "$@"
)
for arg in "$@"; do
    case $arg in
        --mock|--execute) echo "Specify the operation mode only once." >&2; exit 2 ;;
    esac
done
exec python "$scripts/run_g1d_client.py" "$mode" --reset-first "${args[@]}"
