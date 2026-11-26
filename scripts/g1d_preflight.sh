#!/usr/bin/env bash

# Read-only preflight for the G1-D + Dex3 XR teleoperation path.
# This script does not instantiate a robot controller or create a DDS publisher.

set -u

physical_checks=false
if [[ "${1:-}" == "--physical" ]]; then
    physical_checks=true
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--physical]" >&2
    exit 2
fi

failures=0

ok() {
    echo "[ok] $*"
}

fail() {
    echo "[blocked] $*" >&2
    failures=$((failures + 1))
}

require_file() {
    if [[ -r "$1" ]]; then
        ok "readable: $1"
    else
        fail "missing or unreadable: $1"
    fi
}

require_port() {
    local port="$1"
    if ss -lnt | awk '{print $4}' | grep -Eq ":${port}$"; then
        ok "TCP port ${port} is listening"
    else
        fail "TCP port ${port} is not listening"
    fi
}

require_topic() {
    local topic="$1"
    if grep -Fq "topic_name: ${topic}" <<<"${dds_output}"; then
        ok "DDS topic discovered: ${topic}"
    else
        fail "DDS topic not discovered: ${topic}"
    fi
}

code_root="${G1D_XR_ROOT:-/home/unitree/zhoumy/xr_teleoperate}"
tv_python="${G1D_TV_PYTHON:-/home/unitree/miniconda3/envs/tv/bin/python}"
ddsls="${G1D_DDSLS:-/home/unitree/miniconda3/envs/tv/bin/ddsls}"

require_file "${code_root}/teleop/teleop_hand_and_arm.py"
require_file "${code_root}/scripts/probe_dex3_state.py"
require_file "/home/unitree/.config/xr_teleoperate/cert.pem"
require_file "/home/unitree/.config/xr_teleoperate/key.pem"

if systemctl is-active --quiet teleimager.service; then
    ok "teleimager.service is active"
else
    fail "teleimager.service is not active"
fi

for port in 60000 55555 55556 55557 60001 60002 60003 7000; do
    require_port "${port}"
done

if ip -4 -o addr show dev eth0 | grep -Fq "192.168.123.164/"; then
    ok "robot DDS interface eth0 has 192.168.123.164"
else
    fail "eth0 does not have expected address 192.168.123.164"
fi

if ip -4 -o addr show dev wlan0 | grep -Fq "192.168.66.47/"; then
    ok "PICO-facing wlan0 has 192.168.66.47"
else
    fail "wlan0 does not have expected address 192.168.66.47"
fi

competing_processes="$(
    pgrep -af '[t]eleop_hand_and_arm.py|[e]val_g1.py|[r]eplay_robot.py|[t]est_g1_dex1_internal.py' || true
)"
if [[ -z "${competing_processes}" ]]; then
    ok "no known competing teleoperation process is running"
else
    fail "known competing teleoperation process is running"
    echo "${competing_processes}" >&2
fi

if [[ -x "${tv_python}" ]] && "${tv_python}" -c 'import cv2, pinocchio, unitree_sdk2py, zmq' >/dev/null 2>&1; then
    ok "tv Python environment imports required runtime modules"
else
    fail "tv Python environment or required imports are unavailable"
fi

dds_output=""
if [[ -x "${ddsls}" ]]; then
    dds_output="$(timeout 8 "${ddsls}" -a -r 3 2>&1)"
else
    fail "ddsls is not executable: ${ddsls}"
fi

for topic in \
    rt/lowstate \
    rt/hispeed_state \
    rt/cmd_hispeed \
    rt/slamware_ros_sdk_server_node/odom \
    rt/cmd_vel_no_limit \
    rt/waistpitchstate \
    rt/lf/dex3/left/state \
    rt/dex3/left/cmd \
    rt/lf/dex3/right/state \
    rt/dex3/right/cmd; do
    require_topic "${topic}"
done

dex3_probe_output="$(
    timeout 8 "${tv_python}" "${code_root}/scripts/probe_dex3_state.py" \
        --network-interface eth0 --timeout-s 5 2>&1
)"
if [[ $? -eq 0 ]]; then
    echo "${dex3_probe_output}"
else
    echo "${dex3_probe_output}" >&2
    fail "Dex3 topics exist but live left/right state samples are unavailable"
fi

if ${physical_checks}; then
    for readiness in \
        G1D_SUPPORT_READY \
        G1D_SAFETY_OPERATOR_READY \
        G1D_PHYSICAL_ESTOP_READY \
        G1D_CLEAR_ZONE_READY \
        G1D_PICO_READY; do
        if [[ "${!readiness:-0}" == "1" ]]; then
            ok "${readiness} confirmed"
        else
            fail "set ${readiness}=1 only after the physical condition is verified"
        fi
    done
fi

if (( failures > 0 )); then
    echo "[blocked] G1-D preflight failed with ${failures} issue(s); no robot writer was started" >&2
    exit 1
fi

echo "[ok] G1-D preflight passed; no robot writer was started"
