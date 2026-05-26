#!/bin/bash
# start_viewer_novnc.sh
# Start Xvfb + x11vnc + noVNC and launch the RTR-GS Pygame viewer.
#
# Prerequisites (see README.md "Interactive Viewer"):
#   - Xvfb (system-installed, usually already present)
#   - x11vnc (extract to ~/tools/x11vnc/)
#   - noVNC (clone to ~/tools/noVNC/)
#
# Usage:
#   bash scripts/start_viewer_novnc.sh
#
# Then open http://<server_ip>:6080/vnc.html in a browser.
# For security, use an SSH tunnel:
#   ssh -L 6080:localhost:6080 user@server_ip
#   Then open http://localhost:6080/vnc.html

set -e

# ─── Viewer arguments (edit these) ────────────────────────────────────────────
CHECKPOINT="lab_output/360Roam/base_blender/stage2/checkpoint/chkpnt40000.pth"
OCCLUSION_PATH="lab_output/360Roam/base_blender/stage1/checkpoint/occlusion_volumes.pth"
# ENVMAP_PATH="./data/env_maps/TCom_ColorfulAlley_colorful_alley_2K_hdri_sphere.exr"
ENVMAP_PATH=""
# Optional: path to scene directory (provides camera data — initial view defaults to first camera)
SOURCE_PATH="data/360Roam/base_blender"
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
# optional: --transform_path ...
EXTRA_ARGS=""

# ─── Paths ────────────────────────────────────────────────────────────────────
XVFB_BIN="/usr/bin/Xvfb"
X11VNC_BIN="$HOME/tools/x11vnc/usr/bin/x11vnc"
NOVNC_DIR="$HOME/tools/noVNC"

VNC_PORT=5900
NOVNC_PORT=6080
DISPLAY_NUM=99
DISPLAY=:${DISPLAY_NUM}

# ─── Dependency check ─────────────────────────────────────────────────────────
missing=0
if [ ! -x "$XVFB_BIN" ]; then
    echo "[ERROR] Xvfb not found at $XVFB_BIN"
    echo "        Please install Xvfb (ask your administrator, or try: sudo apt install xvfb)"
    missing=1
fi
if [ ! -x "$X11VNC_BIN" ]; then
    echo "[ERROR] x11vnc not found at $X11VNC_BIN"
    echo "        See README.md → Interactive Viewer section for installation instructions."
    missing=1
fi
if [ ! -d "$NOVNC_DIR" ]; then
    echo "[ERROR] noVNC not found at $NOVNC_DIR"
    echo "        See README.md → Interactive Viewer section for installation instructions."
    missing=1
fi
if [ "$missing" -eq 1 ]; then
    exit 1
fi

# ─── Cleanup function ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $XVFB_PID 2>/dev/null || true
    kill $X11VNC_PID 2>/dev/null || true
    kill $NOVNC_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ─── Start Xvfb (virtual framebuffer) ────────────────────────────────────────
echo "Starting Xvfb on display ${DISPLAY}..."
$XVFB_BIN ${DISPLAY} -screen 0 ${IMAGE_WIDTH}x${IMAGE_HEIGHT}x24 &
XVFB_PID=$!
sleep 1

export DISPLAY

# ─── Start x11vnc ────────────────────────────────────────────────────────────
echo "Starting x11vnc on port ${VNC_PORT}..."
X11VNC_LIBDIR="$HOME/tools/x11vnc/usr/lib/x86_64-linux-gnu"
LD_LIBRARY_PATH="${X11VNC_LIBDIR}:${LD_LIBRARY_PATH}" \
    $X11VNC_BIN -display ${DISPLAY} -forever -nopw -rfbport ${VNC_PORT} -quiet &
X11VNC_PID=$!
sleep 1

# ─── Start noVNC ─────────────────────────────────────────────────────────────
echo "Starting noVNC on port ${NOVNC_PORT}..."
${NOVNC_DIR}/utils/novnc_proxy --vnc localhost:${VNC_PORT} --listen ${NOVNC_PORT} &
NOVNC_PID=$!
sleep 2

# ─── Print connection info ──────────────────────────────────────────────────
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "============================================"
echo "  Viewer is ready!"
echo ""
echo "  Local browser:"
echo "    http://${HOST_IP}:${NOVNC_PORT}/vnc.html"
echo ""
echo "  SSH tunnel (recommended for remote):"
echo "    ssh -L ${NOVNC_PORT}:localhost:${NOVNC_PORT} $(whoami)@${HOST_IP}"
echo "    Then open http://localhost:${NOVNC_PORT}/vnc.html"
echo "============================================"
echo ""

# ─── Activate conda & launch viewer ──────────────────────────────────────────
cd "$(dirname "$0")/.."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate odgs-rtr
fi

echo "Launching viewer..."
python viewer_pygame.py \
    -c "$CHECKPOINT" \
    --occlusion_path "$OCCLUSION_PATH" \
    ${ENVMAP_PATH:+--envmap_path "$ENVMAP_PATH"} \
    --image_width "$IMAGE_WIDTH" \
    --image_height "$IMAGE_HEIGHT" \
    ${SOURCE_PATH:+-s "$SOURCE_PATH"} \
    $EXTRA_ARGS
