#!/usr/bin/env bash
# Bring up trellis2-apple on Apple Silicon via the PyTorch-MPS path.
# Mirrors the proven trellis-mac recipe: uv venv (py3.11) + torch 2.13 +
# the PATCHED native Metal stack (FixedStack64 mtlbvh, Bug-A/G mtlmesh).
#
# The native Metal packages are built from trellis-mac's already-patched
# deps/ sources (proven to compile on this exact machine/OS). o_voxel is
# built from THIS repo's o-voxel/ (trellis2-apple's own fork).
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
TMAC="/Users/waylonhuang/Documents/other/trellis-mac"

echo "=== trellis2-apple Apple-Silicon (MPS) setup ==="

if [ ! -d ".venv" ]; then
    echo "Creating venv (python3.11)..."
    uv venv .venv --python python3.11
fi
source .venv/bin/activate
PIP="uv pip install"

echo "Installing Python deps..."
$PIP torch torchvision torchaudio \
     transformers accelerate huggingface_hub safetensors \
     pillow numpy trimesh scipy tqdm easydict kornia timm einops \
     imageio imageio-ffmpeg opencv-python-headless \
     xatlas fast-simplification pygltflib
$PIP setuptools wheel pybind11
# utils3d pinned commit (reuse trellis-mac's clone)
$PIP "$TMAC/deps/utils3d"

# PyTorch MPS headers require macOS 12.0+.
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
PIP_NB="$PIP --no-build-isolation"

echo "Building native Metal stack from trellis-mac's patched deps..."
$PIP_NB "$TMAC/deps/mtlbvh"      || echo "  mtlbvh build FAILED"
$PIP_NB "$TMAC/deps/mtldiffrast" || echo "  mtldiffrast build FAILED"
$PIP_NB "$TMAC/deps/mtlmesh"     || echo "  mtlmesh build FAILED"
$PIP_NB "$TMAC/deps/mtlgemm"     || echo "  mtlgemm build FAILED"

echo "Building o_voxel from trellis2-apple/o-voxel..."
$PIP_NB "$HERE/o-voxel"          || echo "  o_voxel build FAILED"

echo "=== verifying imports ==="
python - <<'PY'
mods = ["torch","trimesh","xatlas","utils3d","mtlbvh","mtldiffrast","cumesh","flex_gemm","o_voxel"]
import importlib
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {str(e)[:120]}")
import torch
print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
PY
echo "=== setup_mac.sh done ==="
