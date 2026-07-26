#!/usr/bin/env bash
# Game-ready cleanup pipeline: dense textured GLB -> watertight adaptive-quad
# low-poly with transferred PBR + tangent normal map, bottom-center origin.
#   bash gameready/run_gameready.sh <name> <src.glb> <verts> <base_tris> <tex>
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; SRC="$2"; VERTS="${3:-70000}"; BASE_TRIS="${4:-160000}"; TEX="${5:-4096}"
BL=/Applications/Blender.app/Contents/MacOS/Blender
PY=.venv/bin/python
W="gameready_work/$NAME"; mkdir -p "$W" gameready_out

echo "### [$NAME] Stage 1: watertight base (target ${BASE_TRIS} tris)"
$BL -b --python gameready/game_base.py -- --input "$SRC" --output "$W/base.obj" --base-tris "$BASE_TRIS" 2>&1 | grep -E "game_base\]" || true

echo "### [$NAME] Stage 2: adaptive quad retopo (target ${VERTS} verts)"
$PY gameready/retopo.py --input "$W/base.obj" --output "$W/quad.obj" --verts "$VERTS" 2>&1 | grep -E "retopo\]" || true

echo "### [$NAME] Stage 3: finish + PBR/normal bake (tex ${TEX})"
$BL -b --python gameready/finish_bake.py -- --quad "$W/quad.obj" --donor "$SRC" \
    --out-base "gameready_out/$NAME" --tex "$TEX" 2>&1 | grep -E "bake\]|finish_bake\]" || true

echo "### [$NAME] Validate"
$PY gameready/validate.py --glb "gameready_out/$NAME.glb" --obj "gameready_out/$NAME.obj" \
    --donor "$SRC" --base-png "gameready_out/${NAME}_baseColor.png" \
    --normal-png "gameready_out/${NAME}_normal.png" 2>&1 | grep -vE "Warning|warn"
echo "### [$NAME] DONE -> gameready_out/$NAME.{glb,obj,fbx} + _*.png"
