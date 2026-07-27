---
name: trellis2-highpoly
description: Generate a HIGH-POLY textured, watertight 3D mesh (GLB) from a single image using the trellis2-apple PyTorch-MPS inference pipeline (the original trellis2 pipeline on Apple Silicon). Use when the user gives an image path and wants image-to-3D / trellis2 inference / a high-poly mesh. This runs ONLY inference + native texture bake — it does NOT run the game-ready quad-retopo/cleanup pipeline.
---

# trellis2-apple — high-poly inference

Runs the original TRELLIS.2 pipeline on Apple Silicon (MPS) to turn one image into
a high-poly, watertight, PBR-textured GLB. **Do NOT run anything under `gameready/`**
— this skill is inference only.

Repo: `/Users/waylonhuang/Documents/other/trellis2-apple`

## Steps

1. **Get the image path** from the user's request (the argument to this skill). If
   it's relative, resolve it (they usually mean a file under the repo's `assets/`
   or an absolute path). Confirm the file exists; if not, stop and ask.

2. **Pick an output basename.** Default: `output_<imagestem>_highpoly` in the repo
   root (e.g. `assets/T.png` → `output_T_highpoly`). Honor an explicit output name
   if the user gave one.

3. **Run inference** from the repo (this is the only command this skill runs). It
   takes ~13 min at `1024_cascade` on an M5 Max, so run it in the background and
   report progress:

   ```bash
   cd /Users/waylonhuang/Documents/other/trellis2-apple
   source .venv/bin/activate
   HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 python generate_mps.py "<IMAGE_PATH>" \
       --pipeline-type 1024_cascade \
       --texture-size 2048 \
       --alpha-mode opaque \
       --min-component-faces 50 \
       --output <OUTPUT_BASENAME>
   ```

   - `--pipeline-type 1024_cascade` is the model's default / best quality. Use
     `512` only if the user explicitly asks for a fast preview (~3× faster, lower
     detail).
   - `--alpha-mode opaque` keeps solid meshes solid (see docs/APPLE_SILICON.md §3.3).
     Use `blend` only if the user wants transparent glass.
   - `--min-component-faces 50` culls tiny remesh floaters. Drop it (set 0) if the
     user wants the rawest output.
   - Add `--save-intermediate /tmp/<name>.pt` if the user may want to re-bake at a
     different texture size later (re-bake with `--load-intermediate`, ~4–6.5 min).

4. **Report the result.** The only output file is `<OUTPUT_BASENAME>.glb`
   (triangulated, textured, watertight) — `generate_mps.py` does **not** write an
   `.obj`. Optionally verify quality:

   ```bash
   python assess_mesh.py <OUTPUT_BASENAME>.glb
   ```

   Report vertex/triangle count, open-boundary % (should be ~0), and the output path.

## Do NOT
- Do not run `gameready/` (game_base.py / retopo.py / finish_bake.py /
  run_gameready.sh) or any quad-retopo / decimation / cleanup. This skill delivers
  the high-poly mesh only.
- Do not change the compute path to MLX — this is the PyTorch-MPS path.

## If generation fails with an empty mesh
That's the macOS GPU watchdog killing a long Metal kernel (see the message
`generate_mps.py` prints). Suggest: run headless (lid closed / over SSH), or
prefix `MTL_CAPTURE_ENABLED=1`.
