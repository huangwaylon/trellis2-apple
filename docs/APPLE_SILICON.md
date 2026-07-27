# TRELLIS.2 on Apple Silicon (PyTorch-MPS path)

Runs the **original `trellis2/` pipeline on MPS** + the native Metal
`o_voxel`/`cumesh`/`mtlbvh`/`mtldiffrast` stack. Produces near-watertight,
CUDA-comparable textured GLBs. This is the recommended path for best mesh
quality; the `mlx_backend/` (see README) is the alternative compute path.

Verified on M5 Max / 64 GB / macOS 26 / torch 2.13.

---

## TL;DR

```bash
bash setup_mac.sh                       # uv venv (py3.11) + native Metal stack
source .venv/bin/activate
python generate_mps.py assets/jeep.png --pipeline-type 1024_cascade \
       --texture-size 2048 --min-component-faces 50 --output out
python assess_mesh.py out.glb           # watertightness metrics
```

`1024_cascade` (model default) ≈ **11–13 min** on M5 Max (8 min sample+decode,
4–6.5 min bake). `512` is a ~3× faster preview and already beats prior ports.
Bake time scales with remesh complexity: a single prop (jeep) bakes in ~4 min,
a dense scene (mart2, 31.2 M-face remesh) took 6.5 min.

---

## 1. Which backend

| | PyTorch-MPS (this doc) | MLX (`mlx_backend/`, `app_mlx.py`) |
|---|---|---|
| Runs | original `trellis2/` on MPS | reimplemented DiTs/VAE in MLX |
| Fidelity | highest (least divergence) | reimplementation risk |
| Mesh export | shared native `o_voxel` stack | shared |

We chose PyTorch-MPS: "rely on the original implementation as much as possible"
+ lowest risk of numeric divergence.

## 2. Setup (`setup_mac.sh`)

- uv venv, **Python 3.11**, **torch 2.13** (`requirements_macos.txt` pins
  ≥2.11: `mtlgemm` calls `at::mps::dispatch_sync_with_rethrow`, promoted to
  `at::mps::` in 2.11).
- Native Metal packages built `--no-build-isolation` (they need torch at build
  time) with `MACOSX_DEPLOYMENT_TARGET=12.0` (MPS headers need 12.0+).
  `mtlbvh`/`mtlmesh` **must carry the FixedStack64 + Bug-A/G shader fixes**
  (see §3.6); we build from a source tree that already has them.
- `o_voxel` built **editable** from this repo's `o-voxel/` (carries our alpha
  fix, §3.3).
- Weights: `microsoft/TRELLIS.2-4B` (~15 GB) + DINOv3 + RMBG-2.0, HF cache.
  First download: `HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120`.

## 3. Issues discovered & resolutions

### 3.1 Sparse attention was O(N²)-slow at high resolution — the key fix
- **Symptom:** `1024_cascade` HR refine ran at **~1096 s/step (~3.3 h total)**;
  GPU pinned at 99% (genuinely compute-bound, not hung).
- **Cause:** the fused Metal `flex_gemm_sparse_attn` kernel is serial over the
  KV dimension. Upstream commit `afaaf33` removed its seqlen cap
  (`FLEX_GEMM_ATTN_MAX_SEQLEN` defaults 0 → always fused), so the HR pass's
  huge sequence ran the serial kernel.
- **Measured** (`benchmark_attn.py`, M5 Max, fp16): SDPA is **5–19× faster at
  every seqlen** and **bit-identical to the fused kernel in fp32**
  (`max|Δ| = 0.00000` through N=8192; SDPA scales to N=21504 in 0.13 s). No
  pytorch#179352 accuracy cliff on torch 2.13.
- **Fix (two parts):**
  1. `generate_mps.py` sets `FLEX_GEMM_ATTN_MAX_SEQLEN=2048` → sequences above
     that use SDPA.
  2. `full_attn.py` fallback rewritten: for a single packed sequence (the HR
     inference case) call flash SDPA directly with **no `[B,Lq,Lkv]` mask** —
     the old masked-padded fallback materialised an O(N²) mask that OOMs at
     1024³ token counts.
- **Result:** HR pass **1096 → 18 s/step (~60×)**; full `1024_cascade` ~13 min.
  Geometry not collapsed (fills AABB; no Bug-D variance collapse).

### 3.2 DINOv3 gated repo + block nesting
`facebook/dinov3-vitl16-pretrain-lvd1689m` is gated (403). Redirect to the
ungated byte-identical `camenduru/…` mirror (env `TRELLIS_DINOV3_REPO`). Also,
transformers 5.14 nests blocks under `.model.layer` (top-level exposes
`embeddings`/`rope_embeddings`/`model`/`norm`), so `extract_features` uses
`self.model.model.layer`. — `trellis2/modules/image_feature_extractor.py`

### 3.3 See-through mesh / "inconsistent normals" — was the BLEND material
- **Symptom:** orbiting the kei car, the surface reads as see-through — you look
  *into* the interior through the body, as if normals were flipped/random.
- **Not the geometry:** kei's winding disagreement is 0.019% (as clean as
  jeep's), vertex normals are present, and a from-outside ray test hits a
  **front-facing** triangle first on **100%** of rays (0% back-facing). The
  exterior shell is consistently outward-oriented.
- **Cause:** the model predicts ~4% low-alpha voxels on kei (its glass), so the
  auto-detector set the single PBR material to `alphaMode='BLEND'`. In glTF a
  BLEND material **does not write the depth buffer** — so the *entire* surface
  (including the 96% fully-opaque body) stops occluding, and back/interior
  geometry shows through. jeep stayed OPAQUE → writes depth → solid.
- **Fix:** default `alpha_mode='OPAQUE'` (matches upstream TRELLIS.2, which
  keeps alpha in the texture but inactive). `to_glb` gains an `alpha_mode`
  param (`OPAQUE`|`BLEND`|`MASK`|`auto`); `generate_mps.py --alpha-mode`
  defaults to `opaque`. Result: kei renders solid, geometry unchanged.
- Related bf16 note: even the old "any texel <250 → BLEND" heuristic (upstream
  PR #1) wrongly flipped solid meshes to BLEND from drift; `auto` now needs
  >1% of texels <128. But BLEND itself is the see-through trigger, so OPAQUE is
  the default regardless. — `o-voxel/o_voxel/postprocess.py`

### 3.4 `--load-intermediate` broken on torch ≥2.6
`torch.load` now defaults `weights_only=True`, rejecting the layout dict's
`slice` objects. Load our own trusted blob with `weights_only=False`.

### 3.5 Misc device fixes / expected fallbacks
- `Pipeline.cuda()` → mps when available (`pipelines/base.py`).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` required: BiRefNet `deform_conv2d` and
  `aten::segment_reduce` (sampling) have no MPS kernel → CPU fallback
  (a warning, expected; segment_reduce adds a small per-step CPU cost).
- `ATTN_BACKEND=sdpa` for the **dense** attention path (default `flash_attn`
  is CUDA-only). Keep `SPARSE_ATTN_BACKEND=flex_gemm_sparse_attn` for sparse.

### 3.6 Scale risks that held (did NOT need mitigation)
trellis-mac disables decode-time `cumesh` and pre-decimates to 300 K before the
bake, fearing segfaults/watchdog on large meshes. On this stack (macOS 26,
patched native libs) both survived at full scale:
- decode-time `fill_holes` on 8–10 M-vertex meshes: OK.
- native narrow-band DC remesh on **21.3 M faces** (FixedStack64 BVH): OK.
If a run dies with an empty mesh (`BVH needs at least 8 triangles` /
`non-zero size`), it's the **GPU watchdog** killing a long Metal kernel — run
headless (lid closed / over SSH) or `MTL_CAPTURE_ENABLED=1`.

## 4. Mesh quality

**Methodology** (`assess_mesh.py`): **weld vertices by position** (round 1e-6)
*before* counting — trimesh's default merge respects UV seams and reports a
healthy GLB as a sieve. Metrics: open-boundary %, non-manifold %, winding
disagreement %, connected components, largest-component fraction.

Final deliverables (`1024_cascade`, tex 2048, `remesh_project=0`, `cc<50`):

| | faces | open boundary | non-manifold | winding | comps | alpha |
|---|---|---|---|---|---|---|
| jeep | 932 K | **0.011 %** | 0.247 % | 0.020 % | 124 | OPAQUE |
| kei  | 981 K | 0.126 % | 0.168 % | 0.019 % | 106 | OPAQUE |
| mart2 | 964 K | 0.046 % | 0.465 % | 0.065 % | 256 | OPAQUE |

Reference (trellis-mac jeep @1024): ~0.15 % boundary, ~0.6 % non-manifold.
kei's higher boundary is genuine open thin-surface geometry (glass), not a
defect; it renders solid in OPAQUE mode (§3.3). mart2 (a whole storefront, not a
single prop) is the busiest input tried: its 31.2 M-face remesh leaves more
non-manifold edges and a longer floater tail (911 sub-50-face components culled),
but boundary stays near 0.

## 5. Experiments

### 5.1 `remesh_project` (surface snap-back) — keep **0.0**
| project | boundary% | non-manifold% | winding% | components |
|---|---|---|---|---|
| **0.0** | **0.011** | 0.250 | **0.020** | **266** |
| 0.3 | 0.012 | 0.210 | 0.055 | 420 |
| 0.9 | 0.027 | 0.376 | 0.062 | 554 |

Remesh geometry is identical across all three (21.3 M faces); `project>0`
snaps final vertices back onto the **defective raw dual-grid surface**,
reintroducing floaters + winding errors (0.9 also worsens boundary). trellis-mac
uses 0.9 only because it pre-decimates to 300 K first (different tradeoff).

### 5.2 Component cleanup (`--min-component-faces`) — use **50**
Meshes are one dominant body + a few legit large parts (wheels ≈50 K faces,
can't be merged) + a tail of tiny remesh floaters.
| threshold | jeep comps | faces dropped | boundary% |
|---|---|---|---|
| off | 266 | — | 0.011 |
| <50 | **124** | 0.28 % | 0.011 |
| <100 | 89 | 0.55 % | 0.010 |

`<50` halves the count with **zero** change to boundary/winding; `<100` starts
removing small legit detail. `drop_small_components` welds by position and
preserves per-vertex UVs.

## 6. Options considered & rejected

- **MLX backend** for compute — reimplementation divergence risk; chose
  PyTorch-MPS.
- **Keep fused attention unconditionally** (upstream `afaaf33`) — 3.3 h/image;
  capped it instead.
- **`naive` chunked-fp32 attention** (trellis-mac's cliff workaround) —
  unnecessary on torch 2.13 (no cliff) and slower than SDPA; not ported.
- **Pre-decimate to 300 K before bake** (trellis-mac) — sacrifices detail;
  full-res native remesh survives here, so we skip it.
- **`remesh_project` 0.3/0.9** — worse topology (§5.1).
- **Auto-detect alphaMode (BLEND on glass)** — BLEND disables whole-mesh
  depth-write → see-through (§3.3); default OPAQUE, BLEND opt-in.
- **`alphaMode=MASK` for kei glass** — alpha-cutout would punch holes at the
  windows (not watertight) and speckle where drift dips below the cutoff.

## 7. `generate_mps.py` reference

| flag | default | notes |
|---|---|---|
| `--pipeline-type` | `1024_cascade` | `512` / `1024` / `1024_cascade` / `1536_cascade` |
| `--texture-size` | 2048 | 512–4096 |
| `--remesh-project` | 0.0 | keep 0 (§5.1) |
| `--alpha-mode` | `opaque` | `opaque`/`blend`/`mask`/`auto`; BLEND renders see-through (§3.3) |
| `--min-component-faces` | 0 | recommend 50 (§5.2) |
| `--unify-winding` | off | majority-vote winding pass (rarely needed: <0.03%) |
| `--save-intermediate` / `--load-intermediate` | — | decouple ~10 min sample+decode from the 4–6.5 min bake |
| `--no-remesh` / `--no-texture` | — | debug |

Env (auto-set by `generate_mps.py`): `PYTORCH_ENABLE_MPS_FALLBACK=1`,
`ATTN_BACKEND=sdpa`, `SPARSE_ATTN_BACKEND=flex_gemm_sparse_attn`,
`SPARSE_CONV_BACKEND=flex_gemm`, `FLEX_GEMM_ATTN_MAX_SEQLEN=2048`.

## 8. Known limitations

- `1536_cascade` untested here (heavy; OOM/watchdog risk on ≤64 GB).
- `segment_reduce` CPU fallback adds per-step overhead (no MPS kernel upstream).
- Single-image outputs are scan-quality static props; deforming characters need
  manual retopo.
