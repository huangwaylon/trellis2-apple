"""
Generate a 3D mesh from a single image with TRELLIS.2 on Apple Silicon (MPS).

PyTorch-MPS path (runs the original trellis2/ code) + the native Metal
o_voxel/cumesh/mtlbvh/mtldiffrast stack for CUDA-quality watertight export.

  python generate_mps.py assets/jeep.png --pipeline-type 1024_cascade --texture-size 2048
"""
import os
import sys

# --- Backend env MUST be set before trellis2 (transitively torch) imports ---
# BiRefNet's deform_conv2d and a few ops have no MPS kernel -> fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Dense attention default is flash_attn (CUDA-only); force SDPA on MPS.
os.environ.setdefault("ATTN_BACKEND", "sdpa")
# Sparse attention: prefer the FUSED Metal flex_gemm kernel (sidesteps the
# PyTorch-MPS fused-SDPA long-sequence accuracy cliff, pytorch#179352, which
# would otherwise collapse mesh quality at 1024/1536). Falls back to sdpa if
# flex_gemm's probe fails (config.py auto-detects on Darwin).
os.environ.setdefault("SPARSE_ATTN_BACKEND", "flex_gemm_sparse_attn")
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
# The fused flex_gemm sparse-attention kernel is a serial-KV Metal kernel that
# degrades to O(N^2) wall-clock at large sequence lengths (the 1024_cascade HR
# refine hits ~18 min/step). Benchmarked: SDPA is 5-19x faster at every seqlen
# and bit-matches the fused kernel in fp32 (no pytorch#179352 cliff on torch
# 2.13). Cap the fused path at 2048 tokens; above that use the fast SDPA
# fallback (single-sequence flash path, no O(N^2) mask). Override with the env.
os.environ.setdefault("FLEX_GEMM_ATTN_MAX_SEQLEN", "2048")

import argparse
import time
import numpy as np
import torch
from PIL import Image as PILImage


def unify_winding(vertices, faces, iters=12):
    """Local majority-vote winding unifier (from trellis-mac). Flips any face
    disagreeing with the majority of its edge-neighbors; welds by position for
    the topology analysis (GLB has UV-seam-split verts) then applies flips to
    the original UV-split faces (reversing vertex order preserves per-vertex
    UVs). Finishes with a global outward flip by signed volume."""
    import trimesh
    F = np.asarray(faces).copy().astype(np.int64)
    V = np.asarray(vertices)
    nF = len(F)
    if nF == 0:
        return faces
    _, inv = np.unique(np.round(V.astype(np.float64), 6), axis=0, return_inverse=True)
    Fw = inv[F]
    flipped = np.zeros(nF, bool)
    last = -1
    for _ in range(iters):
        E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], axis=0)
        fid = np.tile(np.arange(nF), 3)
        canon = np.sort(E, axis=1)
        d = np.where(E[:, 0] < E[:, 1], 1, -1).astype(np.int64)
        uniq, invE = np.unique(canon, axis=0, return_inverse=True)
        P = np.zeros(len(uniq), np.int64); M = np.zeros(len(uniq), np.int64)
        np.add.at(P, invE, (d == 1).astype(np.int64))
        np.add.at(M, invE, (d == -1).astype(np.int64))
        flip_score = np.where(d == 1, (P[invE] - 1) - M[invE], (M[invE] - 1) - P[invE])
        fs = np.zeros(nF, np.int64); np.add.at(fs, fid, flip_score)
        flip = fs > 0
        n = int(flip.sum())
        if n == 0 or n == last:
            break
        last = n
        Fw[flip] = Fw[flip][:, ::-1]
        flipped ^= flip
    F[flipped] = F[flipped][:, ::-1]
    try:
        if trimesh.Trimesh(V, F, process=True).volume < 0:
            F = F[:, ::-1]
    except Exception:
        pass
    return F


def _watchdog_help():
    return (
        "\nERROR: decoder produced an empty mesh.\n"
        "On Apple Silicon this is almost always the macOS GPU watchdog killing a\n"
        "long Metal kernel in the SLat decoder (prints "
        "'kIOGPUCommandBufferCallbackErrorImpactingInteractivity' to stderr but\n"
        "raises no Python exception). Workarounds, cheapest first:\n"
        "  1. Run headless (close lid / unplug external displays, re-run over SSH).\n"
        "  2. MTL_CAPTURE_ENABLED=1 python generate_mps.py ...  (extends watchdog).\n"
        "  3. SPARSE_CONV_BACKEND=pytorch python generate_mps.py ...  (slower path).\n"
    )


def drop_small_components(g, min_faces):
    """Remove connected components smaller than `min_faces` (remesh floater
    debris). Welds by position first (GLB verts are UV-seam-split), so a legit
    part isn't split across seams. Preserves per-vertex UVs — trimesh remaps
    visual.uv when unreferenced verts are dropped. Returns (faces_dropped,
    n_components_before)."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    V = np.asarray(g.vertices); F = np.asarray(g.faces)
    if len(F) == 0:
        return 0, 0
    _, inv = np.unique(np.round(V.astype(np.float64), 6), axis=0, return_inverse=True)
    Fw = inv[F]; nV = int(inv.max()) + 1
    r = np.concatenate([Fw[:, 0], Fw[:, 1], Fw[:, 2]])
    c = np.concatenate([Fw[:, 1], Fw[:, 2], Fw[:, 0]])
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(nV, nV))
    n, lab = csg.connected_components(A, directed=False)
    fcomp = lab[Fw[:, 0]]
    sizes = np.bincount(fcomp, minlength=n)
    keep = sizes[fcomp] >= min_faces
    dropped = int((~keep).sum())
    if dropped:
        g.update_faces(keep)
        g.remove_unreferenced_vertices()
    return dropped, n


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 image->3D on Apple Silicon (MPS)")
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="output_3d", help="Output basename (no ext)")
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--texture-size", type=int, default=2048, choices=[512, 1024, 2048, 4096])
    ap.add_argument("--decimation-target", type=int, default=1_000_000)
    ap.add_argument("--remesh-project", type=float, default=0.0,
                    help="0=README/CUDA-example default; >0 snaps remeshed verts back to surface")
    ap.add_argument("--alpha-mode", default="opaque", choices=["opaque", "blend", "mask", "auto"],
                    help="GLB material alphaMode. Default 'opaque' (solid; matches upstream). "
                         "BLEND makes the whole mesh non-depth-writing -> renders see-through; "
                         "'auto' = BLEND only if >1%% texels transparent.")
    ap.add_argument("--no-remesh", action="store_true", help="Skip narrow-band DC remesh (NOT watertight)")
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--unify-winding", action="store_true",
                    help="Apply trellis-mac majority-vote winding pass after remesh")
    ap.add_argument("--min-component-faces", type=int, default=0,
                    help="Drop connected components smaller than N faces (remesh floater cleanup; 0=off)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--save-intermediate", default=None,
                    help="Save decoded mesh+voxels to a .pt for fast bake iteration (PR #13)")
    ap.add_argument("--load-intermediate", default=None,
                    help="Skip sampling; load decoded mesh+voxels from a .pt and bake")
    args = ap.parse_args()

    device = torch.device("mps")
    t_gen = 0.0

    if args.load_intermediate:
        print(f"Loading intermediate: {args.load_intermediate}")
        blob = torch.load(args.load_intermediate, map_location="cpu", weights_only=False)
    else:
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found"); sys.exit(1)
        print("Loading pipeline...")
        t0 = time.time()
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.to(device)
        print(f"Loaded + moved to MPS in {time.time()-t0:.0f}s")

        img = PILImage.open(args.image)
        print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})  pipeline={args.pipeline_type} seed={args.seed}")
        overrides = {"steps": args.steps} if args.steps else {}
        t0 = time.time()
        try:
            outputs = pipeline.run(
                img, seed=args.seed, pipeline_type=args.pipeline_type,
                sparse_structure_sampler_params=overrides,
                shape_slat_sampler_params=overrides,
                tex_slat_sampler_params=overrides,
            )
        except (IndexError, AssertionError) as e:
            if any(s in str(e) for s in ("non-zero size", "BVH needs at least 8 triangles")):
                print(_watchdog_help()); sys.exit(2)
            raise
        t_gen = time.time() - t0
        m = outputs[0] if isinstance(outputs, list) else outputs
        blob = {
            "vertices": m.vertices.cpu(), "faces": m.faces.cpu(),
            "attrs": m.attrs.cpu(), "coords": m.coords.cpu(),
            "layout": m.layout, "voxel_size": m.voxel_size,
        }
        if args.save_intermediate:
            torch.save(blob, args.save_intermediate)
            print(f"Saved intermediate: {args.save_intermediate}")

    verts = blob["vertices"].numpy(); faces = blob["faces"].numpy()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        print(_watchdog_help()); sys.exit(2)
    print(f"Mesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} triangles"
          + (f"  (gen {t_gen:.1f}s)" if t_gen else ""))

    glb_path = f"{args.output}.glb"
    if not args.no_texture and blob["attrs"] is not None:
        import o_voxel
        print(f"\nBaking PBR + remesh via native Metal stack (tex={args.texture_size}, "
              f"remesh={not args.no_remesh}, project={args.remesh_project})...")
        t_bake = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=blob["vertices"], faces=blob["faces"],
            attr_volume=blob["attrs"], coords=blob["coords"],
            attr_layout=blob["layout"], voxel_size=blob["voxel_size"],
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=not args.no_remesh,
            remesh_band=1,
            remesh_project=args.remesh_project,
            alpha_mode=args.alpha_mode.upper() if args.alpha_mode != "auto" else "auto",
            verbose=True,
        )
        if args.unify_winding:
            try:
                import trimesh as _tm
                geoms = list(glb.geometry.values()) if isinstance(glb, _tm.Scene) else [glb]
                for g in geoms:
                    g.faces = unify_winding(g.vertices, g.faces)
            except Exception as e:
                print(f"  (winding unify skipped: {e})")
        if args.min_component_faces > 0:
            try:
                import trimesh as _tm
                geoms = list(glb.geometry.values()) if isinstance(glb, _tm.Scene) else [glb]
                for g in geoms:
                    dropped, ncomp = drop_small_components(g, args.min_component_faces)
                    print(f"  component cleanup: dropped {dropped} faces from {ncomp} components "
                          f"(threshold <{args.min_component_faces} faces)")
            except Exception as e:
                print(f"  (component cleanup skipped: {e})")
        glb.export(glb_path)
        print(f"Saved: {glb_path}  (bake {time.time()-t_bake:.0f}s)")
    else:
        import trimesh
        trimesh.Trimesh(vertices=verts, faces=faces).export(glb_path)
        print(f"Saved (geometry only): {glb_path}")

    print(f"\nDone. Total gen {t_gen:.1f}s.")


if __name__ == "__main__":
    main()
