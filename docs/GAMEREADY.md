# Game-ready cleanup pipeline

Turns a dense TRELLIS.2 output (`output_*_1024_clean.glb`) into a **game-ready**
asset: watertight, 100%-quad, field-aligned topology at ~50–100K verts, with the
source PBR textures + a baked tangent-space normal map transferred on, and the
origin at the mesh bottom-center. Lives in `gameready/`.

```bash
bash gameready/run_gameready.sh <name> <src.glb> <verts> <base_tris> <tex>
# e.g. hero settings used for the shipped assets:
bash gameready/run_gameready.sh jeep   output_jeep_1024_clean.glb   70000  160000 4096
bash gameready/run_gameready.sh kei    output_kei_1024_clean.glb    70000  160000 4096
bash gameready/run_gameready.sh house1 output_house1_1024_clean.glb 100000 200000 4096
```

Outputs to `gameready_out/<name>.{glb,obj,fbx}` (+ `_baseColor/_metallicRoughness/_normal.png`).
Quad OBJ/FBX are the DCC deliverables (glTF stores triangles only, so the GLB is
triangulated for engines).

## Stages
1. **`game_base.py`** `[Blender]` — join all parts → **one** voxel remesh
   (watertight + single-body by construction; per-part would leave un-weldable
   seams) → drop tiny islands → hole-fill → recalc normals. `--base-tris` is an
   intermediate budget (denser than final) that gives the retopo field resolution.
2. **`retopo.py`** `[QuadriFlow]` — `-mcf -adaptive` → **100% pure quads**,
   watertight, field-aligned edge flow, curvature-adaptive. `-f ≈ output verts`,
   so `--verts` is seeded directly (lands in ~1 pass). (InstantMeshes is a
   `--tool` fallback but leaves ~1–4% holes on closed inputs.)
3. **`finish_bake.py`** `[Blender Cycles]` — clean/hole-fill, smooth normals
   (angle 50°), UV unwrap + pack, then **one selected-to-active Cycles pass**
   bakes baseColor (DIFFUSE) + roughness (ROUGHNESS) + metallic (EMIT) + tangent
   NORMAL (+Y/OpenGL) from the original dense donor; dilate maps; **bottom-center
   origin on the FINAL mesh**; export quad OBJ + FBX + tri GLB.
4. **`validate.py`** — coordinate-welded gate: boundary %, non-manifold %, quad %,
   vert band, origin, texture coverage, normal-map variance.

## Shipped results (hero: 4K maps, single watertight shell, +Y normals)
| asset | verts (quad) | quad % | open boundary | non-manifold | origin | normal-map |
|---|---|---|---|---|---|---|
| jeep   | 64,162 | 100% | 0.000% | 0.001% | y₀=0, xz-ctr | var 0.048 |
| kei    | 89,178 | 100% | 0.000% | 0.000% | y₀=0, xz-ctr | var 0.053 |
| house1 | 92,394 | 100% | 0.000% | 0.000% | y₀=0, xz-ctr | var 0.051 |

All watertight, 100% quad, correctly-origined, PBR + normal maps transferred,
not variance-collapsed. Source→gameready reduction ≈ 8–15× (e.g. jeep 720K→64K verts).

## Key findings / decisions
- **No ray tracer / no `pip` in the venv** → a custom per-texel ray-cast baker is
  non-viable; **Blender Cycles** does all four maps in one selected-to-active pass
  (shared cage/atlas/UV → no cross-tool desync, no empty-map). A normal-map
  variance assertion guards against the historical empty-bake.
- **Blender is Z-up** (glTF Y-up ↔ Blender Z-up on import/export): bottom-center
  origin translates the *up = co[2]* axis; computed on the FINAL low-poly AABB
  (remesh/retopo shift the surface, so the source AABB is wrong).
- **Watertight comes from QuadriFlow `-mcf`**, not InstantMeshes (IM leaves
  hundreds of real holes at singularities on closed inputs).
- **Adaptivity is mild.** QuadriFlow `-adaptive` adapts in the correct direction
  (curvature↔quad-size corr ≈ −0.33) but only ~1.1× smaller quads in
  high-curvature regions — because TRELLIS surfaces are **smooth** (little
  curvature contrast to adapt to). Uniform field-aligned quads are appropriate
  (and standard) topology for these hard-surface/organic reconstructions.
- **QuadriFlow `-mcf` is slow** (min-cost-flow); feed a ~150–200K base (not the
  full 400K+) — it downsamples anyway. ~3–5 min/asset retopo + ~5–8 min 4K bake.

## Known limitations
- **UV atlas utilization varies** (kei ~79%, house1 ~38%): `smart_project`+`pack`
  makes more islands on complex shapes → looser packing. Island colors are
  correct; a better packer (or per-part UVs) would raise effective texel density.
- **Strong per-region density variation** would need feature-preserving retopo
  input (risks watertight/genus) — deliberately not done; ask if wanted.
- Single unified watertight shell (per decision): movable parts (wheels) are not
  separable. Thin features fused by the voxel base are recovered only in the
  normal map, not the silhouette.
