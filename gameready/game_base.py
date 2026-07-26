"""
Stage 1 [Blender headless]: dense multi-part GLB -> WATERTIGHT, single-body,
low-genus TRIANGLE base for quad retopology.

  join all parts -> ONE voxel remesh (watertight + single-body BY CONSTRUCTION)
  -> drop tiny islands -> fill residual holes -> recalc normals.

Per-part remesh is deliberately NOT used: independently voxelized parts quantize
shared seams to different grids and won't weld -> not globally watertight.

Run:
  /Applications/Blender.app/Contents/MacOS/Blender -b --python gameready/game_base.py -- \
      --input output_kei_1024_clean.glb --output gameready_work/kei/base.obj --base-tris 400000
"""
import bpy, bmesh, sys, argparse, math


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-tris", type=int, default=400000,
                   help="intermediate tri budget for the retopo feed (denser than final)")
    p.add_argument("--voxel-detail", type=int, default=300)
    p.add_argument("--adaptivity", type=float, default=0.0,
                   help="voxel remesh adaptivity 0-1 (>0 = fewer polys on flats -> denser detail)")
    return p.parse_args(a)


def largest_island_cleanup(obj, min_frac=0.02, fill=True):
    bm = bmesh.new(); bm.from_mesh(obj.data)
    seen = set(); islands = []
    for f in bm.faces:
        if f.index in seen:
            continue
        stack = [f]; comp = []; seen.add(f.index)
        while stack:
            cf = stack.pop(); comp.append(cf)
            for e in cf.edges:
                for nf in e.link_faces:
                    if nf.index not in seen:
                        seen.add(nf.index); stack.append(nf)
        islands.append(comp)
    if islands:
        biggest = max(len(c) for c in islands)
        doomed = [f for c in islands if len(c) < max(20, biggest * min_frac) for f in c]
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context="FACES")
    if fill:
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data); bm.free()


def tri_count(obj):
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    n = len(bm.faces); bm.free()
    return n


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.input)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for o in objs:
        o.select_set(True)
    if len(objs) > 1:
        bpy.ops.object.join()
    orig = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    n_hi = len(orig.data.polygons)
    dims = orig.dimensions
    maxdim = max(dims.x, dims.y, dims.z)

    # iterative voxel remesh to hit the intermediate tri budget (watertight by
    # construction; faces ~ area / voxel^2, so tune detail by sqrt).
    detail = a.voxel_detail
    best = None
    lo = None
    for _ in range(6):
        bpy.ops.object.select_all(action="DESELECT")
        orig.select_set(True); bpy.context.view_layer.objects.active = orig
        bpy.ops.object.duplicate()
        cand = bpy.context.view_layer.objects.active
        cand.data.remesh_voxel_size = maxdim / max(16, detail)
        cand.data.remesh_voxel_adaptivity = a.adaptivity
        bpy.ops.object.voxel_remesh()
        tris = len(cand.data.polygons) * 2
        print(f"[game_base]   detail={detail:.0f} -> ~{tris} tris")
        if best is None or abs(tris - a.base_tris) < abs(best[1] - a.base_tris):
            best = (detail, tris)
        if lo is not None:
            bpy.data.objects.remove(lo, do_unlink=True)
        lo = cand
        if 0.8 * a.base_tris <= tris <= 1.2 * a.base_tris:
            break
        bpy.data.objects.remove(lo, do_unlink=True); lo = None
        detail *= math.sqrt(a.base_tris / max(1, tris))
        detail = min(max(detail, 32), 700)
    if lo is None:
        bpy.ops.object.select_all(action="DESELECT")
        orig.select_set(True); bpy.context.view_layer.objects.active = orig
        bpy.ops.object.duplicate(); lo = bpy.context.view_layer.objects.active
        lo.data.remesh_voxel_size = maxdim / max(16, best[0])
        lo.data.remesh_voxel_adaptivity = a.adaptivity
        bpy.ops.object.voxel_remesh()

    bpy.data.objects.remove(orig, do_unlink=True)
    me = lo.data
    bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    largest_island_cleanup(lo, fill=True)
    n_lo = tri_count(lo)
    print(f"[game_base] hi={n_hi} tris -> watertight base={n_lo} tris (target {a.base_tris})")

    bpy.ops.object.select_all(action="DESELECT")
    lo.select_set(True); bpy.context.view_layer.objects.active = lo
    bpy.ops.wm.obj_export(filepath=a.output, export_selected_objects=True,
                          export_materials=False, export_uv=False, export_normals=True)
    print(f"[game_base] wrote {a.output}")


main()
