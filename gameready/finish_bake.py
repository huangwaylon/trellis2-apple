"""
Stage 3 [Blender headless]: finish + bake. ONE Blender session so the low-poly
topology, UVs, cage and all bakes stay consistent (no cross-tool desync).

  import quad OBJ (low, active) + original textured GLB (high donor, selected)
  -> clean/holefill/quad-fill -> smooth normals (angle) -> UV unwrap
  -> Cycles selected-to-active bake: baseColor (DIFFUSE) + roughness (ROUGHNESS)
     + metallic (EMIT) + tangent NORMAL (+Y / OpenGL)
  -> pack metallicRoughness, dilate maps -> bottom-center origin on FINAL AABB
  -> export tri GLB (+3 maps) and quad OBJ + FBX.

  /Applications/Blender.app/Contents/MacOS/Blender -b --python gameready/finish_bake.py -- \
    --quad gameready_work/kei/quad.obj --donor output_kei_1024_clean.glb \
    --out-base gameready_out/kei --tex 4096
"""
import bpy, bmesh, sys, os, argparse, math, numpy as np


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--quad", required=True)
    p.add_argument("--donor", required=True)
    p.add_argument("--out-base", required=True, help="output basename (writes .glb/.obj/.fbx + _*.png)")
    p.add_argument("--tex", type=int, default=4096)
    p.add_argument("--smooth-angle", type=float, default=50.0)
    return p.parse_args(a)


def import_quad(path):
    bpy.ops.wm.obj_import(filepath=path)
    o = [o for o in bpy.context.selected_objects if o.type == "MESH"][0]
    return o


def import_donor(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    news = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    bpy.context.view_layer.objects.active = news[0]
    for o in news:
        o.select_set(True)
    if len(news) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def clean_quads(obj, eps):
    me = obj.data
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=eps)
    be = [e for e in bm.edges if e.is_boundary]
    if be:
        bmesh.ops.holes_fill(bm, edges=be, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()


def new_img(name, size, non_color):
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=non_color)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def setup_low_material(obj):
    mat = bpy.data.materials.new("baked"); mat.use_nodes = True
    obj.data.materials.clear(); obj.data.materials.append(mat)
    return mat


def add_target_node(mat, img):
    nt = mat.node_tree
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = img
    for nn in nt.nodes:
        nn.select = False
    n.select = True
    nt.nodes.active = n
    return n


def bake_pass(low, high, bake_type, img, mat, ext, ray, margin, pass_color=False,
              normal=False):
    add_target_node(mat, img)
    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bake = bpy.context.scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = ext
    bake.max_ray_distance = ray
    bake.margin = margin
    bake.margin_type = "ADJACENT_FACES"
    if bake_type == "DIFFUSE":
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.use_pass_color = True
    kw = {}
    if normal:
        kw = dict(normal_space="TANGENT",
                  normal_r="POS_X", normal_g="POS_Y", normal_b="POS_Z")  # OpenGL +Y
    bpy.ops.object.bake(type=bake_type, **kw)


def donor_emit_input(high, input_name):
    """Rewire donor material so surface = Emission(<Principled input>), for EMIT bake.
    Returns a restore() closure."""
    mat = high.data.materials[0]
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    emit = nt.nodes.new("ShaderNodeEmission")
    inp = bsdf.inputs[input_name]
    if inp.is_linked:
        nt.links.new(inp.links[0].from_socket, emit.inputs["Color"])
    else:
        v = inp.default_value
        emit.inputs["Color"].default_value = (v, v, v, 1) if not hasattr(v, "__len__") else v
    orig_from = out.inputs["Surface"].links[0].from_socket
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    def restore():
        nt.links.new(orig_from, out.inputs["Surface"])
        nt.nodes.remove(emit)
    return restore


def dilate(img, iters=16):
    S = img.size[0]
    px = np.array(img.pixels[:], dtype=np.float32).reshape(S, S, 4)
    rgb = px[..., :3]; a = px[..., 3]
    filled = a > 0.5
    for _ in range(iters):
        holes = ~filled
        if not holes.any():
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = np.roll(np.roll(filled, dy, 0), dx, 1)
            take = holes & src
            rgb[take] = np.roll(np.roll(rgb, dy, 0), dx, 1)[take]
            filled[take] = True
            holes = ~filled
    px[..., :3] = rgb; px[..., 3] = 1.0
    img.pixels[:] = px.reshape(-1)


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = 1

    low = import_quad(a.quad)
    dims = low.dimensions
    maxdim = max(dims.x, dims.y, dims.z)
    clean_quads(low, eps=1e-5 * maxdim)

    # smooth normals with crease preservation (Blender 5.2 "shade smooth by angle")
    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True); bpy.context.view_layer.objects.active = low
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(a.smooth_angle))

    # UV unwrap the final quad mesh
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.002)
    bpy.ops.uv.pack_islands(rotate=True, margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")

    high = import_donor(a.donor)

    mat = setup_low_material(low)
    S = a.tex
    ext = 0.02 * maxdim
    ray = 0.02 * maxdim
    margin = 16

    base_img = new_img("baseColor", S, non_color=False)
    rough_img = new_img("rough", S, non_color=True)
    metal_img = new_img("metal", S, non_color=True)
    normal_img = new_img("normal", S, non_color=True)

    print("[bake] baseColor (DIFFUSE)...")
    bake_pass(low, high, "DIFFUSE", base_img, mat, ext, ray, margin, pass_color=True)
    print("[bake] roughness (ROUGHNESS)...")
    bake_pass(low, high, "ROUGHNESS", rough_img, mat, ext, ray, margin)
    print("[bake] metallic (EMIT)...")
    restore = donor_emit_input(high, "Metallic")
    bake_pass(low, high, "EMIT", metal_img, mat, ext, ray, margin)
    restore()
    print("[bake] normal (TANGENT +Y)...")
    bake_pass(low, high, "NORMAL", normal_img, mat, ext, ray, margin, normal=True)

    # normal-map variance sanity (defeat the historical empty-map failure)
    npx = np.array(normal_img.pixels[:], dtype=np.float32).reshape(S, S, 4)[..., :3]
    nvar = float(npx.reshape(-1, 3).var(axis=0).sum())
    print(f"[bake] normal-map variance = {nvar:.5f} (must be > ~1e-4; flat/empty if ~0)")

    for im in (base_img, rough_img, metal_img, normal_img):
        dilate(im, 16)

    # pack glTF metallicRoughness: R=1, G=roughness, B=metallic
    mr = np.ones((S, S, 4), np.float32)
    rg = np.array(rough_img.pixels[:], np.float32).reshape(S, S, 4)[..., 0]
    mb = np.array(metal_img.pixels[:], np.float32).reshape(S, S, 4)[..., 0]
    mr[..., 0] = 0.0; mr[..., 1] = rg; mr[..., 2] = mb; mr[..., 3] = 1.0
    mr_img = new_img("metallicRoughness", S, non_color=True)
    mr_img.pixels[:] = mr.reshape(-1)

    # save PNGs
    out = a.out_base
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    def save(img, path):
        img.filepath_raw = path; img.file_format = "PNG"; img.save()
    save(base_img, f"{out}_baseColor.png")
    save(mr_img, f"{out}_metallicRoughness.png")
    save(normal_img, f"{out}_normal.png")

    # build final glTF-style material on the low-poly
    bpy.data.materials.remove(mat)
    fm = bpy.data.materials.new("gameready"); fm.use_nodes = True
    nt = fm.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    def texnode(img, non_color, x):
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = img; t.location = (x, 0)
        if non_color: t.image.colorspace_settings.name = "Non-Color"
        return t
    tb = texnode(base_img, False, -600); nt.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(texnode(mr_img, True, -600).outputs["Color"], nt.nodes.new("ShaderNodeSeparateColor").inputs[0]) if False else None
    # separate MR
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.location = (-400, -300)
    tmr = texnode(mr_img, True, -600); nt.links.new(tmr.outputs["Color"], sep.inputs[0])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    nmap = nt.nodes.new("ShaderNodeNormalMap"); nmap.location = (-300, -600)
    tn = texnode(normal_img, True, -600); nt.links.new(tn.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    low.data.materials.clear(); low.data.materials.append(fm)

    # remove donor; bottom-center origin on FINAL low-poly AABB.
    # NOTE: Blender is Z-UP (glTF Y-up is converted to Blender Z-up on import,
    # and back on export_yup). So "up" is co[2]; center co[0] and co[1].
    bpy.data.objects.remove(high, do_unlink=True)
    me = low.data
    coords = np.array([v.co[:] for v in me.vertices])
    cx = (coords[:, 0].min() + coords[:, 0].max()) / 2
    cy = (coords[:, 1].min() + coords[:, 1].max()) / 2
    zmin = coords[:, 2].min()
    for v in me.vertices:
        v.co[0] -= cx; v.co[1] -= cy; v.co[2] -= zmin
    me.update()

    bpy.ops.object.select_all(action="DESELECT")
    low.select_set(True); bpy.context.view_layer.objects.active = low
    # quad OBJ + FBX (preserve quads), tri GLB
    bpy.ops.wm.obj_export(filepath=f"{out}.obj", export_selected_objects=True,
                          export_materials=True, export_uv=True, export_normals=True,
                          export_triangulated_mesh=False)
    bpy.ops.export_scene.fbx(filepath=f"{out}.fbx", use_selection=True,
                             mesh_smooth_type="FACE", axis_forward="-Z", axis_up="Y")
    bpy.ops.export_scene.gltf(filepath=f"{out}.glb", export_format="GLB",
                              use_selection=True, export_yup=True)
    nq = sum(1 for p in me.polygons if len(p.vertices) == 4)
    print(f"[finish_bake] verts={len(me.vertices)} quads={nq}/{len(me.polygons)} "
          f"tex={S} normal_var={nvar:.5f}")
    print(f"[finish_bake] wrote {out}.glb / .obj / .fbx (+ _baseColor/_metallicRoughness/_normal .png)")


main()
