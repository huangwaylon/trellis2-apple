"""
Validation gate for game-ready outputs. Run in the venv.
  ./.venv/bin/python gameready/validate.py --glb gameready_out/kei.glb \
      --obj gameready_out/kei.obj --donor output_kei_1024_clean.glb
"""
import argparse, numpy as np, trimesh
from PIL import Image


def welded_boundary_pct(V, F):
    _, inv = np.unique(np.round(V.astype(np.float64), 6), axis=0, return_inverse=True)
    Fw = inv[F]
    E = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1)
    u, c = np.unique(E, axis=0, return_counts=True)
    ncomp_boundary = 100 * (c == 1).sum() / len(u)
    nonman = 100 * (c > 2).sum() / len(u)
    return ncomp_boundary, nonman, inv.max() + 1


def quad_pct(obj):
    q = t = n = 0
    with open(obj) as fh:
        for ln in fh:
            if ln.startswith("f "):
                k = len(ln.split()) - 1
                if k == 4: q += 1
                elif k == 3: t += 1
                else: n += 1
    return 100 * q / max(1, q + t + n), q, t, n


def img_stats(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--obj")
    p.add_argument("--donor")
    p.add_argument("--base-png")
    p.add_argument("--normal-png")
    a = p.parse_args()

    g = list(trimesh.load(a.glb, process=False).geometry.values())[0]
    V = np.asarray(g.vertices); F = np.asarray(g.faces)
    b, nm, nVw = welded_boundary_pct(V, F)
    ext = V.max(0) - V.min(0)
    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    cx = (V[:, 0].min() + V[:, 0].max()) / 2
    cz = (V[:, 2].min() + V[:, 2].max()) / 2

    print(f"\n=== {a.glb} ===")
    print(f"  verts (welded):     {nVw:,}   tris: {len(F):,}")
    print(f"  open boundary:      {b:.3f}%   non-manifold: {nm:.3f}%   {'WATERTIGHT' if b<0.5 else 'HAS HOLES'}")
    print(f"  bbox extent:        {np.round(ext,4)}")
    print(f"  ORIGIN bottom-ctr:  y_min={ymin:+.4f} (want ~0)  x_ctr={cx:+.4f} z_ctr={cz:+.4f} (want ~0)")
    ok_origin = abs(ymin) < 1e-3 and abs(cx) < 1e-3 and abs(cz) < 1e-3
    print(f"                      -> {'OK' if ok_origin else 'OFF'}")
    try:
        m = g.visual.material
        def has(x):
            t = getattr(m, x, None); return None if t is None else np.asarray(t).shape
        print(f"  material: base={has('baseColorTexture')} MR={has('metallicRoughnessTexture')} "
              f"normal={has('normalTexture')} alpha={getattr(m,'alphaMode',None)} 2sided={getattr(m,'doubleSided',None)}")
    except Exception as e:
        print(f"  material: n/a ({e})")
    if a.obj:
        qp, q, t, n = quad_pct(a.obj)
        print(f"  quad%: {qp:.1f}%  ({q} quads, {t} tris, {n} ngons)")
    if a.base_png:
        bc = img_stats(a.base_png)
        nonblack = 100 * (bc.sum(-1) > 0.02).mean()
        print(f"  baseColor: mean_rgb={np.round(bc.reshape(-1,3).mean(0),3)} coverage(non-black)={nonblack:.1f}%")
        if a.donor:
            dm = list(trimesh.load(a.donor, process=False).geometry.values())[0].visual.material
            dbc = np.asarray(dm.baseColorTexture.convert("RGB")).astype(np.float32)/255
            print(f"  donor baseColor mean_rgb={np.round(dbc.reshape(-1,3).mean(0),3)} "
                  f"(compare hue/brightness to baked)")
    if a.normal_png:
        nrm = img_stats(a.normal_png)
        var = nrm.reshape(-1, 3).var(0).sum()
        print(f"  normal map: mean={np.round(nrm.reshape(-1,3).mean(0),3)} variance={var:.5f} "
              f"({'has detail' if var>1e-4 else 'FLAT/EMPTY'})")


if __name__ == "__main__":
    main()
