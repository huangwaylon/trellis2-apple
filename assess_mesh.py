"""
Assess mesh quality / watertightness of a GLB (or OBJ).

Uses COORDINATE-WELDED metrics (round verts to 1e-6, then count) — trimesh's
default merge respects UV/normal splits and will report a healthy GLB as a
"sieve". See trellis-mac NOTES §4 "Measurement gotchas".

  python assess_mesh.py output_3d.glb
"""
import sys
import numpy as np
import trimesh


def welded_topology(V, F, decimals=6):
    """Weld vertices by rounded position; return welded faces + maps."""
    key = np.round(V.astype(np.float64), decimals)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    Fw = inv[F]
    # drop degenerate faces created by welding
    good = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    return Fw[good], inv, int((~good).sum())


def edge_stats(Fw):
    E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], axis=0)
    E = np.sort(E, axis=1)
    uniq, counts = np.unique(E, axis=0, return_counts=True)
    n_edges = len(uniq)
    boundary = int((counts == 1).sum())      # open edges (1 incident face)
    nonmanifold = int((counts > 2).sum())     # >2 incident faces
    return n_edges, boundary, nonmanifold


def winding_disagreement(Fw):
    """% of interior manifold edges whose two faces disagree on direction."""
    E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], axis=0)
    d = np.where(E[:, 0] < E[:, 1], 1, -1)
    canon = np.sort(E, axis=1)
    uniq, inv, counts = np.unique(canon, axis=0, return_inverse=True, return_counts=True)
    dsum = np.zeros(len(uniq), np.int64)
    dcnt = np.zeros(len(uniq), np.int64)
    np.add.at(dsum, inv, d)
    np.add.at(dcnt, inv, 1)
    manifold = dcnt == 2
    if manifold.sum() == 0:
        return 0.0
    # consistent manifold edge has dsum==0 (one +1 one -1); ==±2 means disagree
    disagree = np.abs(dsum[manifold]) == 2
    return 100.0 * disagree.sum() / manifold.sum()


def connected_components(Fw, nV):
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    rows = np.concatenate([Fw[:, 0], Fw[:, 1], Fw[:, 2]])
    cols = np.concatenate([Fw[:, 1], Fw[:, 2], Fw[:, 0]])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nV, nV))
    n, _ = csg.connected_components(A, directed=False)
    # subtract isolated welded verts not used by any face
    used = np.unique(Fw)
    return n - (nV - len(used))


def assess(path):
    scene = trimesh.load(path, process=False)
    geoms = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
    print(f"\n=== {path} ===")
    print(f"geometries: {len(geoms)}")
    for gi, g in enumerate(geoms):
        V = np.asarray(g.vertices); F = np.asarray(g.faces)
        Fw, inv, ndeg = welded_topology(V, F)
        nVw = inv.max() + 1
        n_edges, boundary, nonman = edge_stats(Fw)
        b_pct = 100.0 * boundary / n_edges if n_edges else 0
        nm_pct = 100.0 * nonman / n_edges if n_edges else 0
        wind = winding_disagreement(Fw)
        try:
            ncomp = connected_components(Fw, nVw)
        except Exception:
            ncomp = "?"
        # watertight test on welded mesh
        try:
            tw = trimesh.Trimesh(V, F, process=True)
            watertight = tw.is_watertight
            vol = tw.volume
        except Exception:
            watertight, vol = "?", "?"
        print(f"\n-- geom[{gi}] --")
        print(f"  raw verts/faces:      {len(V):,} / {len(F):,}")
        print(f"  welded verts:         {nVw:,}   (degenerate faces dropped: {ndeg})")
        print(f"  open boundary edges:  {boundary:,} / {n_edges:,}  = {b_pct:.3f}%   <- 0% = watertight")
        print(f"  non-manifold edges:   {nonman:,}  = {nm_pct:.3f}%")
        print(f"  winding disagreement: {wind:.3f}%  (adjacent manifold pairs)")
        print(f"  connected components: {ncomp}")
        print(f"  trimesh watertight:   {watertight}   signed volume: {vol}")
        # material / alpha
        try:
            mat = g.visual.material
            print(f"  material: alphaMode={getattr(mat,'alphaMode',None)} doubleSided={getattr(mat,'doubleSided',None)}")
            bc = getattr(mat, 'baseColorTexture', None)
            if bc is not None:
                arr = np.asarray(bc)
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    a = arr[..., 3]
                    print(f"  baseColor alpha: min={a.min()} mean={a.mean():.1f} frac<128={100*(a<128).mean():.2f}%")
        except Exception as e:
            print(f"  material: (n/a: {e})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        assess(p)
