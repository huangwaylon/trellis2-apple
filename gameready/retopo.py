"""
Stage 2 [venv python -> QuadriFlow / InstantMeshes]: watertight low-genus base
-> adaptive pure-QUAD retopology with field-aligned edge flow.

Default tool = QuadriFlow -mcf -adaptive: 100% pure quads, WATERTIGHT (min-cost-flow
singularity matching), and curvature-ADAPTIVE (smaller quads in detailed regions).
InstantMeshes (--tool instantmeshes) is curvature-adaptive too but tends to leave
small holes at singularities on closed inputs, so it's the fallback, not default.

We calibrate the face target to land the requested VERT count (for a closed pure-quad
mesh verts ~= quads); QuadriFlow -f overshoots, so iterate.

  ./.venv/bin/python gameready/retopo.py --input gameready_work/kei/base.obj \
      --output gameready_work/kei/quad.obj --verts 70000
"""
import argparse, os, subprocess, sys, numpy as np

BIN = "/Users/waylonhuang/Documents/other/trellis-mac/bin"
QF = os.path.join(BIN, "quadriflow")
IM = os.path.join(BIN, "InstantMeshes")


def face_tokens(obj):
    v = t = q = n = 0
    with open(obj) as fh:
        for ln in fh:
            if ln.startswith("v "): v += 1
            elif ln.startswith("f "):
                k = len(ln.split()) - 1
                if k == 3: t += 1
                elif k == 4: q += 1
                else: n += 1
    return v, t, q, n


def boundary_pct(obj):
    V = []; F = []
    with open(obj) as fh:
        for ln in fh:
            if ln.startswith("v "):
                V.append([float(x) for x in ln.split()[1:4]])
            elif ln.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in ln.split()[1:]]
                for i in range(1, len(idx) - 1):
                    F.append([idx[0], idx[i], idx[i + 1]])
    V = np.array(V); F = np.array(F)
    _, inv = np.unique(np.round(V, 6), axis=0, return_inverse=True)
    Fw = inv[F]
    E = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1)
    u, c = np.unique(E, axis=0, return_counts=True)
    return 100 * (c == 1).sum() / len(u)


def run_qf(inp, out, faces, adaptive=True, sharp=False):
    cmd = [QF, "-i", inp, "-o", out, "-f", str(int(faces)), "-mcf"]
    if adaptive: cmd.append("-adaptive")
    if sharp: cmd.append("-sharp")
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)


def run_im(inp, out, verts, crease):
    subprocess.run([IM, inp, "-o", out, "-r", "4", "-p", "4", "-d",
                    "-c", str(crease), "-v", str(int(verts))],
                   check=True, capture_output=True, timeout=1800)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verts", type=int, default=70000)
    p.add_argument("--tool", choices=["quadriflow", "instantmeshes"], default="quadriflow")
    p.add_argument("--adaptive", type=int, default=1)
    p.add_argument("--crease", type=int, default=25)
    p.add_argument("--tol", type=float, default=0.22)
    a = p.parse_args()

    # QuadriFlow -adaptive: -f (quad target) ~= output vert count (measured), so
    # seed -f directly at the vert target. Each -mcf pass is expensive, so cap at
    # 2 attempts with a wide tolerance (the seed usually lands in one).
    req = a.verts
    best = None
    for attempt in range(2):
        if a.tool == "quadriflow":
            run_qf(a.input, a.output, req, adaptive=bool(a.adaptive))
        else:
            run_im(a.input, a.output, req, a.crease)
        v, t, q, n = face_tokens(a.output)
        qp = 100 * q / max(1, t + q + n)
        bnd = boundary_pct(a.output)
        print(f"[retopo] try{attempt} req={int(req)} -> {v} verts, {q}q {t}t {n}n "
              f"({qp:.1f}% quad, boundary {bnd:.2f}%)")
        if best is None or abs(v - a.verts) < abs(best[1] - a.verts):
            best = (req, v)
        if (1 - a.tol) * a.verts <= v <= (1 + a.tol) * a.verts:
            break
        req = max(2000, req * a.verts / max(1, v))
    v, t, q, n = face_tokens(a.output)
    bnd = boundary_pct(a.output)
    print(f"[retopo] wrote {a.output}: {v} verts, {100*q/max(1,t+q+n):.1f}% quad, "
          f"boundary {bnd:.2f}% (target {a.verts} verts)")
    if bnd > 1.0:
        print(f"[retopo] NOTE boundary {bnd:.2f}% — finish_bake hole-fill will close it", file=sys.stderr)


if __name__ == "__main__":
    main()
