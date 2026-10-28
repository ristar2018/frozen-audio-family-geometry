# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening 2/6: controlled geometric interventions (causal evidence) + competing diagnoses

Answers R1-2/R1-4 of the review ("geometric health is not a model-independent quantity", "geometry -> downstream is a causal claim"):
upgrades "geometric health -> downstream performance" from an observed correlation to controlled causal evidence.

Design: fit a family of geometric transforms on each encoder dev embeddings (fit on dev only, then frozen and applied to test,
preventing leakage) and recompute the triad metrics and the downstream tasks under the same test protocol, observing whether
changing the geometry changes the downstream results predictably.

Transform family (fit on dev):
  identity            baseline
  center              subtract the dev mean
  pc_remove_{1,4}     remove the top k principal components (strip the globally shared component)
  rank_{64,256}       keep the top r principal components (rank reduction)
  power_{beta}        PCA anisotropy power transform: s_i=(lambda_i/lambda_bar)^(-beta)
                        beta=+0.5 whitening (less anisotropy) / beta=-0.5 more anisotropy
  rand_orth           random orthogonal rotation (negative control: cosine unchanged, metrics should barely move)
  dim_shuffle         random dimension permutation (destroys structure, negative control)

Metrics (on test): sil_proto / Delta-cos / proto_acc / knn_acc / few1 / few8 / R@1-ins,
plus the dev geometry quantities eff_rank / aniso_top1.

Key self-check: every metric under rand_orth should change by ~0 relative to identity (cosine is invariant),

CPU only. Output results/e3/geometry_intervention.json + console matrices.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import (  # noqa
    prepare_features, l2norm, proto_predict, knn_predict, delta_cos, ENCODERS)

OUT = PROJ / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]
FEW_REPEAT = 10


# ============================================================
# geometric transforms (fit on dev)
# ============================================================
def build_transforms(Xd, seed=20260501):
    d = Xd.shape[1]
    mu = Xd.mean(0, keepdims=True)
    A = Xd - mu
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    C = A.T @ A / len(A)
    w, V = np.linalg.eigh(C)                 # ascending order
    w = np.maximum(w, 1e-12)
    wm = w.mean()
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    perm = np.random.default_rng(seed + 1).permutation(d)

    Ts = {}
    Ts["identity"] = lambda X: X
    Ts["center"] = lambda X: X - mu
    for k in (1, 4):
        Vk = Vt[:k]
        Ts[f"pc_remove_{k}"] = (lambda Vk: lambda X: (X - mu) - ((X - mu) @ Vk.T) @ Vk)(Vk)
    for r in (64, 256):
        Vr = Vt[:r]
        Ts[f"rank_{r}"] = (lambda Vr: lambda X: ((X - mu) @ Vr.T) @ Vr)(Vr)
    for beta in (0.5, 0.25, -0.25, -0.5):
        s = (w / wm) ** (-beta)
        W = (V * s) @ V.T
        Ts[f"power_{beta:+.2f}"] = (lambda W: lambda X: (X - mu) @ W)(W)
    Ts["rand_orth"] = lambda X: (X - mu) @ Q
    Ts["dim_shuffle"] = lambda X: X[:, perm]
    return Ts


# ============================================================
# geometry quantities and evaluation
# ============================================================
def proto_sil(X, y_class, y_family):
    classes = np.array(sorted(np.unique(y_class)))
    P, pf = [], []
    for c in classes:
        idx = np.where(y_class == c)[0]
        P.append(l2norm(X[idx].mean(0, keepdims=True))[0])
        pf.append(np.unique(y_family[idx])[0])
    return float(silhouette_score(np.stack(P), np.array(pf), metric="cosine"))


def eff_rank(X):
    Xc = X - X.mean(0, keepdims=True)
    w = np.linalg.eigvalsh(Xc.T @ Xc / len(Xc))
    w = np.maximum(w, 1e-12); w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def aniso_top1(X):
    Xc = X - X.mean(0, keepdims=True)
    w = np.linalg.eigvalsh(Xc.T @ Xc / len(Xc))
    w = np.maximum(w, 1e-12)
    return float(w.max() / w.sum())


def evaluate(Xd, yd_c, yd_f, Xt, yt_c, yt_f, rng):
    Xd = l2norm(Xd); Xt = l2norm(Xt)
    r = {}
    r["sil_proto"] = proto_sil(Xt, yt_c, yt_f)
    r["dcos"] = delta_cos(Xt, yt_f, rng)
    r["proto_acc"] = float((proto_predict(Xd, yd_f, Xt) == yt_f).mean())
    r["knn_acc"] = float((knn_predict(Xd, yd_f, Xt) == yt_f).mean())
    for ns in (1, 8):
        accs = []
        for _ in range(FEW_REPEAT):
            sel = []
            for f in np.unique(yd_f):
                idx = np.where(yd_f == f)[0]
                sel += list(rng.choice(idx, size=min(ns, len(idx)), replace=False))
            sel = np.array(sel)
            accs.append(float((proto_predict(Xd[sel], yd_f[sel], Xt) == yt_f).mean()))
        r[f"few{ns}"] = float(np.mean(accs))
    S = Xt @ Xd.T
    r["r1_ins"] = float((yd_c[np.argmax(S, axis=1)] == yt_c).mean())
    r["eff_rank"] = eff_rank(Xd)
    r["aniso_top1"] = aniso_top1(Xd)
    return r


COLS = ["sil_proto", "dcos", "proto_acc", "knn_acc", "few1", "few8", "r1_ins",
        "eff_rank", "aniso_top1"]


def main():
    t0 = time.time()
    feats = prepare_features()
    results = {"few_repeat": FEW_REPEAT, "encoders": {}}

    for w in ORDER:
        if w not in feats or feats[w]["test"] is None:
            continue
        Xd, dfd = feats[w]["dev"]
        Xt, dft = feats[w]["test"]
        yd_c, yd_f = dfd["instrument_class"].to_numpy(), dfd["family"].to_numpy()
        yt_c, yt_f = dft["instrument_class"].to_numpy(), dft["family"].to_numpy()
        Ts = build_transforms(Xd)
        rows = {}
        for name, T in Ts.items():
            rng = np.random.default_rng(20260501)          # fixed, so runs stay comparable
            rows[name] = evaluate(T(Xd), yd_c, yd_f, T(Xt), yt_c, yt_f, rng)
        results["encoders"][w] = rows

        # self-check: an orthogonal rotation leaves cosine unchanged => rand_orth should be ~= center (both subtract the mean);
        #             a dimension permutation leaves cosine unchanged => dim_shuffle should be ~= identity. Any deviation => pipeline bug.
        cen, ro = rows["center"], rows["rand_orth"]
        base, ds = rows["identity"], rows["dim_shuffle"]
        d_ro = max(abs(cen[c] - ro[c]) for c in COLS)
        d_ds = max(abs(base[c] - ds[c]) for c in COLS)
        # threshold 1e-3: low-dimensional embeddings (MFCC) carry ~1e-4 of numerical tie noise at the retrieval argmax
        ok = (d_ro < 1e-3) and (d_ds < 1e-3)
        print(f"[self-check] {w}: max|rand_orth-center|={d_ro:.2e} "
              f"max|dim_shuffle-identity|={d_ds:.2e} ({'OK' if ok else 'CHECK!'})",
              flush=True)

        # causal monotonicity: across transforms, Spearman rank correlation between the geometry quantity and the downstream metric
        try:
            from scipy.stats import spearmanr
            names = list(Ts)
            trend = {}
            for g in ("aniso_top1", "eff_rank", "dcos", "sil_proto"):
                for t in ("proto_acc", "few1", "few8"):
                    gv = [rows[n][g] for n in names]
                    tv = [rows[n][t] for n in names]
                    rho, _ = spearmanr(gv, tv)
                    trend[f"{g}~{t}"] = round(float(rho), 3)
            results.setdefault("trend_spearman", {})[w] = trend
        except Exception as e:  # noqa
            print(f"[trend] skipped: {e}")

        print("\n" + "=" * 100)
        print(f"GEOMETRY INTERVENTION — {w}")
        print("=" * 100)
        print(f"{'transform':16s} " + "".join(f"{c:>10s}" for c in COLS))
        for name in Ts:
            r = rows[name]
            print(f"{name:16s} " + "".join(f"{r[c]:10.4f}" for c in COLS), flush=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    outp = OUT / "geometry_intervention.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {outp}  ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
