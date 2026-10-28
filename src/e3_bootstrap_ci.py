# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening 6/6 (Part A): statistical rigour (answers Reviewer 3)

Reviewer 3 notes: the main results lack uncertainty quantification; class imbalance (bowed 334 vs wind 6630) biases accuracy;
the differences between encoders were not paired-tested; Table 4 lacks a family-level retrieval column.

This script provides:
  1) fragment-level bootstrap 95% CI (resampling test fragments) for sil_proto / Delta-cos / proto_acc /
     knn_acc / few1 / few8 / R@1-ins. Set-level statistics (sil/Delta-cos) are recomputed, per-sample ones resampled.
  2) paired difference CI between encoders: BEATs minus {MERT,AST,AudioMAE,MFCC} (same resampling indices, paired).
  3) family-level retrieval R@k (fills Table 4) + bootstrap std; the instrument level is reported alongside.
  4) balanced accuracy + majority-class baseline (answers the imbalance).

CPU only. Output results/e3/bootstrap_ci.json + console.
"""
import json, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from sklearn.metrics import silhouette_score, balanced_accuracy_score

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import (  # noqa
    prepare_features, l2norm, proto_predict, knn_predict, delta_cos, ENCODERS)
from analysis import compute_class_prototypes  # noqa

OUT = PROJ / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]
B = 1000
SEED = 20260501


def proto_sil(X, y_class, y_family):
    classes = np.array(sorted(np.unique(y_class)))
    P, pf = [], []
    for c in classes:
        idx = np.where(y_class == c)[0]
        P.append(l2norm(X[idx].mean(0, keepdims=True))[0])
        pf.append(np.unique(y_family[idx])[0])
    return float(silhouette_score(np.stack(P), np.array(pf), metric="cosine"))


def dcos_vec(S, y):
    """within/between family cosine difference (class-equal weight, same protocol as the per-family n_q sampling of delta_cos).
    S is the precomputed similarity matrix; the bootstrap only slices submatrices, avoiding recomputing X@X every round."""
    n = len(y)
    ds = []
    for c in np.unique(y):
        memb = np.where(y == c)[0]
        other = np.where(y != c)[0]
        if len(memb) < 2 or len(other) == 0:
            continue
        blk = S[np.ix_(memb, memb)]
        within = (blk.sum(1) - 1.0) / (len(memb) - 1)
        between = S[np.ix_(memb, other)].sum(1) / len(other)
        ds.append(float(within.mean() - between.mean()))
    return float(np.mean(ds)) if ds else float("nan")


def retrieve(S, gal_lab, q_lab, ks):
    """query (rows) -> gallery (columns), hit rate @k."""
    out = {}
    for k in ks:
        top = np.argsort(-S, axis=1)[:, :k]
        out[f"R@{k}"] = float(np.mean([np.any(gal_lab[top[i]] == q_lab[i])
                                       for i in range(len(q_lab))]))
    return out


def prep(w, feats):
    Xd, dfd = feats[w]["dev"]; Xt, dft = feats[w]["test"]
    Xd, Xt = l2norm(Xd), l2norm(Xt)
    d = dict(
        Xd=Xd, Xt=Xt,
        yd_c=dfd["instrument_class"].to_numpy(), yd_f=dfd["family"].to_numpy(),
        yt_c=dft["instrument_class"].to_numpy(), yt_f=dft["family"].to_numpy())
    # per-sample correctness (computed once; the bootstrap only resamples)
    d["proto_pred"] = proto_predict(Xd, d["yd_f"], Xt)
    d["proto_correct"] = (d["proto_pred"] == d["yt_f"])
    d["knn_correct"] = (knn_predict(Xd, d["yd_f"], Xt) == d["yt_f"])
    Sins = Xt @ Xd.T
    d["r1_correct"] = (d["yd_c"][np.argmax(Sins, axis=1)] == d["yt_c"])
    # few-shot: one fixed support draw (same seed), per-query correctness
    rng = np.random.default_rng(SEED)
    for ns in (1, 8):
        sel = []
        for f in np.unique(d["yd_f"]):
            idx = np.where(d["yd_f"] == f)[0]
            sel += list(rng.choice(idx, size=min(ns, len(idx)), replace=False))
        sel = np.array(sel)
        d[f"few{ns}_correct"] = (proto_predict(Xd[sel], d["yd_f"][sel], Xt) == d["yt_f"])
    return d


def point_and_ci(d):
    """Point estimate + bootstrap CI. Set-level statistics (sil/Delta-cos) recomputed, per-sample ones resampled."""
    Xt, yt_c, yt_f = d["Xt"], d["yt_c"], d["yt_f"]
    n = len(Xt)
    rng = np.random.default_rng(SEED)
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    per_keys = ["proto_correct", "knn_correct", "r1_correct",
                "few1_correct", "few8_correct"]
    S_test = Xt @ Xt.T                       # precomputed; the bootstrap only slices
    point = {
        "sil_proto": proto_sil(Xt, yt_c, yt_f),
        "dcos": dcos_vec(S_test, yt_f),
        "proto_acc": float(d["proto_correct"].mean()),
        "knn_acc": float(d["knn_correct"].mean()),
        "few1": float(d["few1_correct"].mean()),
        "few8": float(d["few8_correct"].mean()),
        "r1_ins": float(d["r1_correct"].mean()),
        "balanced_acc": float(balanced_accuracy_score(yt_f, d["proto_pred"])),
    }
    draws = {k: [] for k in point}
    for ix in idxs:
        Xb, yb_c, yb_f = Xt[ix], yt_c[ix], yt_f[ix]
        try:
            draws["sil_proto"].append(proto_sil(Xb, yb_c, yb_f))
        except Exception:
            draws["sil_proto"].append(np.nan)
        draws["dcos"].append(dcos_vec(S_test[np.ix_(ix, ix)], yb_f))
        for k in per_keys:
            draws[{"proto_correct": "proto_acc", "knn_correct": "knn_acc",
                   "r1_correct": "r1_ins", "few1_correct": "few1",
                   "few8_correct": "few8"}[k]].append(float(d[k][ix].mean()))
        draws["balanced_acc"].append(float(balanced_accuracy_score(yb_f, d["proto_pred"][ix])))
    ci = {}
    for k, v in draws.items():
        v = np.array([x for x in v if np.isfinite(x)])
        ci[k] = {"point": point[k], "lo": float(np.quantile(v, .025)),
                 "hi": float(np.quantile(v, .975)), "std": float(v.std())}
    return ci, point, idxs


def paired_diff(dA, dB, idxs):
    """BEATs(A) minus other(B) paired bootstrap difference CI (per-sample quantities)."""
    out = {}
    for k, col in [("proto_acc", "proto_correct"), ("knn_acc", "knn_correct"),
                   ("few1", "few1_correct"), ("few8", "few8_correct"),
                   ("r1_ins", "r1_correct")]:
        diffs = [float(dA[col][ix].mean() - dB[col][ix].mean()) for ix in idxs]
        diffs = np.array(diffs)
        out[k] = {"diff": float(dA[col].mean() - dB[col].mean()),
                  "lo": float(np.quantile(diffs, .025)),
                  "hi": float(np.quantile(diffs, .975))}
    return out


def main():
    t0 = time.time()
    feats = prepare_features()
    results = {"B": B, "encoders": {}, "paired_vs_beats": {}}

    prepped = {w: prep(w, feats) for w in ORDER if w in feats and feats[w]["test"] is not None}

    for w, d in prepped.items():
        ci, point, idxs = point_and_ci(d)
        # family / instrument retrieval R@k (dev gallery -> test query)
        Xd, Xt = d["Xd"], d["Xt"]
        S = Xt @ Xd.T
        ret_f = retrieve(S, d["yd_f"], d["yt_f"], (1, 5, 10))
        ret_c = retrieve(S, d["yd_c"], d["yt_c"], (1, 5, 10))
        maj = float(np.bincount(np.unique(d["yt_f"], return_inverse=True)[1]).max() / len(d["yt_f"]))
        results["encoders"][w] = {"ci": ci, "retrieval_family": ret_f,
                                  "retrieval_instrument": ret_c,
                                  "majority_baseline_family": maj}
        print("\n" + "=" * 96)
        print(f"BOOTSTRAP CI — {w}")
        print("=" * 96)
        for k, v in ci.items():
            print(f"  {k:14s} = {v['point']:.4f}  95%CI=[{v['lo']:.4f},{v['hi']:.4f}]  sd={v['std']:.4f}")
        print(f"  retrieval family R@1/5/10 = "
              f"{ret_f['R@1']:.3f}/{ret_f['R@5']:.3f}/{ret_f['R@10']:.3f} | "
              f"instrument = {ret_c['R@1']:.3f}/{ret_c['R@5']:.3f}/{ret_c['R@10']:.3f} | "
              f"majority(fam)={maj:.3f}", flush=True)

        if w != "beats" and "beats" in prepped:
            pd = paired_diff(prepped["beats"], d, idxs)
            results["paired_vs_beats"][w] = pd
            print(f"  paired BEATs−{w}: " + " ".join(
                f"{k}={v['diff']:+.4f}[{v['lo']:+.4f},{v['hi']:+.4f}]" for k, v in pd.items()))

    results["elapsed_sec"] = round(time.time() - t0, 1)
    outp = OUT / "bootstrap_ci.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {outp} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
