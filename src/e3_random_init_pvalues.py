# -*- coding: utf-8 -*-
"""FAPN · Q8/A8-1: exact permutation p for each random-initialisation seed.

Same null as the confirmatory test: all 180,180 assignments of the fixed
multiset [bowed]*2 + [plucked]*7 + [wind]*6 to the 15 ordered class
prototypes. Statistic: cosine silhouette over the 15 class prototypes on the
held-out split (identical to e3_exact_permutation.py).

Vectorised re-implementation of sil_all_from_dist; validated by reproducing the
pretrained exact p-values stored in results/e3/exact_permutation.json.

Output: results/e3/random_init_pvalues.json
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)

EMB = PROJ / "results" / "e3" / "random_init" / "embeddings"
E1 = PROJ / "results" / "e1" / "embeddings"
E2 = PROJ / "results" / "e2_supplement" / "embeddings"
OUT = PROJ / "results" / "e3" / "random_init_pvalues.json"

SIZES = (2, 7, 6)
NCLS = 15


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z["embeddings"], z["instrument_class"], z["family"]


def enumerate_labels(n=NCLS, sizes=SIZES):
    rows = []
    for g1 in combinations(range(n), sizes[0]):
        s1 = set(g1)
        rest1 = [i for i in range(n) if i not in s1]
        for g2 in combinations(rest1, sizes[1]):
            lab = np.full(n, 2, dtype=np.int8)
            lab[list(g1)] = 0
            lab[list(g2)] = 1
            rows.append(lab)
    return np.stack(rows, 0)


def sil_all(D, LAB, batch=20000):
    """D:(15,15) cosine distance, LAB:(M,15) -> (M,) silhouette (vectorised)."""
    M = LAB.shape[0]
    n = D.shape[0]
    out = np.empty(M, dtype=np.float64)
    ar = np.arange(n)
    for s in range(0, M, batch):
        lab = LAB[s:s + batch]                      # (B,15)
        same = lab[:, :, None] == lab[:, None, :]   # (B,15,15)
        cnt3 = np.zeros((lab.shape[0], 3), dtype=np.int64)
        for g in range(3):
            cnt3[:, g] = (lab == g).sum(1)
        cnt_i = cnt3[np.arange(lab.shape[0])[:, None], lab]   # (B,15)
        a = (D[None, :, :] * same).sum(2) / np.maximum(cnt_i - 1, 1)
        b = np.full_like(a, np.inf)
        for g in range(3):
            gm = (lab == g)[:, None, :]                        # (B,1,15)
            mean_g = (D[None, :, :] * gm).sum(2) / np.maximum(cnt3[:, g][:, None], 1)
            b = np.where(lab == g, b, np.minimum(b, mean_g))
        denom = np.maximum(np.maximum(a, b), 1e-12)
        sil = (b - a) / denom
        sil = np.where(cnt_i <= 1, 0.0, sil)
        out[s:s + batch] = sil.mean(1)
    return out


def class_protos(X, y_class):
    classes = np.array(sorted(np.unique(y_class)))
    P = []
    for c in classes:
        idx = np.where(y_class == c)[0]
        P.append(l2norm(X[idx].mean(0, keepdims=True))[0])
    return np.stack(P, 0)


def exact_p(X, y_class, LAB, D_ref=None):
    P = class_protos(X, y_class)
    D = 1.0 - (P @ P.T)
    np.fill_diagonal(D, 0.0)
    from sklearn.metrics import silhouette_score
    # real value from the class->family map implied by the data
    return P, D


def run():
    t0 = time.time()
    LAB = enumerate_labels()
    print("M =", LAB.shape[0], flush=True)

    # ---- validate the vectorised silhouette on the pretrained embeddings ----
    ref = json.load(open(PROJ / "results/e3/exact_permutation.json", encoding="utf-8"))
    res = {"n_assignments": int(LAB.shape[0]), "split": "test", "encoders": {}}
    encs = ["beats", "mert", "ast", "audiomae"]

    for name in encs:
        Xt, yc, yf = load_npz(E2 / f"{name}_folkmusic_test.npz")
        rec = {}
        # pretrained (validation against exact_permutation.json)
        P, D = exact_p(l2norm(Xt), yc, LAB)
        classes = np.array(sorted(np.unique(yc)))
        cls_fam = np.array([np.unique(yf[yc == c])[0] for c in classes])
        from sklearn.metrics import silhouette_score
        sil_pre = float(silhouette_score(P, cls_fam, metric="cosine"))
        null = sil_all(D, LAB)
        p_pre = float((null >= sil_pre - 1e-12).sum() / LAB.shape[0])
        ref_p = ref["splits"]["test"][name]["exact_p"]
        ok = abs(p_pre - ref_p) < 1e-6 and abs(sil_pre - ref["splits"]["test"][name]["sil_real_proto"]) < 1e-5
        rec["pretrained"] = {"sil": round(sil_pre, 6), "exact_p": p_pre,
                             "ref_exact_p": ref_p, "validation_ok": bool(ok)}
        print(f"{name:9s} pretrained sil={sil_pre:+.4f} p={p_pre:.6g} (ref {ref_p:.6g}) {'OK' if ok else 'MISMATCH'}", flush=True)

        # random seeds
        seeds = {}
        for k in (0, 1, 2):
            f = EMB / f"{name}_seed{k}_test.npz"
            if not f.exists():
                continue
            Xr, yc2, yf2 = load_npz(f)
            P2, D2 = exact_p(l2norm(Xr), yc2, LAB)
            cls2 = np.array(sorted(np.unique(yc2)))
            fam2 = np.array([np.unique(yf2[yc2 == c])[0] for c in cls2])
            sil_r = float(silhouette_score(P2, fam2, metric="cosine"))
            null_r = sil_all(D2, LAB)
            p_r = float((null_r >= sil_r - 1e-12).sum() / LAB.shape[0])
            seeds[f"seed{k}"] = {"sil": round(sil_r, 6), "exact_p": p_r,
                                 "null_mean": round(float(null_r.mean()), 6),
                                 "null_q99": round(float(np.quantile(null_r, 0.99)), 6)}
            print(f"          seed{k} sil={sil_r:+.4f} p={p_r:.6g}", flush=True)
        rec["seeds"] = seeds
        res["encoders"][name] = rec

    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved ->", OUT, f"({res['elapsed_sec']}s)")


if __name__ == "__main__":
    run()
