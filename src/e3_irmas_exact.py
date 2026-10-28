# -*- coding: utf-8 -*-
"""FAPN · Q12/A12-1: exact (enumerated) family-assignment test on IRMAS.

Null matched to the confirmatory test: instrument-class identity is held fixed
(11 class prototypes are untouched) and only the family labels are reassigned,
with the observed family sizes preserved. All distinct assignments are
enumerated, so the p-value is exact rather than Monte Carlo.

Statistics reported per encoder:
  sil_proto_class  -- cosine silhouette over the 11 class prototypes
  exact_p          -- #{null >= observed} / M   (observed included)

Output: results/e3/irmas_exact_family.json
"""
import json
import sys
import time
from collections import Counter
from itertools import permutations
from math import factorial
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parent.parent
EMB = PROJ / "results" / "e3" / "irmas_emb"
ROOT = PROJ / "data" / "irmas" / "IRMAS-TrainingData"
OUT = PROJ / "results" / "e3" / "irmas_exact_family.json"

IRMAS_FAMILY = {
    "cel": "bowed_strings", "vio": "bowed_strings",
    "gac": "plucked_strings", "gel": "plucked_strings",
    "cla": "woodwind", "flu": "woodwind", "sax": "woodwind",
    "tru": "brass", "org": "keys", "pia": "keys", "voi": "voice",
}
CODES = set(IRMAS_FAMILY)
ENCODERS = ["beats", "mert", "ast", "audiomae"]


def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def parse_classes():
    wavs, cls = [], []
    for w in sorted(ROOT.rglob("*.wav")):
        for tok in [w.parent.name] + w.stem.split("__"):
            t = tok.strip().lower()
            if t in CODES:
                wavs.append(w)
                cls.append(t)
                break
    return np.array(cls)


def distinct_assignments(sizes):
    """All distinct label vectors with the given group sizes."""
    K = len(sizes)
    n = sum(sizes)
    seen = set()
    rows = []
    base = np.empty(n, dtype=np.int8)
    # enumerate via combinations over positions, group by group
    from itertools import combinations
    def rec(start, g, free):
        if g == K:
            rows.append(base.copy())
            return
        for comb in combinations(free, sizes[g]):
            s = set(comb)
            base[list(comb)] = g
            rec(0, g + 1, [i for i in free if i not in s])
        base[list(free)] = 0 if g == 0 else base[list(free)]
    # simpler: product over multiset permutations
    from itertools import combinations as C
    rows = []
    idx = list(range(n))
    def rec2(g, free):
        if g == K - 1:
            base[free] = K - 1
            rows.append(base.copy())
            return
        for comb in C(free, sizes[g]):
            s = set(comb)
            base[list(comb)] = g
            rec2(g + 1, [i for i in free if i not in s])
    rec2(0, idx)
    return np.stack(rows, 0)


def sil_all(D, LAB, K, batch=20000):
    M = LAB.shape[0]
    n = D.shape[0]
    out = np.empty(M, dtype=np.float64)
    for s in range(0, M, batch):
        lab = LAB[s:s + batch]
        B = lab.shape[0]
        same = lab[:, :, None] == lab[:, None, :]
        cnt = np.zeros((B, K), dtype=np.int64)
        for g in range(K):
            cnt[:, g] = (lab == g).sum(1)
        cnt_i = cnt[np.arange(B)[:, None], lab]
        a = (D[None, :, :] * same).sum(2) / np.maximum(cnt_i - 1, 1)
        b = np.full_like(a, np.inf)
        for g in range(K):
            gm = (lab == g)[:, None, :]
            mean_g = (D[None, :, :] * gm).sum(2) / np.maximum(cnt[:, g][:, None], 1)
            b = np.where(lab == g, b, np.minimum(b, mean_g))
        den = np.maximum(np.maximum(a, b), 1e-12)
        sil = np.where(cnt_i <= 1, 0.0, (b - a) / den)
        out[s:s + batch] = sil.mean(1)
    return out


def main():
    t0 = time.time()
    cls_all = parse_classes()
    print("wav parsed:", len(cls_all), dict(Counter(cls_all)), flush=True)
    res = {"n_wav": int(len(cls_all)), "n_class": int(len(set(cls_all.tolist()))),
           "encoders": {}}
    for enc in ENCODERS:
        z = np.load(EMB / f"{enc}_irmas.npz", allow_pickle=True)
        ok = np.asarray(z["ok_idx"])
        X = l2norm(z["embeddings"])
        cls = cls_all[ok]
        classes = np.array(sorted(np.unique(cls)))
        P = np.stack([l2norm(X[cls == c].mean(0, keepdims=True))[0] for c in classes])
        pf = np.array([IRMAS_FAMILY[c] for c in classes])
        fam_names = sorted(set(pf.tolist()))
        pf_int = np.array([fam_names.index(f) for f in pf], dtype=np.int8)
        sizes = [int((pf_int == g).sum()) for g in range(len(fam_names))]
        D = 1.0 - P @ P.T
        np.fill_diagonal(D, 0.0)
        sil = float(silhouette_score(P, pf, metric="cosine"))
        LAB = distinct_assignments(sizes)
        M = LAB.shape[0]
        # observed must be inside the enumerated space
        inside = any((LAB == pf_int).all(1))
        null = sil_all(D, LAB, len(fam_names))
        p = float((null >= sil - 1e-12).sum() / M)
        res["encoders"][enc] = {
            "sil_proto_class": round(sil, 6), "exact_p": p, "n_assignments": int(M),
            "family_sizes": sizes, "family_names": fam_names,
            "null_mean": round(float(null.mean()), 6),
            "null_q99": round(float(np.quantile(null, 0.99)), 6),
            "observed_in_null_space": bool(inside),
        }
        print(f"{enc:9s} sil={sil:+.4f} exact_p={p:.6g} (M={M}) null_mean={null.mean():+.4f} "
              f"inside={inside}", flush=True)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved ->", OUT, res["elapsed_sec"], "s")


if __name__ == "__main__":
    main()
