# -*- coding: utf-8 -*-
"""
Verify the per-class leave-one-out accuracy of FolkMusic 15-class (self excluded).
Section 3.5 of the paper claims "guzheng 0.934 for all encoders; every other class at 0.999 or 1.00",
but the per_class output of e1 is bit-identical across the four encoders, which suggests an analysis defect;
this script recomputes it under the same protocol as e2 (cosine, k=5, distance weights, query excluded).
Output: results/e3/folk15_perclass.json
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import prepare_features, l2norm, KNN_K  # noqa

OUT = PROJ / "results" / "e3" / "folk15_perclass.json"
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]


def loo_pred(X, y, k=KNN_K, chunk=512):
    X = l2norm(np.asarray(X, dtype=np.float64))
    classes, inv = np.unique(y, return_inverse=True)
    n = len(X)
    out = np.empty(n, dtype=classes.dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = X[s:e] @ X.T
        rows = np.arange(e - s)
        S[rows, np.arange(s, e)] = -np.inf
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        d = np.take_along_axis(S, idx, axis=1)
        lab = inv[idx]
        w = np.maximum(d, 0.0)
        votes = np.zeros((e - s, len(classes)))
        for j in range(k):
            votes[rows, lab[:, j]] += w[:, j]
        out[s:e] = classes[votes.argmax(axis=1)]
    return out


def main():
    feats = prepare_features()
    res = {}
    for w in ORDER:
        if w not in feats or feats[w].get("dev") is None:
            continue
        X, df = feats[w]["dev"]
        y = df["instrument_class"].to_numpy()
        pred = loo_pred(X, y)
        acc = {c: round(float((pred[y == c] == c).mean()), 6) for c in sorted(set(y))}
        n_c = {c: int((y == c).sum()) for c in sorted(set(y))}
        res[w] = {"overall": round(float((pred == y).mean()), 6),
                  "per_class": acc, "n_per_class": n_c}
        worst = sorted(acc.items(), key=lambda x: x[1])[:3]
        print(f"{w:9s} overall={(pred == y).mean():.4f}  worst: " +
              ", ".join(f"{c}={v:.3f}(n={n_c[c]})" for c, v in worst))
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
