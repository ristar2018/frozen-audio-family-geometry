# -*- coding: utf-8 -*-
"""
Back-fill: leave-one-out fragment-level family kNN accuracy on FolkMusic dev (9,573 fragments).
The kNN_clip column of Table h1a previously held test-set numbers (labelled dev) with no source;
this script recomputes it on dev under the same protocol as e2 (cosine, k=5, distance weights, query excluded).
Output: results/e3/dev_clip_knn.json
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import prepare_features, l2norm, KNN_K  # noqa

OUT = PROJ / "results" / "e3" / "dev_clip_knn.json"
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]


def loo_knn_acc(X, y, k=KNN_K, chunk=512):
    """leave-one-out kNN (self excluded), cosine, distance-weighted vote."""
    X = l2norm(np.asarray(X, dtype=np.float64))
    labels = np.asarray(y)
    classes, inv = np.unique(labels, return_inverse=True)
    n = len(X)
    correct = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = X[s:e] @ X.T                       # (b, n)
        rows = np.arange(e - s)
        S[rows, np.arange(s, e)] = -np.inf     # exclude self
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        d = np.take_along_axis(S, idx, axis=1)
        lab = inv[idx]
        w = np.maximum(d, 0.0)                 # cosine similarity as the distance weight
        votes = np.zeros((e - s, len(classes)))
        for j in range(k):
            votes[rows, lab[:, j]] += w[:, j]
        pred = classes[votes.argmax(axis=1)]
        correct += int((pred == labels[s:e]).sum())
    return correct / n


def main():
    feats = prepare_features()
    res = {}
    for w in ORDER:
        if w not in feats or feats[w].get("dev") is None:
            print(f"[skip] {w}")
            continue
        X, df = feats[w]["dev"]
        yf = df["family"].to_numpy()
        acc = loo_knn_acc(X, yf)
        res[w] = {"n": int(len(X)), "knn_family_loo": round(float(acc), 6)}
        print(f"{w:10s} n={len(X):6d}  knn_family_loo={acc:.6f}")
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
