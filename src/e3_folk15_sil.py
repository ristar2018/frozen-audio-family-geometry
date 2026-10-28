# -*- coding: utf-8 -*-
"""
e3_folk15_sil.py -- exact recomputation of the FolkMusic dev 15-class clip-level silhouette

Why: e1_nsynth_and_15class.py used silhouette_score(..., sample_size=6000,
random_state=0), a subsampled approximation over 9,573 fragments, and tab:fine cites that value directly.
This script computes it exactly (cosine) over all 9,573 fragments, removing the subsampling approximation.

Note: with a single prototype per class the prototype-level silhouette is undefined (singleton cluster s=0),
so the 15-class silhouette can only be computed at fragment level.

Output: results/e3/folk15_sil.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parent.parent
EMB_DIR = PROJ / "results" / "e1" / "embeddings"
SPLIT_DIR = PROJ / "results" / "e1" / "splits"
OUT = PROJ / "results" / "e3" / "folk15_sil.json"

ENCODERS = ["beats", "mert", "ast", "audiomae", "mfcc"]


def main() -> None:
    df = pd.read_csv(SPLIT_DIR / "folkmusic_dev_fragments.csv")
    df = df.sort_values(["recording_id", "fragment_id"]).reset_index(drop=True)
    y = df["instrument_class"].to_numpy()
    print(f"n={len(y)}  n_class={len(set(y))}")

    out = {}
    for enc in ENCODERS:
        p = EMB_DIR / f"{enc}_folkmusic_dev.npz"
        if not p.exists():
            print(f"[skip] {enc}: no embeddings")
            continue
        X = np.load(p)["embeddings"].astype(np.float64)
        s = float(silhouette_score(X, y, metric="cosine"))
        out[enc] = round(s, 6)
        print(f"  {enc:>8s}  sil_15class_exact={s:.6f}")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
