# -*- coding: utf-8 -*-
"""FAPN · Q5/A5-3 + A5-4: two-way cluster bootstrap and bootstrap chance bands
for the Saraga 2x2 pair design.

A5-3  The published intervals resample rāgs. Several lead artists appear in more
      than one rāg, so a rāg-cluster bootstrap does not remove artist-level
      dependence. We therefore also resample artists and report, for each
      contrast, the union (the wider) of the rāg-cluster and artist-cluster
      95% intervals.
A5-4  The retrieval chance level is an analytic function of the rāg-size
      distribution of this 106-recording set. A rāg-cluster bootstrap
      duplicates recordings and would therefore inflate p_same, so the
      chance level is not bootstrapped: it is computed per query from the
      exact rāg-size distribution and reported as a distribution over
      queries (mean and range), not as a single pooled value.

Output: results/e3/saraga_twoway.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
SARAGA_EMB = PROJ / "results" / "saraga" / "embeddings"
SARAGA_MFCC = PROJ / "results" / "e2_supplement" / "mfcc" / "mfcc_saraga.npy"
SARAGA_MANIFEST = PROJ / "data" / "openethnic" / "saraga" / "manifest.csv"
OUT = PROJ / "results" / "e3" / "saraga_twoway.json"

ENCODERS = ["beats", "mert", "ast", "audiomae"]
N_BOOT = 2000
SEED = 20260501


def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def load():
    df = pd.read_csv(SARAGA_MANIFEST)
    mask = df["lead_artist"].notna().to_numpy()
    df = df[mask].reset_index(drop=True)
    raag = df["raag_common"].astype(str).to_numpy()
    artist = df["lead_artist"].astype(str).to_numpy()
    feats = {}
    for w in ENCODERS:
        p = SARAGA_EMB / f"{w}.npy"
        if p.exists():
            feats[w] = l2norm(np.load(p)[mask])
    if SARAGA_MFCC.exists():
        feats["mfcc"] = l2norm(np.load(SARAGA_MFCC)[mask])
    return df, raag, artist, feats


def cells(S, raag, artist):
    n = len(raag)
    iu = np.triu_indices(n, 1)
    s = S[iu]
    sr = raag[iu[0]] == raag[iu[1]]
    sa = artist[iu[0]] == artist[iu[1]]
    m = {"same_raag_cross_artist": sr & ~sa, "same_raag_same_artist": sr & sa,
         "diff_raag_same_artist": ~sr & sa, "diff_raag_cross_artist": ~sr & ~sa}
    out = {k: {"mean": float(s[v].mean()) if v.sum() else None, "n_pairs": int(v.sum())}
           for k, v in m.items()}
    return out, s, m


def contrast_boot(S, raag, artist, cluster, n_boot=N_BOOT, seed=SEED):
    """Resample clusters of the named variable; return (mean, lo, hi) per contrast."""
    rng = np.random.default_rng(seed)
    var = raag if cluster == "raag" else artist
    uniq = np.unique(var)
    idx_of = {u: np.where(var == u)[0] for u in uniq}
    e_raag, e_art = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[u] for u in pick])
        sub = S[np.ix_(idx, idx)]
        r, a = raag[idx], artist[idx]
        iu = np.triu_indices(len(idx), 1)
        s = sub[iu]
        sr = r[iu[0]] == r[iu[1]]
        sa = a[iu[0]] == a[iu[1]]
        mSR, mDB, mDA = sr & ~sa, ~sr & ~sa, ~sr & sa
        if mSR.sum() and mDB.sum():
            e_raag.append(float(s[mSR].mean() - s[mDB].mean()))
        if mDA.sum() and mDB.sum():
            e_art.append(float(s[mDA].mean() - s[mDB].mean()))

    def ci(x):
        x = np.array([v for v in x if np.isfinite(v)])
        if not len(x):
            return None
        return [float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975))]

    return {"raag_effect": ci(e_raag), "artist_effect": ci(e_art)}


def chance_per_query(raag, ks=(1, 5, 10)):
    """Retrieval chance level per query under this 106-recording raag distribution.

    A query in a raag of size c has probability (c-1)/(n-1) that one random
    other recording shares its raag; R@k chance is 1-(1-p)^k. The mean equals
    the pooled same-raag pair probability, and the spread shows the floor is
    not uniform across queries.
    """
    n = len(raag)
    _, inv, cnt = np.unique(raag, return_inverse=True, return_counts=True)
    p = (cnt[inv] - 1) / (n - 1)
    out = {}
    for k in ks:
        v = 1.0 - (1.0 - p) ** k
        out["R@%d" % k] = {"mean": float(np.mean(v)), "min": float(np.min(v)),
                           "max": float(np.max(v))}
    return out


def main():
    t0 = time.time()
    df, raag, artist, feats = load()
    n = len(df)
    # artist <-> raag overlap
    a2r = df.groupby("lead_artist")["raag_common"].nunique()
    r2a = df.groupby("raag_common")["lead_artist"].nunique()
    res = {
        "n": int(n), "n_raag": int(len(np.unique(raag))), "n_artist": int(len(np.unique(artist))),
        "n_pairs_total": int(n * (n - 1) // 2),
        "artist_spanning_multiple_raag": int((a2r > 1).sum()),
        "artist_max_raag": int(a2r.max()),
        "raag_with_multiple_artists": int((r2a > 1).sum()),
        "raag_max_artist": int(r2a.max()),
        "n_boot": N_BOOT,
        "chance_per_query": chance_per_query(raag),
        "encoders": {},
    }
    print(f"n={n} rāg={res['n_raag']} artist={res['n_artist']} pairs={res['n_pairs_total']}")
    print(f"artists spanning >1 rāg: {res['artist_spanning_multiple_raag']}/{res['n_artist']} "
          f"(max {res['artist_max_raag']})")
    print(f"rāgs with >1 artist: {res['raag_with_multiple_artists']}/{res['n_raag']} "
          f"(max {res['raag_max_artist']})")
    print("chance per query:", {k: {kk: round(vv, 4) for kk, vv in v.items()}
                                for k, v in res["chance_per_query"].items()})

    for name, X in feats.items():
        S = X @ X.T          # l2-normalised rows -> cosine similarity
        cell, _, _ = cells(S, raag, artist)
        b_raag = contrast_boot(S, raag, artist, "raag")
        b_art = contrast_boot(S, raag, artist, "artist")
        union = {}
        for key in ("raag_effect", "artist_effect"):
            lo = min(b_raag[key][1], b_art[key][1])
            hi = max(b_raag[key][2], b_art[key][2])
            union[key] = {"lo": lo, "hi": hi, "includes_zero": bool(lo <= 0 <= hi),
                          "width": hi - lo}
        res["encoders"][name] = {"cells": cell, "boot_raag_cluster": b_raag,
                                 "boot_artist_cluster": b_art, "two_way_union": union}
        print(f"{name:9s} rāg eff raag-CI={[round(x,4) for x in b_raag['raag_effect']]} "
              f"artist-CI={[round(x,4) for x in b_art['raag_effect']]} "
              f"| artist eff raag-CI={[round(x,4) for x in b_raag['artist_effect']]} "
              f"artist-CI={[round(x,4) for x in b_art['artist_effect']]}", flush=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved ->", OUT, res["elapsed_sec"], "s")


if __name__ == "__main__":
    main()
