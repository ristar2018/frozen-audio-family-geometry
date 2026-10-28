# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening 5/6: the "different rag" control for the Saraga cross-cultural probe (answers R2/R3)

Review concern: the cross-artist agreement reported on Saraga (0.847-0.956) may be a trivial effect of "recordings of the same
rag are naturally similar" (one rag shares scale and melodic material), rather than the encoder encoding musical structure.

This script does four things, separating "same-rag similarity" from "artist-identity similarity" and adding chance references:
  1) four-way decomposition of pairwise similarity (cross-artist isolates the rag effect; same-artist isolates the artist effect)
       same-rag & cross-artist  (S_R)   <- the protocol used in the paper
       same-rāg & same-artist   (S_RA)
       diff-rāg & same-artist   (D_A)
       diff-rāg & cross-artist  (D_B)
     Key contrast: rag effect = S_R - D_B (both cross-artist, differing only in rag); artist effect = D_A - D_B.
  2) rag retrieval R@k (query recording -> gallery of the other recordings, hit = same rag), against majority-rag chance.
  3) rag silhouette (76 classes, cosine).
  4) rag-cluster bootstrap 95% CI (resampling rags together with their recordings) for every key quantity.

Two weightings: pair-weighted mean (the 0.847/0.956 of the paper) and rag-equal-weighted (the min 0.209 of the paper).
CPU only. Output results/e3/saraga_controls.json + console.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import l2norm  # noqa

SARAGA_EMB = PROJ / "results" / "saraga" / "embeddings"
SARAGA_MFCC = PROJ / "results" / "e2_supplement" / "mfcc" / "mfcc_saraga.npy"
SARAGA_MANIFEST = PROJ / "data" / "openethnic" / "saraga" / "manifest.csv"
OUT = PROJ / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)
ENCODERS = ["beats", "mert", "ast", "audiomae"]
N_BOOT = 2000
RNG = np.random.default_rng(20260501)


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


def pair_sims(S, raag, artist):
    n = len(raag)
    iu = np.triu_indices(n, 1)
    s = S[iu]
    sr = raag[iu[0]] == raag[iu[1]]
    sa = artist[iu[0]] == artist[iu[1]]
    g = {
        "same_raag_cross_artist": sr & ~sa,
        "same_raag_same_artist": sr & sa,
        "diff_raag_same_artist": ~sr & sa,
        "diff_raag_cross_artist": ~sr & ~sa,
    }
    out = {k: (float(s[m].mean()) if m.sum() else float("nan"), int(m.sum()))
           for k, m in g.items()}
    return out, iu, s, sr, sa


def raag_equal_weighted(S, raag, artist):
    """Average the same-rag cross-artist similarities inside each rag, then average across rags (the min protocol of the paper)."""
    vals = []
    for r in np.unique(raag):
        idx = np.where(raag == r)[0]
        if len(idx) < 2:
            continue
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        a = artist[idx]
        cross = a[iu[0]] != a[iu[1]]
        if cross.sum():
            vals.append(float(sub[iu][cross].mean()))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def raag_retrieval(S, raag, ks=(1, 5, 10)):
    n = len(raag)
    hits = {k: [] for k in ks}
    for i in range(n):
        order = np.argsort(-S[i])
        order = order[order != i]
        for k in ks:
            hits[k].append(bool(np.any(raag[order[:k]] == raag[i])))
    # chance: the rate at which a random gallery hits the same rag (majority-rag baseline)
    _, cnt = np.unique(raag, return_counts=True)
    p_same = float(np.sum(cnt * (cnt - 1)) / (n * (n - 1)))
    chance = {f"R@{k}": float(1 - (1 - p_same) ** k) for k in ks}
    return {f"R@{k}": float(np.mean(hits[k])) for k in ks}, chance


def cluster_bootstrap(S, raag, artist, n_boot=N_BOOT, seed=20260501):
    """rag-cluster bootstrap: resample rags (with all their recordings) to give the 95% CI of the rag effect and the rest."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(raag)
    raag_idx = {r: np.where(raag == r)[0] for r in uniq}
    eff_raag, eff_artist, same_r = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([raag_idx[r] for r in pick])
        sub = S[np.ix_(idx, idx)]
        r = raag[idx]; a = artist[idx]
        n = len(idx)
        iu = np.triu_indices(n, 1)
        s = sub[iu]
        sr = r[iu[0]] == r[iu[1]]
        sa = a[iu[0]] == a[iu[1]]
        m_SR = sr & ~sa          # same-raag cross-artist
        m_DB = ~sr & ~sa         # diff-raag cross-artist
        m_DA = ~sr & sa          # diff-raag same-artist
        if m_SR.sum() and m_DB.sum():
            eff_raag.append(float(s[m_SR].mean() - s[m_DB].mean()))
            same_r.append(float(s[m_SR].mean()))
        if m_DA.sum() and m_DB.sum():
            eff_artist.append(float(s[m_DA].mean() - s[m_DB].mean()))
    def ci(x):
        x = np.array([v for v in x if np.isfinite(v)])
        return (float(np.mean(x)), float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    return {"rāg_effect(S_R-D_B)": ci(eff_raag),
            "artist_effect(D_A-D_B)": ci(eff_artist),
            "same_raag_cross_artist": ci(same_r)}


def main():
    t0 = time.time()
    df, raag, artist, feats = load()
    print(f"[saraga] n={len(df)} rāg={len(np.unique(raag))} artist={len(np.unique(artist))}")
    results = {"n": len(df), "n_raag": int(len(np.unique(raag))),
               "n_artist": int(len(np.unique(artist))), "n_boot": N_BOOT, "encoders": {}}

    for w, X in feats.items():
        S = X @ X.T
        ps, iu, s, sr, sa = pair_sims(S, raag, artist)
        req, n_req = raag_equal_weighted(S, raag, artist)
        try:
            sil = float(silhouette_score(X, raag, metric="cosine"))
        except Exception:
            sil = float("nan")
        rk, chance = raag_retrieval(S, raag)
        boot = cluster_bootstrap(S, raag, artist)

        results["encoders"][w] = {
            "pair_sims": {k: {"mean": v[0], "n_pairs": v[1]} for k, v in ps.items()},
            "rāg_effect": ps["same_raag_cross_artist"][0] - ps["diff_raag_cross_artist"][0],
            "artist_effect": ps["diff_raag_same_artist"][0] - ps["diff_raag_cross_artist"][0],
            "same_raag_pair_weighted": ps["same_raag_cross_artist"][0],
            "same_raag_raag_equal_weighted": {"mean": req, "n_raag": n_req},
            "n_raag_with_cross_artist_pairs": n_req,
            "raag_silhouette": sil,
            "retrieval": rk, "retrieval_chance": chance,
            "bootstrap": boot,
        }
        print("\n" + "=" * 92)
        print(f"SARAGA CONTROLS — {w}")
        print("=" * 92)
        for k, v in ps.items():
            print(f"  {k:26s} cos={v[0]:.4f}  (n={v[1]})")
        print(f"  rāg effect (S_R - D_B)   = {results['encoders'][w]['rāg_effect']:+.4f}")
        print(f"  artist effect (D_A - D_B)= {results['encoders'][w]['artist_effect']:+.4f}")
        print(f"  same-raag pair-weighted  = {ps['same_raag_cross_artist'][0]:.4f} | "
              f"rāg-equal-weighted = {req:.4f} (n_rāg={n_req})")
        print(f"  rāg silhouette = {sil:.4f} | "
              f"R@1={rk['R@1']:.3f}(chance {chance['R@1']:.3f}) "
              f"R@5={rk['R@5']:.3f}({chance['R@5']:.3f}) "
              f"R@10={rk['R@10']:.3f}({chance['R@10']:.3f})")
        for k, (m, lo, hi) in boot.items():
            print(f"  boot {k:26s} mean={m:+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}]", flush=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    outp = OUT / "saraga_controls.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {outp} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
