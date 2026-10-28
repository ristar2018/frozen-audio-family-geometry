# -*- coding: utf-8 -*-
"""
Route A (supplementary experiment): the Saraga 1.5 Carnatic "ensemble mix vs solo stem" paired experiment.

Motivation (state of the paper): Section 4.8 reports Saraga 1.5 Hindustani as a negative control --
ensemble recordings, no family labels, and a very high similarity floor (0.93-0.96 even across rag and across performer),
so the rag structure cannot be resolved. A natural review question is whether this means "the encoder does not encode
musical structure" or "the ensemble mix washes the structure out".

This experiment uses the stems of the same item in Saraga 1.5 Carnatic as a paired design:
  mix (ensemble mix) / violin (solo stem, bowed) / vocal (vocal stem)
For each representation, the same statistics as the Saraga probe of the paper:
  1) four-way decomposition of pairwise similarity: S_R (same rag, cross-performer), S_RA (same rag, same performer),
     D_A (different rag, same performer), D_B (different rag, cross-performer, i.e. the similarity floor)
     rag effect = S_R - D_B; performer effect = D_A - D_B
  2) rag silhouette (cosine)
  3) rag retrieval R@1 (query -> the rest of the gallery, hit = same rag) and majority-rag chance
  4) rag-cluster bootstrap 95% CI (resampling rags together with their recordings; SEED=20260501, B=2000)
  5) paired difference: the 95% CI of Delta = stat(stem) - stat(mix) under the same resampling
     (the core of this experiment: does the mix significantly depress the structural statistics)

Honest boundaries (must be reported together with the results):
  - Carnatic != Hindustani: this experiment tests the mechanism of the recording format (ensemble mix) and
    cannot directly overturn or rewrite the conclusion of the paper on the Hindustani negative control.
  - The Carnatic stems only contain violin (bowed) / vocal / mridangam / ghatam,
    with no wind or plucked melodic stem, so the 3-family test of the paper cannot be reproduced.
  - The stems are not perfectly isolated (there is bleed); the Sanidha paper quantifies bleed in the Saraga stems.

Audio protocol identical to the existing pipeline: encoders max_sec=20 s (the encoders.load_audio_mono default),
MFCC 16 kHz/max_sec=10 s, n_mfcc=40 + delta + delta-delta frame mean/std (240 dims).

Output: results/e4/carnatic_mix_vs_stem.json (+ console report)
"""
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))

ROOT = PROJ
ZIP = ROOT / "data" / "supp" / "saraga-carnatic" / "saraga1.5_carnatic.zip"
STEM_DIR = ROOT / "data" / "supp" / "saraga-carnatic" / "extracted"
OUT = ROOT / "results" / "e4"
EMB_OUT = OUT / "embeddings"
MFCC_OUT = OUT / "mfcc"
MANIFEST = OUT / "carnatic_stems_manifest.csv"

ENCODERS = ["beats", "mert", "ast", "audiomae"]
REPS = ["mix", "violin", "vocal"]
MFCC_SR = 16000
MFCC_MAX_SEC = 10.0
N_MFCC = 40
N_BOOT = 2000
SEED = 20260501
RNG = np.random.default_rng(SEED)


# ============================================================
# 1) extract only the needed files from the archive (mix / violin / vocal)
# ============================================================
def extract_needed(df):
    need = []
    for _, r in df.iterrows():
        for rep, col in (("mix", "mix_path"), ("violin", "violin_path"), ("vocal", "vocal_path")):
            p = str(r[col]).strip()
            if p:
                need.append(p)
    need = sorted(set(need))
    todo = [p for p in need if not (STEM_DIR / p).exists()]
    if todo:
        print(f"[unzip] {len(todo)}/{len(need)} files to extract -> {STEM_DIR}")
        z = zipfile.ZipFile(ZIP)
        t0 = time.time()
        for i, p in enumerate(todo, 1):
            dst = STEM_DIR / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(p) as src, open(dst, "wb") as fh:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            if i % 100 == 0:
                print(f"   {i}/{len(todo)}  {time.time()-t0:.0f}s")
        print(f"[unzip] done {time.time()-t0:.0f}s")
    else:
        print(f"[unzip] all present ({len(need)} files)")
    return {rep: [str(STEM_DIR / str(r[col]).strip()) if str(r[col]).strip() else ""
                  for _, r in df.iterrows()]
            for rep, col in (("mix", "mix_path"), ("violin", "violin_path"), ("vocal", "vocal_path"))}


# ============================================================
# 2) features: four encoders + MFCC (per representation)
# ============================================================
def extract_encoders(paths_by_rep):
    from encoders import create_encoder
    EMB_OUT.mkdir(parents=True, exist_ok=True)
    for rep in REPS:
        paths = paths_by_rep[rep]
        for name in ENCODERS:
            dst = EMB_OUT / f"{name}_{rep}.npy"
            msk = EMB_OUT / f"{name}_{rep}_ok.npy"
            if dst.exists() and msk.exists():
                print(f"[enc] {rep}/{name}: cached {np.load(dst).shape}")
                continue
            print(f"\n[enc] {rep}/{name}: {len(paths)} files")
            t0 = time.time()
            enc = create_encoder(name)
            r = enc.encode_dataset(paths, batch_size=None, log_every=100)
            if r["failed"]:
                print(f"   [warn] {len(r['failed'])} failed: {r['failed'][:3]}")
            X = r["embeddings"]
            ok = np.zeros(len(paths), dtype=bool)
            ok[np.asarray(r["ok_idx"], dtype=int)] = True
            np.save(dst, X)
            np.save(msk, ok)
            print(f"[enc] {rep}/{name} {X.shape} ok={ok.sum()}/{len(paths)} {time.time()-t0:.0f}s")
            del enc


def mfcc_vector(path):
    from encoders import load_audio_mono
    import librosa
    y = load_audio_mono(path, MFCC_SR, max_sec=MFCC_MAX_SEC)
    M = librosa.feature.mfcc(y=y, sr=MFCC_SR, n_mfcc=N_MFCC, n_fft=1024, hop_length=160)
    d1 = librosa.feature.delta(M)
    d2 = librosa.feature.delta(M, order=2)
    return np.concatenate([M.mean(1), M.std(1), d1.mean(1), d1.std(1),
                           d2.mean(1), d2.std(1)]).astype(np.float32)


def extract_mfcc(paths_by_rep):
    MFCC_OUT.mkdir(parents=True, exist_ok=True)
    for rep in REPS:
        dst = MFCC_OUT / f"mfcc_{rep}.npy"
        msk = MFCC_OUT / f"mfcc_{rep}_ok.npy"
        if dst.exists() and msk.exists():
            print(f"[mfcc] {rep}: cached {np.load(dst).shape}")
            continue
        t0 = time.time()
        rows, ok = [], np.zeros(len(paths_by_rep[rep]), dtype=bool)
        for i, p in enumerate(paths_by_rep[rep]):
            try:
                rows.append(mfcc_vector(p))
                ok[i] = True
            except Exception as ex:
                print(f"   [warn] mfcc fail {p}: {ex!r}")
                rows.append(np.full(6 * N_MFCC, np.nan, dtype=np.float32))
        X = np.stack(rows)
        np.save(dst, X)
        np.save(msk, ok)
        print(f"[mfcc] {rep} {X.shape} ok={ok.sum()}/{len(ok)} {time.time()-t0:.0f}s")


# ============================================================
# 3) statistics (same protocol as e3_saraga_controls)
# ============================================================
def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def pair_stats(X, raag, artist):
    S = X @ X.T
    n = len(raag)
    iu = np.triu_indices(n, 1)
    s = S[iu]
    sr = raag[iu[0]] == raag[iu[1]]
    sa = artist[iu[0]] == artist[iu[1]]
    out = {
        "S_R_same_raag_cross_artist": (float(s[sr & ~sa].mean()), int((sr & ~sa).sum())),
        "S_RA_same_raag_same_artist": (float(s[sr & sa].mean()) if (sr & sa).sum() else float("nan"),
                                       int((sr & sa).sum())),
        "D_A_diff_raag_same_artist": (float(s[~sr & sa].mean()) if (~sr & sa).sum() else float("nan"),
                                      int((~sr & sa).sum())),
        "D_B_diff_raag_cross_artist": (float(s[~sr & ~sa].mean()), int((~sr & ~sa).sum())),
    }
    raag_eff = out["S_R_same_raag_cross_artist"][0] - out["D_B_diff_raag_cross_artist"][0]
    art_eff = (out["D_A_diff_raag_same_artist"][0] - out["D_B_diff_raag_cross_artist"][0]
               if not np.isnan(out["D_A_diff_raag_same_artist"][0]) else float("nan"))
    return out, float(raag_eff), float(art_eff), S, iu, sr, sa


def raag_silhouette(X, raag):
    from sklearn.metrics import silhouette_score
    out = {}
    if len(set(raag)) > 1 and len(set(raag)) < len(X):
        out["sil_all_labels"] = float(silhouette_score(X, raag, metric="cosine"))
    cnt = pd.Series(raag).value_counts()
    keep = np.isin(raag, cnt[cnt >= 2].index.to_numpy())
    r2 = raag[keep]
    if len(set(r2)) > 1 and len(set(r2)) < int(keep.sum()):
        out["sil_repeated_raagas"] = float(silhouette_score(X[keep], r2, metric="cosine"))
        out["n_items_repeated"] = int(keep.sum())
        out["n_raagas_repeated"] = int(len(set(r2)))
    return out


def raag_retrieval(X, raag):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    top = np.argmax(S, axis=1)
    hit = (raag[top] == raag).mean()
    vc = pd.Series(raag).value_counts()
    chance = float(((vc / vc.sum()) ** 2).sum())
    return float(hit), float(chance)


def boot_ci(values_fn, raags, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    uniq = np.unique(raags)
    idx_by_raag = {r: np.where(raags == r)[0] for r in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_raag[r] for r in pick])
        v = values_fn(idx)
        if v is not None and not np.isnan(v):
            stats.append(v)
    if not stats:
        return [float("nan")] * 3
    a = np.asarray(stats)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)), float(np.percentile(a, 97.5))]


def main():
    df = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    df = df[(df.has_violin == 1) & (df.has_mix == 1) & (df.raaga.notna())].reset_index(drop=True)
    df = df[df.vocal_path.notna() & (df.vocal_path.astype(str).str.len() > 0)].reset_index(drop=True)
    print(f"[data] paired items (with mix/violin/vocal and a rag label) = {len(df)}")
    print(f"        rāg={df.raaga.nunique()}  artist={df.lead_artist.nunique()}  "
          f"duration={df.duration_s.sum()/3600:.1f} h per representation")

    paths_by_rep = extract_needed(df)
    extract_encoders(paths_by_rep)
    extract_mfcc(paths_by_rep)

    raag_all = df.raaga.to_numpy()
    artist_all = df.lead_artist.to_numpy()

    # features: encoders (L2-normalised) + MFCC (mean-variance standardisation -> L2)
    # first take the items where every (representation x feature) succeeded, so mix/violin/vocal pair up row by row
    feats, masks = {}, {}
    for rep in REPS:
        feats[rep] = {}
        masks[rep] = {}
        for name in ENCODERS:
            feats[rep][name] = l2norm(np.load(EMB_OUT / f"{name}_{rep}.npy"))
            masks[rep][name] = np.load(EMB_OUT / f"{name}_{rep}_ok.npy")
        Xm = np.load(MFCC_OUT / f"mfcc_{rep}.npy").astype(np.float64)
        masks[rep]["mfcc"] = np.load(MFCC_OUT / f"mfcc_{rep}_ok.npy")
        mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0)
        sd[sd == 0] = 1.0
        Xm = np.nan_to_num((Xm - mu) / sd, nan=0.0)
        feats[rep]["mfcc"] = l2norm(Xm)

    common = np.ones(len(df), dtype=bool)
    for rep in REPS:
        for k, m in masks[rep].items():
            if len(m) != len(df):
                raise AssertionError(f"mask length mismatch {rep}/{k}: {len(m)} vs {len(df)}")
            common &= m
    print(f"[align] common valid items = {common.sum()}/{len(df)}"
          f" (dropped {len(df)-common.sum()}: extraction failed for at least one representation/feature)")
    if common.sum() < len(df):
        drop = df.loc[~common, "item"].tolist()
        print(f"        dropped: {drop[:10]}")
        df = df[common].reset_index(drop=True)
        for rep in REPS:
            for k in feats[rep]:
                feats[rep][k] = feats[rep][k][common]
    raag = df.raaga.to_numpy()
    artist = df.lead_artist.to_numpy()
    print(f"[align] final n={len(df)}  rag={df.raaga.nunique()}  artist={df.lead_artist.nunique()}")

    result = {"meta": {
        "zip": str(ZIP), "n_items": int(len(df)), "n_raaga": int(df.raaga.nunique()),
        "n_artists": int(df.lead_artist.nunique()),
        "audio_hours_per_rep": round(float(df.duration_s.sum() / 3600), 2),
        "encoder_max_sec": 20.0, "mfcc_max_sec": MFCC_MAX_SEC,
        "n_boot": N_BOOT, "seed": SEED,
        "caveat": ("Carnatic != Hindustani; the stems only cover violin (bowed), vocal and percussion, "
                   "there is no wind or plucked melodic stem, so the 3-family test of the paper cannot be reproduced; "
                   "the stems carry bleed. This experiment tests the mechanism of the ensemble-mix format."),
    }, "encoders": {}}

    for name in ENCODERS + ["mfcc"]:
        entry = {}
        for rep in REPS:
            X = feats[rep][name]
            ps, reff, aeff, S, iu, sr, sa = pair_stats(X, raag, artist)
            entry[rep] = {
                "pair_stats": {k: {"mean": v[0], "n_pairs": v[1]} for k, v in ps.items()},
                "raag_effect": reff, "artist_effect": aeff,
                **raag_silhouette(X, raag),
                "raag_retrieval_R1": raag_retrieval(X, raag)[0],
                "raag_retrieval_chance": raag_retrieval(X, raag)[1],
            }
        # paired bootstrap: Delta = stem - mix (under the same rag resampling)
        for rep in ("violin", "vocal"):
            Xm_ = feats["mix"][name]
            Xs_ = feats[rep][name]
            def delta(idx, Xm_=Xm_, Xs_=Xs_):
                r, a = raag[idx], artist[idx]
                _, reff_m, _, _, _, _, _ = pair_stats(l2norm(Xm_[idx]), r, a)
                _, reff_s, _, _, _, _, _ = pair_stats(l2norm(Xs_[idx]), r, a)
                return reff_s - reff_m
            lo, med, hi = boot_ci(delta, raag)
            entry[f"delta_raag_effect_{rep}_minus_mix"] = {"lo": lo, "med": med, "hi": hi,
                                                           "excludes_zero": bool(lo > 0 or hi < 0)}
            def dfloor(idx, Xm_=Xm_, Xs_=Xs_):
                r, a = raag[idx], artist[idx]
                pm, _, _, _, _, _, _ = pair_stats(l2norm(Xm_[idx]), r, a)
                psx, _, _, _, _, _, _ = pair_stats(l2norm(Xs_[idx]), r, a)
                return (psx["D_B_diff_raag_cross_artist"][0] - pm["D_B_diff_raag_cross_artist"][0])
            lo, med, hi = boot_ci(dfloor, raag)
            entry[f"delta_floor_{rep}_minus_mix"] = {"lo": lo, "med": med, "hi": hi,
                                                     "excludes_zero": bool(lo > 0 or hi < 0)}
        result["encoders"][name] = entry
        print(f"\n=== {name} ===")
        for rep in REPS:
            e = entry[rep]
            print(f"  {rep:6s} floor(D_B)={e['pair_stats']['D_B_diff_raag_cross_artist']['mean']:.3f} "
                  f"S_R={e['pair_stats']['S_R_same_raag_cross_artist']['mean']:.3f} "
                  f"raag_eff={e['raag_effect']:+.3f} art_eff={e['artist_effect']:+.3f} "
                  f"sil_rep={e.get('sil_repeated_raagas', float('nan')):+.3f} "
                  f"R@1={e['raag_retrieval_R1']:.3f}")
        for rep in ("violin", "vocal"):
            d = entry[f"delta_raag_effect_{rep}_minus_mix"]
            f = entry[f"delta_floor_{rep}_minus_mix"]
            print(f"  Δ raag_eff ({rep}-mix) = {d['med']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] "
                  f"excl0={d['excludes_zero']} | Δ floor = {f['med']:+.3f} "
                  f"[{f['lo']:+.3f},{f['hi']:+.3f}] excl0={f['excludes_zero']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "carnatic_mix_vs_stem.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {OUT / 'carnatic_mix_vs_stem.json'}")


if __name__ == "__main__":
    main()
