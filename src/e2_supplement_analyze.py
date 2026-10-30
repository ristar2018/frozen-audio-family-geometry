# -*- coding: utf-8 -*-
"""
FAPN E2 supplementary experiments, stage B: analysis (CPU only, no GPU; reads the stage-A cache)

Four strengthening blocks (each answers a weakness raised in review):
  Part 1  Held-out TEST generalisation (closes the "dev only" gap)
          prototypes/kNN built on dev -> evaluated on the 20% recording-isolated test split:
          family and 15-class accuracy, test family silhouette, within/between Delta-cos.
          Reports the four encoders plus the MFCC baseline; the MFCC scaler is fit on dev only (no leakage).

  Part 2  MFCC / chance / permutation baselines (give "high cos / high structure" a reference point)
          FolkMusic dev: MFCC sil/Delta-cos/acc; for every feature, 2000 family-label
          permutations give the empirical null (mean/q95/q99) of silhouette and Delta-cos plus an empirical p.

  Part 3  Saraga cross-cultural: reproduce the per-raag x cross-artist agreement (44 pairs)
          and add chance/permutation baselines plus an MFCC baseline, showing that 0.847-0.956 is not trivial.
          Also report the permutation significance of the coarse-family silhouette.

  Part 4  Downstream tasks (turn "healthy geometry" into a usable advantage)
          few-shot prototype classification with dev support and test queries
          (family, n_shot=1/2/4/8, mean and std over 20 repetitions);
          retrieval Recall@k with dev as gallery (family and instrument levels, k=1/5/10).

Output: results/e2_supplement/e2_supplement_results.json + console tables.

Usage:
  python e2_supplement_analyze.py
  python e2_supplement_analyze.py --n_perm 2000 --n_repeat 20
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from data_loader import split_folkmusic_dev_test, load_folkmusic_manifest, DEV_TEST_SEED  # noqa
from analysis import class_level_permutation_test  # noqa: same protocol as H1a: 15 class prototypes + restricted permutation

E1_EMB = PROJ / "results" / "e1" / "embeddings"
E2 = PROJ / "results" / "e2_supplement"
TEST_EMB = E2 / "embeddings"
MFCC_DIR = E2 / "mfcc"
SARAGA_EMB = PROJ / "results" / "saraga" / "embeddings"
SARAGA_MANIFEST = PROJ / "data" / "openethnic" / "saraga" / "manifest.csv"

ENCODERS = ["beats", "mert", "ast", "audiomae"]
ENC_LABEL = {"beats": "BEATs", "mert": "MERT-v2", "ast": "AST",
             "audiomae": "AudioMAE", "mfcc": "MFCC(240d)"}
KNN_K = 5
SHOTS = [1, 2, 4, 8]
RECALL_KS = [1, 5, 10]
DELTA_SAMPLE_Q = 200      # queries sampled per family (same protocol as the E1 visualisation)
SIL_SAMPLE = 6000


# ============================================================
# Shared geometry helpers
# ============================================================
def l2norm(X, eps=1e-8):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def proto_predict(Xtr, ytr, Xte):
    """Class-mean prototypes (after normalisation) + cosine nearest-centre assignment."""
    classes = np.array(sorted(np.unique(ytr)))
    P = np.stack([l2norm(Xtr[ytr == c].mean(axis=0, keepdims=True))[0]
                  for c in classes], axis=0)
    return classes[np.argmax(Xte @ P.T, axis=1)]


def knn_predict(Xtr, ytr, Xte, k=KNN_K):
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine",
                               weights="distance", algorithm="brute")
    knn.fit(Xtr, ytr)
    return knn.predict(Xte)


def delta_cos(X, y, rng, n_q=DELTA_SAMPLE_Q):
    """within/between cosine difference; n_q queries sampled per family (same protocol as E1)."""
    classes = np.unique(y)
    ins, outs = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        q = rng.choice(idx, size=min(n_q, len(idx)), replace=False)
        for i in q:
            same = idx[idx != i]
            diff = np.where(y != c)[0]
            ins.append(float((X[i] @ X[same].mean(axis=0))))
            outs.append(float((X[i] @ X[diff].mean(axis=0))))
    return float(np.mean(ins) - np.mean(outs))


def sil_fast(X, y, rng):
    if len(X) > SIL_SAMPLE:
        idx = rng.choice(len(X), SIL_SAMPLE, replace=False)
        X, y = X[idx], y[idx]
    return float(silhouette_score(X, y, metric="cosine"))


def perm_null_delta(X, y_fam, n_perm, rng, base_delta):
    """Empirical null of the within/between Delta-cos under a whole-label permutation of fragment-level family labels.

    Gives a chance reference for the geometry-margin strength Delta-cos (the significance of the silhouette goes
    through the H1a class-level restricted permutation instead, see class_level_permutation_test). Queries are fixed;
    the budget is S=X@X[qs].T and the within-family sums exclude the query itself (self-cosine = 1).
    """
    n = len(y_fam)
    classes = np.array(sorted(np.unique(y_fam)))
    qs = np.concatenate([
        rng.choice(np.where(y_fam == c)[0],
                   size=min(DELTA_SAMPLE_Q, (y_fam == c).sum()), replace=False)
        for c in classes])
    nq = len(qs)
    S = X @ X[qs].T
    col_sum_all = S.sum(axis=0)
    d_null = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y_fam)
        lab_q = yp[qs]
        within = np.empty(nq)
        between = np.empty(nq)
        for c in classes:
            qcols = np.where(lab_q == c)[0]
            memb = yp == c
            cnt_in = max(memb.sum() - 1, 1)
            cnt_out = max(n - memb.sum(), 1)
            block = S[memb][:, qcols]
            sum_in = block.sum(axis=0) - 1.0            # remove the query itself
            sum_all_c = col_sum_all[qcols]
            within[qcols] = sum_in / cnt_in
            between[qcols] = (sum_all_c - (sum_in + 1.0)) / cnt_out
        d_null[b] = within.mean() - between.mean()
    return {
        "delta_real": float(base_delta),
        "delta_null_mean": float(d_null.mean()),
        "delta_null_q95": float(np.quantile(d_null, .95)),
        "delta_null_q99": float(np.quantile(d_null, .99)),
        "delta_p": float((1 + np.sum(d_null >= base_delta)) / (n_perm + 1)),
    }


# ============================================================
# Feature loading
# ============================================================
def load_folk_feature(which, split):
    """which in ENCODERS+['mfcc']; split in {'dev','test'}. Returns (X, df[id,class,fam])."""
    if which == "mfcc":
        d = np.load(MFCC_DIR / f"mfcc_folkmusic_{split}.npz", allow_pickle=True)
        Xraw = d["features"].astype(np.float64)
        df = pd.DataFrame({"fragment_id": d["fragment_id"],
                           "instrument_class": d["instrument_class"],
                           "family": d["family"]})
        return Xraw, df, True  # needs standardisation
    path = (E1_EMB / f"{which}_folkmusic_dev.npz") if split == "dev" \
        else (TEST_EMB / f"{which}_folkmusic_test.npz")
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    df = pd.DataFrame({"fragment_id": d["fragment_id"],
                       "instrument_class": d["instrument_class"],
                       "family": d["family"]})
    return d["embeddings"].astype(np.float64), df, False


def prepare_features():
    """Returns dict: which -> dict(dev=(X,df), test=(X,df)); the MFCC scaler is fit on dev."""
    feats = {}
    # MFCC: fit the scaler on dev
    Xtr_raw, dftr, _ = load_folk_feature("mfcc", "dev")
    scaler = StandardScaler().fit(Xtr_raw)
    Xte_raw, dfte, _ = load_folk_feature("mfcc", "test")
    feats["mfcc"] = {
        "dev": (l2norm(scaler.transform(Xtr_raw)), dftr),
        "test": (l2norm(scaler.transform(Xte_raw)), dfte),
    }
    for e in ENCODERS:
        dev = load_folk_feature(e, "dev")
        test = load_folk_feature(e, "test")
        if test is None or test[0] is None:
            print(f"[warn] missing {e} test embeddings; Part1/Part4 will skip it (run stage A enc_test first)")
            if dev and dev[0] is not None:
                feats[e] = {"dev": (dev[0], dev[1]), "test": None}
            continue
        feats[e] = {"dev": (dev[0], dev[1]), "test": (test[0], test[1])}
    return feats


# ============================================================
# Part 1: held-out test
# ============================================================
def proto_sil_from(X, y_class, y_family):
    from sklearn.metrics import silhouette_score as _ss
    classes = np.array(sorted(np.unique(y_class)))
    P, pfam = [], []
    for c in classes:
        idx = np.where(y_class == c)[0]
        P.append(l2norm(X[idx].mean(axis=0, keepdims=True))[0])
        f = np.unique(y_family[idx])
        assert len(f) == 1, f"class {c} maps to multiple families {f}"
        pfam.append(f[0])
    P = np.stack(P, 0)
    return float(_ss(P, np.array(pfam), metric="cosine"))


def part1_heldout(feats, rng, n_perm):
    print("\n" + "=" * 96)
    print("Part 1  Held-out TEST generalisation (train on dev -> 20% recording-isolated test)")
    print("=" * 96)
    rows = []
    order = ["beats", "mert", "ast", "audiomae", "mfcc"]
    print(f"{'feature':12s} {'proto-fam':>10s} {'knn-fam':>9s} {'proto-15':>9s} "
          f"{'knn-15':>8s} {'sil-seg':>8s} {'sil-proto':>10s} {'H1a-p(test)':>11s} {'Δcos':>7s}")
    for w in order:
        if w not in feats or feats[w]["test"] is None:
            continue
        Xtr, dtr = feats[w]["dev"]
        Xte, dte = feats[w]["test"]
        ytr_f, yte_f = dtr["family"].to_numpy(), dte["family"].to_numpy()
        ytr_c, yte_c = dtr["instrument_class"].to_numpy(), dte["instrument_class"].to_numpy()
        pf = (proto_predict(Xtr, ytr_f, Xte) == yte_f).mean()
        kf = (knn_predict(Xtr, ytr_f, Xte) == yte_f).mean()
        pc = (proto_predict(Xtr, ytr_c, Xte) == yte_c).mean()
        kc = (knn_predict(Xtr, ytr_c, Xte) == yte_c).mean()
        sil_seg = sil_fast(Xte, yte_f, rng)
        sil_proto = proto_sil_from(Xte, yte_c, yte_f)
        # H1a on the test split: class-level restricted permutation, same protocol as the paper (15 prototypes, fast)
        h1a_te = class_level_permutation_test(
            Xte, yte_c, yte_f, n_permutations=n_perm, seed=20260501)
        dc = delta_cos(Xte, yte_f, rng)
        rows.append({"feature": w, "label": ENC_LABEL[w],
                     "test_n": int(len(Xte)),
                     "proto_acc_family": float(pf), "knn_acc_family": float(kf),
                     "proto_acc_15": float(pc), "knn_acc_15": float(kc),
                     "sil_family_segment_test": sil_seg,
                     "sil_family_prototype_test": sil_proto,
                     "h1a_test": {"sil_real": h1a_te["sil_real"],
                                  "sil_perm_q99": h1a_te["sil_perm_q99"],
                                  "p_value": h1a_te["p_value"],
                                  "passed_alpha_0.01": h1a_te["passed"]},
                     "delta_cos_test": dc})
        print(f"{ENC_LABEL[w]:12s} {pf:10.4f} {kf:9.4f} {pc:9.4f} {kc:8.4f} "
              f"{sil_seg:8.3f} {sil_proto:10.3f} {h1a_te['p_value']:11.4f} {dc:7.4f}")
    print("  sil-seg=fragment-level family silhouette; sil-proto/H1a=15-prototype restricted permutation (paper protocol); Delta-cos=between-family geometry margin")
    return rows


# ============================================================
# Part 2: baselines and permutation null on dev
# ============================================================
def part2_baselines_dev(feats, n_perm, rng):
    print("\n" + "=" * 86)
    print(f"Part 2  FolkMusic dev: MFCC/chance baselines (H1a class-level restricted permutation x{n_perm} + Delta-cos chance)")
    print("=" * 86)
    out = {}
    order = ["beats", "mert", "ast", "audiomae", "mfcc"]
    print(f"{'feature':12s} {'sil-proto':>10s} {'q99(null)':>10s} {'H1a-p':>8s} "
          f"{'Δcos':>7s} {'Δ-null':>8s} {'Δ-q99':>7s} {'Δ-p':>7s}")
    for w in order:
        if w not in feats:
            continue
        X, df = feats[w]["dev"]
        yf = df["family"].to_numpy()
        yc = df["instrument_class"].to_numpy()
        # same protocol as H1a: 15-prototype family silhouette + class-level restricted permutation
        h1a = class_level_permutation_test(
            X, yc, yf, n_permutations=n_perm, seed=20260501)
        base_delta = delta_cos(X, yf, rng)
        dnull = perm_null_delta(X, yf, n_perm,
                                np.random.default_rng(20260501), base_delta)
        stat = {
            "h1a_class_proto": {
                "sil_real": h1a["sil_real"], "sil_null_q99": h1a["sil_perm_q99"],
                "sil_null_mean": h1a["sil_perm_mean"], "p_value": h1a["p_value"],
                "passed_alpha_0.01": h1a["passed"]},
            "segment": {"sil_segment": sil_fast(X, yf, rng)},
            "delta_cos": dnull,
        }
        out[w] = stat
        print(f"{ENC_LABEL[w]:12s} {h1a['sil_real']:10.3f} {h1a['sil_perm_q99']:10.3f} "
              f"{h1a['p_value']:8.4f} {base_delta:7.4f} "
              f"{dnull['delta_null_mean']:8.4f} {dnull['delta_null_q99']:7.4f} "
              f"{dnull['delta_p']:7.4f}")
    print("  note: the restricted permutation preserves within-class cohesion (the correct null); an MFCC result that is also significant under H1a would mean the family structure is not specific to SSL.")
    return out


# ============================================================
# Part 3: Saraga replication + baselines
# ============================================================
RAAG_PATTERNS = None  # use the v2 family rules for the coarse silhouette


def _raag_families():
    import re
    pats = [
        ("meta", re.compile(r"^(intro|lyrics|speech|concept|shrutinandan|taal)")),
        ("bhairav", re.compile(r"bhairav|ahir bhair|nat bhairav|bhairabi|bhatiyar")),
        ("kalyan", re.compile(r"kalyan|jait kalyan|shuddha kalyan|shuddh kalyan|yaman")),
        ("malhar", re.compile(r"malhar|megh")),
        ("marwa", re.compile(r"^marwa|puriya|^shree|bibhas|lalit|lalat")),
        ("todi", re.compile(r"todi|multani|asavari|komal rishabh|bilaskhani")),
        ("kafi", re.compile(r"kafi|khamaj|bahar|maajh khamaj|mishra piloo|des\b|jhinjhoti")),
        ("bhairavi", re.compile(r"bhairavi|kauns|malkauns|chandrakauns|madhukauns")),
        ("bilawal", re.compile(r"bilawal|kedar|basanti|bageshree|hameer|kamod|sohini|suha|paraj|sawani|sohani")),
    ]
    def f(name):
        s = (name or "").strip().lower()
        for fam, pat in pats:
            if pat.search(s):
                return fam
        return "standalone"
    return f


def cross_artist_agreement(X, raag, artist):
    """
    Paper protocol (cross-checked and locked; 106 concerts / 44 cross-artist pairs after dropna):
      for every raag with >=2 artists, compute the (raag, artist) prototype (then L2),
      take the cosine between cross-artist prototype pairs inside that raag;
      agreement_mean = pair-equal-weighted mean over all 44 pairs (BEATs=0.847, AST=0.956);
      min_raag_mean  = raag-equal-weighted minimum over the per-raag pair means (BEATs=0.209).
    Both aggregations are returned so the two protocols cannot be confused.
    """
    dfx = pd.DataFrame({"raag": raag, "artist": artist}).reset_index(drop=True)
    raag_means, all_pairs = [], []
    for rg, g in dfx.groupby("raag"):
        arts = g["artist"].unique()
        if len(arts) < 2:
            continue
        prot = []
        for a in arts:
            idx = g.index[g["artist"] == a].to_numpy()  # already reset, = row index into X
            assert len(idx) > 0, f"empty (raag,artist) group: {rg}/{a}"
            prot.append(l2norm(X[idx].mean(axis=0, keepdims=True))[0])
        P = np.stack(prot, 0)
        C = P @ P.T
        iu = np.triu_indices(len(P), 1)
        raag_means.append(float(C[iu].mean()))
        all_pairs.extend(float(v) for v in C[iu].tolist())
    pair_total = len(all_pairs)
    return (float(np.mean(all_pairs)), int(pair_total), raag_means,
            float(np.min(raag_means)), float(np.mean(raag_means)))


def part3_saraga(n_perm, rng):
    print("\n" + "=" * 78)
    print("Part 3  Saraga cross-cultural: cross-artist raag agreement + chance/permutation/MFCC baselines")
    print("=" * 78)
    man = pd.read_csv(SARAGA_MANIFEST, encoding="utf-8-sig")
    raag0 = man["raag_common"].to_numpy()
    artist0 = man["lead_artist"].to_numpy().astype(object)
    # paper protocol: a missing lead_artist (2 rows "by Omkar Dadarkar", the same piece filed under two raags) cannot
    # be assigned to an artist prototype, so dropna removes them -> 106 concerts, 76 raags, 22 artists, 44 cross-artist pairs.
    valid = ~pd.isna(artist0)
    n_drop = int((~valid).sum())
    raag, artist = raag0[valid], artist0[valid]
    print(f"  [note] dropna removed {n_drop} rows with a missing lead_artist -> {valid.sum()} valid concerts / "
          f"{len(np.unique(raag))} raags / {len(np.unique(artist))} artists (paper protocol)")
    fam_of = _raag_families()
    fam = np.array([fam_of(r) for r in raag])

    # features: four encoders (already L2) + MFCC (standardised + L2, descriptive baseline); all aligned on the valid rows
    feat = {}
    for e in ENCODERS:
        feat[e] = np.load(SARAGA_EMB / f"{e}.npy").astype(np.float64)[valid]
    mfcc_raw = np.load(MFCC_DIR / "mfcc_saraga.npy").astype(np.float64)[valid]
    feat["mfcc"] = l2norm(StandardScaler().fit_transform(mfcc_raw))

    # chance reference: the same valid rows and the same cross-artist pair set, but with raag labels shuffled
    # (the null mean is the chance level; no separate global pairwise cosine is needed).
    out = {}
    print(f"{'feature':12s} {'agr(pair-wt)':>13s} {'#pairs':>7s} {'min/raag':>9s} "
          f"{'agr(raag-wt)':>13s} {'null-mean':>10s} {'agr-p':>7s} "
          f"{'sil-fam':>8s} {'sil-p':>7s}")
    for w, X in feat.items():
        agr, npairs, raag_means, mn, agr_rw = cross_artist_agreement(X, raag, artist)
        # null: keep the per-raag artist-count structure, only shuffle the raag labels (106 scale, recomputed each time)
        null_agr = np.empty(n_perm)
        for b in range(n_perm):
            rp = rng.permutation(raag)
            null_agr[b], _, _, _, _ = cross_artist_agreement(X, rp, artist)
        p_agr = float((1 + np.sum(null_agr >= agr)) / (n_perm + 1))
        # coarse thaat-family silhouette + permutation (valid rows)
        sil = float(silhouette_score(X, fam, metric="cosine"))
        n_sil = min(n_perm, 500)
        sil_null = np.array([
            silhouette_score(X, rng.permutation(fam), metric="cosine")
            for _ in range(n_sil)])
        p_sil = float((1 + np.sum(sil_null >= sil)) / (n_sil + 1))
        out[w] = {"agreement_cross_artist_pair_weighted": agr,
                  "agreement_cross_artist_raag_weighted": agr_rw,
                  "n_cross_artist_pairs": npairs,
                  "min_raag_mean": mn,
                  "chance_null_mean": float(null_agr.mean()),
                  "agreement_null_q99": float(np.quantile(null_agr, .99)),
                  "agreement_p": p_agr,
                  "sil_coarse_family": sil,
                  "sil_coarse_p_vs_perm": p_sil,
                  "per_raag_means": raag_means}
        print(f"{ENC_LABEL[w]:12s} {agr:13.3f} {npairs:7d} {mn:9.3f} {agr_rw:13.3f} "
              f"{null_agr.mean():10.3f} {p_agr:7.4f} {sil:8.3f} {p_sil:7.4f}")
    print(f"  ({valid.sum()} valid concerts, {len(np.unique(raag))} raags, "
          f"{len(np.unique(artist))} artists; the paper's 0.847~0.956 is the pair-weighted value)")
    return out


# ============================================================
# Part 4: few-shot + retrieval
# ============================================================
def part4_downstream(feats, n_repeat, rng):
    print("\n" + "=" * 78)
    print(f"Part 4  Downstream tasks: few-shot prototype classification ({n_repeat} repetitions) + retrieval Recall@k")
    print("=" * 78)
    out = {}
    order = ["beats", "mert", "ast", "audiomae", "mfcc"]
    avail = [w for w in order if w in feats and feats[w]["test"] is not None]

    # ---- 4a few-shot family prototype ----
    print(f"\n[4a] few-shot family accuracy (dev support, all test fragments as queries)")
    header = "feature     " + "".join(f"  {s}-shot" for s in SHOTS)
    print(header)
    for w in avail:
        Xtr, dtr = feats[w]["dev"]
        Xte, dte = feats[w]["test"]
        ytr, yte = dtr["family"].to_numpy(), dte["family"].to_numpy()
        classes = np.array(sorted(np.unique(ytr)))
        means = {s: [] for s in SHOTS}
        stds = {s: [] for s in SHOTS}
        for s in SHOTS:
            accs = []
            for rep in range(n_repeat):
                rng_r = np.random.default_rng(20260501 + rep)
                sel = []
                for c in classes:
                    idx = np.where(ytr == c)[0]
                    sel.append(rng_r.choice(idx, size=min(s, len(idx)), replace=False))
                sel = np.concatenate(sel)
                pred = proto_predict(Xtr[sel], ytr[sel], Xte)
                accs.append((pred == yte).mean())
            means[s] = float(np.mean(accs)); stds[s] = float(np.std(accs))
        out.setdefault(w, {})["fewshot_family"] = {
            f"{s}": {"mean": means[s], "std": stds[s]} for s in SHOTS}
        print(f"{ENC_LABEL[w]:12s}" + "".join(f"  {means[s]:5.3f}" for s in SHOTS))
    print("  (std in the JSON)")

    # ---- 4b retrieval Recall@k (dev = gallery, test = query) ----
    print(f"\n[4b] retrieval Recall@k (dev gallery -> test query; family / instrument levels)")
    print(f"{'feature':12s}" + "".join(f"  R@{k}-fam" for k in RECALL_KS)
          + "".join(f"  R@{k}-ins" for k in RECALL_KS))
    for w in avail:
        Xg, dg = feats[w]["dev"]
        Xq, dq = feats[w]["test"]
        yg_f, yq_f = dg["family"].to_numpy(), dq["family"].to_numpy()
        yg_c, yq_c = dg["instrument_class"].to_numpy(), dq["instrument_class"].to_numpy()
        S = Xq @ Xg.T  # (nq, ng)
        kmax = max(RECALL_KS)
        top = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
        row = {}
        cells = []
        for level, yg, yq in [("family", yg_f, yq_f), ("instrument", yg_c, yq_c)]:
            for k in RECALL_KS:
                hit = np.array([np.any(yg[top[q, :k]] == yq[q]) for q in range(len(Xq))])
                v = float(hit.mean())
                row[f"R@{k}_{level}"] = v
                cells.append(v)
        out[w]["retrieval"] = row
        print(f"{ENC_LABEL[w]:12s}" + "".join(f"  {v:7.3f}" for v in cells[:3])
              + "".join(f"  {v:7.3f}" for v in cells[3:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--n_repeat", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(20260501)
    E2.mkdir(parents=True, exist_ok=True)

    # verify the split still matches E1 (compare the number of test rows)
    fm = load_folkmusic_manifest(verify_files=False)
    _, fm_test = split_folkmusic_dev_test(fm, test_size=0.2, seed=DEV_TEST_SEED)
    print(f"[init] reproduced test={len(fm_test)} fragments (expected 2393)")

    feats = prepare_features()

    results = {
        "experiment": "E2_supplement",
        "n_permutations": args.n_perm,
        "fewshot_repeats": args.n_repeat,
        "part1_heldout_test": part1_heldout(feats, rng, args.n_perm),
        "part2_dev_baselines_perm": part2_baselines_dev(feats, args.n_perm, rng),
    }
    if (MFCC_DIR / "mfcc_saraga.npy").exists():
        results["part3_saraga"] = part3_saraga(args.n_perm, rng)
    else:
        print("\n[skip] Part3: mfcc_saraga.npy is missing (stage A was run with --skip_saraga_mfcc)")
    results["part4_downstream"] = part4_downstream(feats, args.n_repeat, rng)

    dst = E2 / "e2_supplement_results.json"
    dst.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                              default=lambda o: float(o) if isinstance(
                                  o, (np.floating, np.integer)) else o.tolist()))
    print(f"\nSaved -> {dst}")


if __name__ == "__main__":
    main()
