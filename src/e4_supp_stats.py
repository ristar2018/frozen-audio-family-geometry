# -*- coding: utf-8 -*-
"""
Route B/C statistics: geometry and discriminability tests on the four supplementary corpora (same protocol as the paper).

Corpora and tests:
  medley   : 3 s solo excerpts (8 instrument classes) --
             (a) class level: 8-class silhouette (label-permutation calibrated) + track-disjoint kNN LOO
             (b) family level: 3-family Delta-cos (family-equal weight) + family silhouette + family kNN LOO
             (c) cross-collection: R@1 with the official training/validation split (MedleyDB) as gallery and
                 test (solosDB) as query
  brid     : Brazilian percussion (one family, 10 instruments x 3 performers) --
             instrument silhouette (permutation calibrated) + cross-performer kNN LOO (neighbours restricted to other performers)
  beijing  : Beijing opera percussion (one family, 4 instruments) -- as above
  mridangam: one instrument x 6 tonics x 10 strokes --
             tonic silhouette against stroke silhouette (each permutation calibrated) +
             Delta-cos(tonic) / Delta-cos(stroke) + kNN LOO hitting tonic / stroke

Statistic definitions identical to the paper: silhouette on cosine distance; kNN LOO takes the maximum cosine and excludes
near-duplicates from the same group; Delta-cos loops over families (or labels) with equal weight.
Output: results/e4/supp_controls.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))

SUPP = PROJ / "data" / "supp"
OUT = PROJ / "results" / "e4" / "supp"
RES = PROJ / "results" / "e4"
ENCODERS = ["beats", "mert", "ast", "audiomae"]
FEATS = ENCODERS + ["mfcc"]
N_PERM = 2000
SEED = 20260501


def l2norm(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


# ---------- silhouette (on a precomputed distance matrix, so it can be permutation calibrated) ----------
def sil_prep(D, keep=None):
    """Precompute, for every point and every class, the distance sum and the number of valid pairs.

    keep[i, j] True means the pair (i, j) enters the silhouette (None = everything except self).
    Near-duplicate removal (same song / same performer / same tonic) depends only on the grouping, not on the labels,
    so keep is invariant under label permutation, which is what makes the permutation calibration valid.
    """
    n = D.shape[0]
    if keep is None:
        keep = ~np.eye(n, dtype=bool)
    Dm = np.where(keep, D, 0.0)
    return Dm, keep


def sil_from_prep(Dm, keep, y):
    n = Dm.shape[0]
    uniq, inv = np.unique(y, return_inverse=True)
    k = len(uniq)
    M = np.empty((n, k))
    cnt = np.empty((n, k))
    for j in range(k):
        ii = np.where(inv == j)[0]
        M[:, j] = Dm[:, ii].sum(1)
        cnt[:, j] = keep[:, ii].sum(1)
    a = M[np.arange(n), inv] / np.maximum(cnt[np.arange(n), inv], 1)
    Mb = np.where(cnt > 0, M / np.maximum(cnt, 1.0), np.inf)
    Mb[np.arange(n), inv] = np.inf
    b = Mb.min(1)
    b = np.where(np.isfinite(b), b, a)            # degenerate to 0 when no other class is available
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / np.where(denom == 0, 1.0, denom), 0.0)
    return float(s.mean())


def sil_from_dist(D, y, keep=None):
    Dm, keep = sil_prep(D, keep)
    return sil_from_prep(Dm, keep, y)


def sil_perm_test(D, y, n_perm=N_PERM, rng=None, keep=None):
    rng = rng or np.random.default_rng(SEED)
    Dm, keep = sil_prep(D, keep)
    real = sil_from_prep(Dm, keep, y)
    null = np.empty(n_perm)
    yy = y.copy()
    for i in range(n_perm):
        rng.shuffle(yy)
        null[i] = sil_from_prep(Dm, keep, yy)
    p_ge = (np.sum(null >= real) + 1) / (n_perm + 1)
    return real, float(p_ge), float(null.mean()), float(np.percentile(null, 95))


# ---------- kNN LOO ----------
def knn_loo(S, labels, groups=None, exclude_same_group=True):
    n = S.shape[0]
    M = S.copy()
    np.fill_diagonal(M, -np.inf)
    if groups is not None and exclude_same_group and len(np.unique(groups)) > 1:
        same = groups[:, None] == groups[None, :]
        M[same] = -np.inf
    valid = np.isfinite(M).any(1)
    nn = np.argmax(np.where(np.isfinite(M), M, -np.inf), axis=1)
    hit = (labels[nn] == labels) & valid
    # random-neighbour baseline: the share of same-class neighbours among those available to each query
    exp = []
    for i in np.where(valid)[0]:
        avail = np.isfinite(M[i])
        if avail.sum() == 0:
            continue
        exp.append((labels[avail] == labels[i]).mean())
    return float(hit.mean()), float(np.mean(exp)) if exp else float("nan"), int(valid.sum())


# ---------- Delta-cos (label-equal weight) ----------
def dcos_equal_weight(S, labels):
    uniq = np.unique(labels)
    within, between = [], []
    for u in uniq:
        m = labels == u
        w = S[np.ix_(m, m)]
        iu = np.triu_indices(m.sum(), 1)
        within.append(float(w[iu].mean()))
        other = ~m
        between.append(float(S[np.ix_(m, other)].mean()))
    return float(np.mean(within)), float(np.mean(between)), \
        float(np.mean(within) - np.mean(between))


def load_features(corpus):
    """Return {feat: (X_norm, idx)} where idx is the manifest row index (ascending) of the successful rows of that feature.
    Encoders store only the successful rows (shape = number of ok rows) while MFCC stores every row (failed rows NaN), so both are
    normalised to the "take rows by idx" form to avoid misalignment."""
    feats = {}
    for f in FEATS:
        p = OUT / f"{f}_{corpus}.npy"
        if not p.exists():
            print(f"   [skip] missing {p.name}")
            continue
        X = np.load(p).astype(np.float64)
        m = np.load(OUT / f"{f}_{corpus}_ok.npy").astype(bool)
        idx = np.where(m)[0]
        if len(X) != len(m):
            assert len(X) == int(m.sum()), f"{corpus}/{f}: rows {len(X)} != ok count {int(m.sum())}"
        else:
            X = X[idx]
        if f == "mfcc":
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
            sd[sd == 0] = 1.0
            X = np.nan_to_num((X - mu) / sd, nan=0.0)
        feats[f] = (l2norm(X), idx)
    return feats


def centred(X):
    return l2norm(X - X.mean(0, keepdims=True))


def analyse(corpus, label_col, group_col=None, extra=None, transfer=None):
    mpath = OUT / f"{corpus}_manifest.csv"
    df = pd.read_csv(mpath, encoding="utf-8-sig")
    feats = load_features(corpus)
    if not feats:
        return {}
    common = None
    for _, idx in feats.values():
        common = idx if common is None else np.intersect1d(common, idx)
    print(f"[{corpus}] valid {len(common)}/{len(df)} (common valid rows)")
    df = df.iloc[common].reset_index(drop=True)
    out = {"n": int(len(df)), "features": {}}
    for f, (X, idx) in feats.items():
        pos = np.searchsorted(idx, common)
        assert (idx[pos] == common).all(), f"{corpus}/{f}: row alignment failed"
        Xs = X[pos]
        y = df[label_col].to_numpy()
        g = df[group_col].to_numpy() if group_col else None
        S = Xs @ Xs.T
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        rng = np.random.default_rng(SEED)
        n_perm = 500 if len(df) > 1000 else N_PERM
        sil, p, nul, nul95 = sil_perm_test(D, y.copy(), n_perm=n_perm, rng=rng)
        acc, exp, nv = knn_loo(S, y, g)
        w, b, d = dcos_equal_weight(S, y)
        # robustness: remove the global mean before taking the cosine (removes the effect of anisotropy on the silhouette)
        Xc = centred(Xs)
        Sc = Xc @ Xc.T
        Dc = 1.0 - Sc
        np.fill_diagonal(Dc, 0.0)
        sil_c = sil_from_dist(Dc, y)
        acc_c, exp_c, _ = knn_loo(Sc, y, g)
        # robustness: silhouette after removing same-group pairs (same song / performer / tonic); descriptive, no permutation
        sil_g = None
        if g is not None and len(np.unique(g)) > 1 and not np.array_equal(
                np.unique(g, return_inverse=True)[1], np.unique(y, return_inverse=True)[1]):
            keep_g = g[:, None] != g[None, :]
            same_lab = y[:, None] == y[None, :]
            if (keep_g & same_lab).sum(1).min() > 0:
                sil_g = sil_from_dist(D, y, keep=keep_g)
        e = {"n": int(len(df)), "n_classes": int(len(np.unique(y))),
             "n_perm": n_perm,
             "silhouette": sil, "sil_p_ge": p, "sil_null_mean": nul,
             "sil_null_p95": nul95,
             "silhouette_centred": sil_c,
             "silhouette_grp_disjoint": sil_g,
             "knn_loo_acc": acc, "knn_loo_chance": exp, "knn_loo_n": nv,
             "knn_loo_acc_centred": acc_c,
             "dcos_within": w, "dcos_between": b, "dcos": d}
        if extra is not None:
            e.update(extra(df, Xs))
        out["features"][f] = e
        print(f"   {f:8s} sil={sil:+.3f} (p={p:.4f}, null95={nul95:+.3f}) "
              f"kNN={acc:.3f} vs {exp:.3f}  dcos={d:+.3f}")
        if transfer is not None:
            tr = transfer(df, Xs)
            out["features"][f].update(tr)
            print(f"   {f:8s} transfer: " + ", ".join(f"{k}={v:.3f}" for k, v in tr.items()))
    return out


def medley_extra(df, X):
    S = X @ X.T
    fam = df["family"].to_numpy()
    m3 = np.isin(fam, ["wind", "bowed", "plucked"])
    if m3.sum() == 0:
        return {}
    S3 = S[np.ix_(m3, m3)]
    y3 = fam[m3]
    g3 = df["group"].to_numpy()[m3]
    w, b, d = dcos_equal_weight(S3, y3)
    acc, exp, _ = knn_loo(S3, y3, g3)
    D3 = 1.0 - S3
    np.fill_diagonal(D3, 0.0)
    sil3 = sil_from_dist(D3, y3)
    X3c = centred(X[np.ix_(m3, np.arange(X.shape[1]))])
    S3c = X3c @ X3c.T
    D3c = 1.0 - S3c
    np.fill_diagonal(D3c, 0.0)
    acc3c, _, _ = knn_loo(S3c, y3, g3)
    return {"n_family3": int(m3.sum()),
            "family3_silhouette": sil3, "family3_silhouette_centred": sil_from_dist(D3c, y3),
            "family3_dcos": d,
            "family3_dcos_within": w, "family3_dcos_between": b,
            "family3_knn_loo_acc": acc, "family3_knn_loo_acc_centred": acc3c,
            "family3_knn_loo_chance": exp,
            "family3_counts": pd.Series(y3).value_counts().to_dict()}


def medley_transfer(df, X):
    gal = df["split"].isin(["training", "validation"]).to_numpy()
    qry = (df["split"] == "test").to_numpy()
    if gal.sum() == 0 or qry.sum() == 0:
        return {}
    S = X[qry] @ X[gal].T
    nn = np.argmax(S, axis=1)
    inst = (df["class_label"].to_numpy()[gal][nn] == df["class_label"].to_numpy()[qry]).mean()
    fam_g = df["family"].to_numpy()[gal]
    fam_q = df["family"].to_numpy()[qry]
    famhit = (fam_g[nn] == fam_q).mean()
    # robustness: centre on the gallery mean before computing (only gallery statistics are used for centring)
    Xc = centred(X - X[gal].mean(0, keepdims=True))
    Sc = Xc[qry] @ Xc[gal].T
    nnc = np.argmax(Sc, axis=1)
    inst_c = (df["class_label"].to_numpy()[gal][nnc] == df["class_label"].to_numpy()[qry]).mean()
    fam_c = (fam_g[nnc] == fam_q).mean()
    # random baseline: the frequency of the query class inside the gallery
    inst_ch = float((pd.Series(df["class_label"].to_numpy()[gal]).value_counts(normalize=True) ** 2).sum())
    fam_ch = float((pd.Series(fam_g).value_counts(normalize=True) ** 2).sum())
    return {"transfer_gallery": int(gal.sum()), "transfer_query": int(qry.sum()),
            "transfer_instrument_R1": float(inst), "transfer_instrument_chance": inst_ch,
            "transfer_family_R1": float(famhit), "transfer_family_chance": fam_ch,
            "transfer_instrument_R1_centred": float(inst_c),
            "transfer_family_R1_centred": float(fam_c)}


def mridangam_extra(df, X):
    S = X @ X.T
    t = df["tonic"].to_numpy()
    s = df["class_label"].to_numpy()
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    rng = np.random.default_rng(SEED)
    sil_t, p_t, _, n95_t = sil_perm_test(D, t.copy(), n_perm=500, rng=rng)
    sil_s, p_s, _, n95_s = sil_perm_test(D, s.copy(), n_perm=500, rng=rng)
    w_t, b_t, d_t = dcos_equal_weight(S, t)
    w_s, b_s, d_s = dcos_equal_weight(S, s)
    acc_t, exp_t, _ = knn_loo(S, t)
    acc_s, exp_s, _ = knn_loo(S, s)
    Xc = centred(X)
    Sc = Xc @ Xc.T
    Dc = 1.0 - Sc
    np.fill_diagonal(Dc, 0.0)
    return {"tonic_silhouette": sil_t, "tonic_sil_p_ge": p_t, "tonic_sil_null_p95": n95_t,
            "tonic_silhouette_centred": sil_from_dist(Dc, t),
            "stroke_silhouette": sil_s, "stroke_sil_p_ge": p_s, "stroke_sil_null_p95": n95_s,
            "stroke_silhouette_centred": sil_from_dist(Dc, s),
            "tonic_dcos": d_t, "stroke_dcos": d_s,
            "knn_tonic_acc": acc_t, "knn_tonic_chance": exp_t,
            "knn_stroke_acc": acc_s, "knn_stroke_chance": exp_s}


def main():
    result = {"meta": {"n_perm": N_PERM, "seed": SEED, "mridangam_n_perm": 500,
                       "features": FEATS,
                       "note": "silhouette uses cosine distance and kNN LOO excludes same-group near-duplicates; "
                               "Delta-cos is label-equal-weighted."}}
    result["medley"] = analyse("medley", "class_label", group_col="group",
                               extra=medley_extra, transfer=medley_transfer)
    result["brid"] = analyse("brid", "class_label", group_col="player")
    result["beijing"] = analyse("beijing", "class_label", group_col="group")
    result["mridangam"] = analyse("mridangam", "class_label", group_col="group",
                                  extra=mridangam_extra)
    (RES / "supp_controls.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {RES / 'supp_controls.json'}")


if __name__ == "__main__":
    main()
