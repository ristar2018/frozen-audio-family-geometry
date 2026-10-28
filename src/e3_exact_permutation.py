# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening 1/6: exact permutation test for H1a (full enumeration)

Answers points R3-1/R3-2 of the review:
  - under B=2000 Monte Carlo, AudioMAE gives p=0.009995 (=19/2001), one more exceedance would flip it, SE~0.0022;
  - the full set of 2/7/6 family labellings is 15!/(2!7!6!) = 180,180, so an exact p can be enumerated;
  - two statistics are reported:
      (a) proto  : cosine silhouette on the 15 class prototypes (= the Table 1/3 protocol of the paper)
      (b) seg    : fragment-level silhouette (after a family-surjective relabelling of the classes, over all fragments),
                   answering "the permutation holds the class counts rather than the fragment marginals / fragment-weighted sensitivity".

Statistic (a) is identical in protocol to analysis.class_level_permutation_test (15 L2 prototypes, cosine).
Null space: all 180,180 assignments of the fixed multiset [0]*2+[1]*7+[2]*6 to the 15 ordered class prototypes;
the true labelling is itself inside the enumeration, so the exact p = #{sil >= sil_real} / 180180 (observed included, i.e. the +1 correction).

CPU only; dev (Table 1) 4 SSL + MFCC, test (Table 3) 5 features.
Output results/e3/exact_permutation.json + console.
"""
import json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score as sk_sil
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import prepare_features, l2norm, ENCODERS  # noqa

OUT = PROJ / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = np.array([2, 7, 6])            # number of instrument classes in bowed/plucked/wind
NCLS = 15
FAM = np.array([0] * 2 + [1] * 7 + [2] * 6)  # true labelling in class order (see the proto_classes sort)


# ---------- enumerate all 180,180 labellings of 2/7/6 ----------
def enumerate_labels(n=15, sizes=(2, 7, 6)):
    """Return (M,15) int8: every assignment of the multiset given by sizes to n ordered positions."""
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


# ---------- vectorised cosine silhouette of the 15 prototypes, all M labellings at once ----------
def sil_all_from_dist(D, labels_mat):
    """D:(15,15) cosine distances; labels_mat:(M,15). Returns (M,) silhouette."""
    M = labels_mat.shape[0]
    n = D.shape[0]
    out = np.empty(M, dtype=np.float64)
    eye = np.eye(n, dtype=bool)
    for m in range(M):
        lab = labels_mat[m]
        same = lab[None, :] == lab[:, None]      # (15,15) same family
        cnt = np.bincount(lab, minlength=3)[lab]  # size of the cluster each point belongs to
        # a_i: mean distance to the same family (self excluded)
        ds = D * same
        a = (ds.sum(1) - 0.0) / np.maximum(cnt - 1, 1)   # diagonal D=0, so subtracting self changes nothing
        # b_i: the smaller of the mean distances to each of the other two families (own family excluded)
        mean_to = np.empty((n, 3))
        for g in range(3):
            gm = lab == g
            ng = int(gm.sum())
            mean_to[:, g] = (D * gm[None, :]).sum(1) / max(ng, 1)
        b = np.where(np.stack([lab == g for g in range(3)], 1),
                     np.inf, mean_to).min(1)
        s = (b - a) / np.maximum(a, b)
        s = np.where(cnt <= 1, 0.0, s)
        out[m] = s.mean()
    return out


def class_prototypes(X, y_class):
    classes = np.array(sorted(np.unique(y_class)))
    P, order = [], []
    for c in classes:
        idx = np.where(y_class == c)[0]
        P.append(l2norm(X[idx].mean(0, keepdims=True))[0])
        order.append(c)
    return np.stack(P, 0), np.array(order), classes


def seg_sil(X, y_class, y_family, labels_for_classes, sample=None, rng=None):
    """Fragment-level family silhouette: given a 15-class -> family labelling, remap every fragment."""
    classes = np.array(sorted(np.unique(y_class)))
    c2g = {c: int(labels_for_classes[i]) for i, c in enumerate(classes)}
    lab = np.array([c2g[c] for c in y_class])
    if sample is not None and len(X) > sample:
        idx = rng.choice(len(X), sample, replace=False)
        Xs, labs = X[idx], lab[idx]
    else:
        Xs, labs = X, lab
    if len(np.unique(labs)) < 2:
        return 0.0
    return float(sk_sil(Xs, labs, metric="cosine"))


def main():
    t0 = time.time()
    print("[enum] generating all 2/7/6 class->family assignments ...", flush=True)
    LAB = enumerate_labels()
    M = len(LAB)
    assert M == 180180, M
    print(f"[enum] M={M} assignments  ({time.time()-t0:.1f}s)", flush=True)

    feats = prepare_features()
    rng = np.random.default_rng(20260501)
    order = ["beats", "mert", "ast", "audiomae", "mfcc"]
    results = {"n_assignments": M, "family_class_sizes": [2, 7, 6], "splits": {}}

    for split in ["dev", "test"]:
        print("\n" + "=" * 88)
        print(f"EXACT permutation test — {split}  (15 class prototypes, cosine silhouette)")
        print("=" * 88)
        print(f"{'feature':10s} {'sil_real':>9s} {'exact_p':>10s} {'#>=':>7s} "
              f"{'null_mean':>10s} {'null_q99':>9s} {'min':>8s} {'max':>8s}  seg_real")
        split_res = {}
        for w in order:
            if w not in feats or feats[w][split] is None:
                continue
            X, df = feats[w][split]
            yc = df["instrument_class"].to_numpy()
            yf = df["family"].to_numpy()
            P, classes, _ = class_prototypes(X, yc)
            assert len(classes) == NCLS, f"{w} {split}: {len(classes)} classes"
            # true labelling (after sorting the classes it should be exactly 2/7/6)
            c2fam = {c: str(np.unique(yf[yc == c])[0]) for c in classes}
            real_fam = np.array([c2fam[c] for c in classes])
            fam_names = sorted(np.unique(real_fam))
            real_lab = np.array([fam_names.index(v) for v in real_fam], dtype=np.int8)
            # if the true order is not [0,0,1..,2..] it is still located by its count multiset
            assert sorted(np.bincount(real_lab, minlength=3).tolist()) == sorted([2, 7, 6]), \
                np.bincount(real_lab, minlength=3)

            D = 1.0 - P @ P.T
            np.fill_diagonal(D, 0.0)
            sil_dist = sil_all_from_dist(D, LAB)

            # true statistic: own implementation against the sklearn cross-check
            sil_real_self = sil_all_from_dist(D, real_lab[None, :])[0]
            sil_real_sk = float(sk_sil(P, real_lab, metric="cosine"))
            assert abs(sil_real_self - sil_real_sk) < 1e-9, (sil_real_self, sil_real_sk)

            ge = int(np.sum(sil_dist >= sil_real_self - 1e-12))
            p_exact = ge / M
            null_mean = float(sil_dist.mean())
            null_q99 = float(np.quantile(sil_dist, 0.99))
            seg_real = seg_sil(X, yc, yf, real_lab, sample=6000, rng=rng)

            split_res[w] = {
                "sil_real_proto": round(float(sil_real_sk), 6),
                "exact_p": round(p_exact, 8), "n_ge": ge, "M": M,
                "null_mean": round(null_mean, 6), "null_q99": round(null_q99, 6),
                "null_min": round(float(sil_dist.min()), 6),
                "null_max": round(float(sil_dist.max()), 6),
                "pass_alpha_0.01": bool(p_exact < 0.01),
                "sil_segment_sampled6000": round(seg_real, 6),
            }
            print(f"{w:10s} {sil_real_sk:9.4f} {p_exact:10.5f} {ge:7d} "
                  f"{null_mean:10.4f} {null_q99:9.4f} {sil_dist.min():8.3f} "
                  f"{sil_dist.max():8.3f}  {seg_real:.3f}", flush=True)
        results["splits"][split] = split_res

    results["elapsed_sec"] = round(time.time() - t0, 1)
    outp = OUT / "exact_permutation.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {outp}  ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
