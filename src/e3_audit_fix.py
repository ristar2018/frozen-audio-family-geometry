# -*- coding: utf-8 -*-
"""
FAPN full-text number audit and back-fill (for the factual corrections)

Background: the Method section states that silhouette / LOO-kNN / Delta-cos were recomputed over
all 180,180 family labellings, but e3_exact_permutation.py only output the exact p of the silhouette. So that
every column of Table h1a and Table heldout has an authoritative source, this script fills in:

  A. exact-enumeration permutation test v2: dev + test x {beats, mert, ast, audiomae, mfcc}
     three statistics, each enumerated over the 180,180 labellings of 2/7/6 on the 15 class prototypes:
       (1) sil        : cosine silhouette (identical protocol to e3_exact_permutation.py, used as a cross-check)
       (2) dcos       : within/between family cosine difference (fragment level, family-equal weight, = the
                        bootstrap_ci.dcos_vec protocol; reconstructed exactly from the 15x15 class-pair cosine-sum
       matrix G so that O(n^2) need not be recomputed each time)
  (3) knn_proto  : LOO-kNN family accuracy on the 15 prototypes (k=5, cosine, distance weights)
       B. paired bootstrap (test, fragment-level resampling, B=1000, same seed as e3_bootstrap_ci.py):
       - paired Delta-cos differences (BEATs minus the others)   <- cited in the discussion without a source
  C. geometry-intervention statistics: Spearman(aniso_top1, few1) and (aniso_top1, proto_acc) per encoder,
     plus the range of knn_acc and r1_ins over all BEATs interventions
  D. IRMAS class/family counts -> largest-family share (majority baseline)

Output results/e3/audit_fix.json + console. CPU only.
"""
import json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from scipy.stats import spearmanr  # noqa: E402
from e2_supplement_analyze import prepare_features, l2norm, proto_predict  # noqa: E402
from e3_exact_permutation import enumerate_labels, class_prototypes, sil_all_from_dist  # noqa: E402

OUT = PROJ / "results" / "e3"
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]
K = 5
B = 1000
SEED = 20260501


# ------------------------------------------------------------------ statistics
def dcos_vec(S, y):
    """within/between family cosine difference (family-equal weight). Identical protocol to e3_bootstrap_ci.dcos_vec."""
    ds = []
    for c in np.unique(y):
        m, o = np.where(y == c)[0], np.where(y != c)[0]
        if len(m) < 2 or len(o) == 0:
            continue
        blk = S[np.ix_(m, m)]
        ds.append(float(((blk.sum(1) - 1.0) / (len(m) - 1)).mean()
                        - S[np.ix_(m, o)].sum(1).mean() / len(o)))
    return float(np.mean(ds)) if ds else float("nan")


def knn_struct(C):
    """Neighbours and weights of the LOO kNN on the 15 prototypes (sklearn weights='distance' semantics)."""
    n = C.shape[0]
    D = 1.0 - C
    np.fill_diagonal(D, np.inf)
    NB = np.argsort(D, axis=1)[:, :K]
    W = np.zeros((n, K))
    for i in range(n):
        d = D[i, NB[i]]
        W[i] = (d == 0).astype(float) if (d == 0).any() else 1.0 / d
    return NB, W


def knn_all_from_C(lab_mat, NB, W):
    """LOO-kNN family accuracy under an enumerated labelling (vectorised). lab_mat:(M,15)"""
    M = lab_mat.shape[0]
    nb_lab = lab_mat[:, NB]                       # (M,15,K)
    sc = np.empty((M, 15, 3), dtype=np.float32)
    for g in range(3):
        eq = (nb_lab == g).astype(np.float32)
        sc[:, :, g] = np.einsum("mik,ik->mi", eq, W.astype(np.float32))
    pred = sc.argmax(2)
    return (pred == lab_mat).mean(1)


def dcos_all_from_G(lab_mat, G, n_cls, N):
    """Fragment-level Delta-cos under an enumerated labelling (vectorised, G = 15x15 class-pair cosine sums)."""
    M = lab_mat.shape[0]
    acc = np.zeros(M)
    for g in range(3):
        m = (lab_mat == g).astype(np.float64)     # (M,15)
        mG = m @ G
        T = (mG * m).sum(1)                       # within-family total (diagonal included)
        U = (mG * (1.0 - m)).sum(1)               # between-family total
        Ng = m @ n_cls
        within = (T - Ng) / np.maximum((Ng - 1) * Ng, 1e-12)
        between = U / np.maximum(Ng * (N - Ng), 1e-12)
        acc += (within - between)
    return acc / 3.0


# ------------------------------------------------------------------ A
def exact_v2(feats, LAB, M):
    res = {}
    for split in ["dev", "test"]:
        split_res = {}
        for w in ORDER:
            if w not in feats or feats[w][split] is None:
                continue
            X, df = feats[w][split]
            yc = df["instrument_class"].to_numpy()
            yf = df["family"].to_numpy()
            P, classes, _ = class_prototypes(X, yc)
            c2fam = {c: str(np.unique(yf[yc == c])[0]) for c in classes}
            fam_names = sorted(set(c2fam.values()))
            real_lab = np.array([fam_names.index(c2fam[c]) for c in classes], dtype=np.int8)

            C = P @ P.T
            D = 1.0 - C
            np.fill_diagonal(D, 0.0)
            n_cls = np.array([int((yc == c).sum()) for c in classes], dtype=np.float64)
            N = float(n_cls.sum())
            Ssum = np.stack([X[yc == c].sum(0) for c in classes])   # (15,d)
            G = Ssum @ Ssum.T                                       # class-pair cosine sums

            # --- (1) silhouette
            sil_null = sil_all_from_dist(D, LAB)
            sil_real = sil_all_from_dist(D, real_lab[None, :])[0]
            p_sil = float((sil_null >= sil_real - 1e-12).sum()) / M

            # --- (2) dcos (self-check first: reconstructed G against the direct computation)
            dcos_real_direct = dcos_vec(X @ X.T, yf)
            dcos_real_G = dcos_all_from_G(real_lab[None, :], G, n_cls, N)[0]
            assert abs(dcos_real_direct - dcos_real_G) < 1e-8, (w, split, dcos_real_direct, dcos_real_G)
            dcos_null = dcos_all_from_G(LAB, G, n_cls, N)
            p_dcos = float((dcos_null >= dcos_real_G - 1e-12).sum()) / M

            # --- (3) knn_proto
            NB, Wt = knn_struct(C)
            knn_null = knn_all_from_C(LAB, NB, Wt)
            knn_real = float(knn_all_from_C(real_lab[None, :], NB, Wt)[0])
            p_knn = float((knn_null >= knn_real - 1e-12).sum()) / M

            split_res[w] = {
                "sil_real": round(float(sil_real), 6), "sil_p": round(p_sil, 8),
                "sil_n_ge": int((sil_null >= sil_real - 1e-12).sum()),
                "sil_null_q99": round(float(np.quantile(sil_null, .99)), 6),
                "dcos_real": round(float(dcos_real_G), 6), "dcos_p": round(p_dcos, 8),
                "dcos_n_ge": int((dcos_null >= dcos_real_G - 1e-12).sum()),
                "dcos_null_mean": round(float(dcos_null.mean()), 6),
                "dcos_null_q99": round(float(np.quantile(dcos_null, .99)), 6),
                "knn_proto_real": round(knn_real, 6), "knn_proto_p": round(p_knn, 8),
                "knn_proto_n_ge": int((knn_null >= knn_real - 1e-12).sum()),
                "knn_proto_null_mean": round(float(knn_null.mean()), 6),
                "knn_proto_null_q99": round(float(np.quantile(knn_null, .99)), 6),
                "knn_proto_correct_of_15": int(round(knn_real * 15)),
                "pass_a05": bool(max(p_sil, p_dcos, p_knn) < 0.05),
                "pass_a01": bool(max(p_sil, p_dcos, p_knn) < 0.01),
            }
            print(f"[exact] {split:4s} {w:9s} sil={sil_real:+.4f}(p={p_sil:.5f}) "
                  f"dcos={dcos_real_G:+.5f}(p={p_dcos:.5f}) "
                  f"knn={knn_real:.3f}(p={p_knn:.5f})  pass@.05={split_res[w]['pass_a05']}",
                  flush=True)
        res[split] = split_res
    return res


# ------------------------------------------------------------------ B
def paired_extra(feats):
    """test fragment-level paired bootstrap: Delta-cos differences + few-shot (20 draws) differences."""
    d = {}
    for w in ORDER:
        if w not in feats or feats[w]["test"] is None:
            continue
        Xd, dfd = feats[w]["dev"]
        Xt, dft = feats[w]["test"]
        Xd, Xt = l2norm(Xd), l2norm(Xt)
        yd_f, yt_f = dfd["family"].to_numpy(), dft["family"].to_numpy()
        S = Xt @ Xt.T
        n = len(Xt)
        # few-shot: replicate e2 part4 (20 draws, seed 20260501+rep), per-query correctness
        fs = {}
        for s in (1, 8):
            acc = np.zeros(n)
            for rep in range(20):
                rng_r = np.random.default_rng(20260501 + rep)
                sel = []
                for f in np.unique(yd_f):
                    idx = np.where(yd_f == f)[0]
                    sel += list(rng_r.choice(idx, size=min(s, len(idx)), replace=False))
                sel = np.array(sel)
                acc += (proto_predict(Xd[sel], yd_f[sel], Xt) == yt_f)
            fs[s] = acc / 20.0
        d[w] = {"S": S, "yf": yt_f, "n": n, "fs1": fs[1], "fs8": fs[8],
                "dcos_point": dcos_vec(S, yt_f),
                "fs1_point": float(fs[1].mean()), "fs8_point": float(fs[8].mean())}
    rng = np.random.default_rng(SEED)
    idxs = [rng.integers(0, d["beats"]["n"], d["beats"]["n"]) for _ in range(B)]
    out = {"B": B, "paired_vs_beats": {}}
    for w in ORDER:
        if w == "beats" or w not in d:
            continue
        dcos_d = [d["beats"]["dcos_point"] - d[w]["dcos_point"]] * 0  # placeholder
        dc = []
        f1, f8 = [], []
        for ix in idxs:
            dc.append(dcos_vec(d["beats"]["S"][np.ix_(ix, ix)], d["beats"]["yf"][ix])
                      - dcos_vec(d[w]["S"][np.ix_(ix, ix)], d[w]["yf"][ix]))
            f1.append(float(d["beats"]["fs1"][ix].mean() - d[w]["fs1"][ix].mean()))
            f8.append(float(d["beats"]["fs8"][ix].mean() - d[w]["fs8"][ix].mean()))
        q = lambda v: [round(float(np.quantile(v, .025)), 6), round(float(np.quantile(v, .975)), 6)]
        out["paired_vs_beats"][w] = {
            "dcos": {"diff": round(d["beats"]["dcos_point"] - d[w]["dcos_point"], 6),
                     "ci": q(dc)},
            "few1_20draw": {"diff": round(d["beats"]["fs1_point"] - d[w]["fs1_point"], 6),
                            "ci": q(f1)},
            "few8_20draw": {"diff": round(d["beats"]["fs8_point"] - d[w]["fs8_point"], 6),
                            "ci": q(f8)},
        }
        print(f"[paired] BEATs−{w}: dcos={out['paired_vs_beats'][w]['dcos']['diff']:+.4f}"
              f"{out['paired_vs_beats'][w]['dcos']['ci']} "
              f"few1={out['paired_vs_beats'][w]['few1_20draw']['diff']:+.4f}"
              f"{out['paired_vs_beats'][w]['few1_20draw']['ci']} "
              f"few8={out['paired_vs_beats'][w]['few8_20draw']['diff']:+.4f}"
              f"{out['paired_vs_beats'][w]['few8_20draw']['ci']}", flush=True)
    out["points"] = {w: {"dcos": round(d[w]["dcos_point"], 6),
                         "few1_20draw": round(d[w]["fs1_point"], 6),
                         "few8_20draw": round(d[w]["fs8_point"], 6)} for w in d}
    return out


# ------------------------------------------------------------------ C
def intervention_stats():
    j = json.loads((OUT / "geometry_intervention.json").read_text(encoding="utf-8"))
    out = {}
    for w, ops in j["encoders"].items():
        names = list(ops)
        an = np.array([ops[k]["aniso_top1"] for k in names])
        f1 = np.array([ops[k]["few1"] for k in names])
        pa = np.array([ops[k]["proto_acc"] for k in names])
        kn = np.array([ops[k]["knn_acc"] for k in names])
        r1 = np.array([ops[k]["r1_ins"] for k in names])
        sil = np.array([ops[k]["sil_proto"] for k in names])
        out[w] = {
            "spearman_aniso_few1": round(float(spearmanr(an, f1).statistic), 3),
            "spearman_aniso_proto": round(float(spearmanr(an, pa).statistic), 3),
            "spearman_aniso_sil": round(float(spearmanr(an, sil).statistic), 3),
            "knn_min": round(float(kn.min()), 4), "knn_max": round(float(kn.max()), 4),
            "r1_min": round(float(r1.min()), 4), "r1_max": round(float(r1.max()), 4),
            "sil_min": round(float(sil.min()), 4), "sil_max": round(float(sil.max()), 4),
            "n_ops": len(names),
        }
        print(f"[interv] {w:9s} rho(aniso,few1)={out[w]['spearman_aniso_few1']:+.3f} "
              f"rho(aniso,proto)={out[w]['spearman_aniso_proto']:+.3f} "
              f"knn∈[{out[w]['knn_min']:.4f},{out[w]['knn_max']:.4f}] "
              f"r1∈[{out[w]['r1_min']:.4f},{out[w]['r1_max']:.4f}] "
              f"sil∈[{out[w]['sil_min']:.3f},{out[w]['sil_max']:.3f}]", flush=True)
    return out


# ------------------------------------------------------------------ D
def irmas_counts():
    root = PROJ / "data" / "irmas" / "IRMAS-TrainingData"
    if not root.exists():
        print("[irmas] data not present, skipping")
        return None
    IRMAS_FAMILY = {"cel": "bowed", "vio": "bowed", "gac": "plucked", "gel": "plucked",
                    "cla": "woodwind", "flu": "woodwind", "sax": "woodwind",
                    "tru": "brass", "org": "keys", "pia": "keys", "voi": "voice"}
    cls, songs = [], []
    for w in sorted(root.rglob("*.wav")):
        c = None
        for tok in [w.parent.name] + w.stem.split("__"):
            t = tok.strip().lower()
            if t in IRMAS_FAMILY:
                c = t
                break
        if c:
            cls.append(c)
            songs.append(w.stem.rsplit("__", 1)[0])
    fam = [IRMAS_FAMILY[c] for c in cls]
    cf, sf = Counter(fam), Counter()
    seen = set()
    for s, f in zip(songs, fam):
        if s not in seen:
            seen.add(s)
            sf[f] += 1
    out = {"n_segment": len(cls), "n_song": len(seen),
           "class_counts": dict(Counter(cls)), "family_counts_segment": dict(cf),
           "family_counts_song": dict(sf),
           "majority_segment": round(max(cf.values()) / len(cls), 4),
           "majority_song": round(max(sf.values()) / len(seen), 4)}
    print(f"[irmas] n_seg={out['n_segment']} n_song={out['n_song']} "
          f"maj_seg={out['majority_segment']} maj_song={out['majority_song']} "
          f"fam_seg={out['family_counts_segment']}")
    return out


def main():
    t0 = time.time()
    LAB = enumerate_labels()
    M = len(LAB)
    assert M == 180180
    feats = prepare_features()
    res = {"M": M}
    print("=" * 100, "\nA. exact permutation v2 (sil / dcos / knn_proto)\n" + "=" * 100)
    res["exact_v2"] = exact_v2(feats, LAB, M)
    print("\n" + "=" * 100, "\nB. paired bootstrap (test, clip-level, B=1000)\n" + "=" * 100)
    res["paired"] = paired_extra(feats)
    print("\n" + "=" * 100, "\nC. geometry intervention stats\n" + "=" * 100)
    res["intervention"] = intervention_stats()
    print("\n" + "=" * 100, "\nD. IRMAS counts\n" + "=" * 100)
    res["irmas"] = irmas_counts()
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "audit_fix.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\nSaved -> results/e3/audit_fix.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
