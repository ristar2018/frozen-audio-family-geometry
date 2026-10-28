# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening (E3-7): sensitivity to the family partition + acoustic baseline

Answers R1-5/R2: does the conclusion depend on this particular 3-way bowed/plucked/wind partition? If so, the
"recovered family structure" of H1a would be an artefact. Approach: fix the 15 class prototypes, swap the family merging rule, recompute H1a.

Partitions:
  HS3   the existing Hornbostel-Sachs 3-way (2/7/6)
  BIN2  chordophones (bowed+plucked) vs aerophones (wind) (9/6)
  FINE5 bowed / plucked / struck (yangqin) / edge / reed (2/6/1/2/4)
  RAND3 random 3 groups (keeping the 2/7/6 class counts), 10 seeds -- negative control

Criterion: is the family silhouette of the real partition significantly above the RAND3 distribution, and does the H1a permutation p still hold?
The MFCC (240d) acoustic baseline is reported alongside, answering "is this just low-level acoustics".

CPU only. Output results/e3/family_partitions.json + console.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
from e2_supplement_analyze import prepare_features, l2norm, ENCODERS  # noqa

OUT = PROJ / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["beats", "mert", "ast", "audiomae", "mfcc"]
N_PERM = 2000
RAND_SEEDS = list(range(10))

HS3 = {
    "bowed": ["erhu", "zhuiqin"],
    "plucked": ["guqin", "guzheng", "liuqin", "pipa", "sanxian", "yangqin", "zhongruan"],
    "wind": ["ba", "dizi", "dongxiao", "hulusi", "sheng", "suona"],
}
BIN2 = {
    "string": HS3["bowed"] + HS3["plucked"],
    "wind": HS3["wind"],
}
FINE5 = {
    "bowed": ["erhu", "zhuiqin"],
    "plucked": ["guqin", "guzheng", "liuqin", "pipa", "sanxian", "zhongruan"],
    "struck": ["yangqin"],
    "edge": ["dizi", "dongxiao"],
    "reed": ["sheng", "suona", "ba", "hulusi"],
}


def class_protos(X, y_class):
    classes = np.array(sorted(np.unique(y_class)))
    P = np.stack([l2norm(X[y_class == c].mean(0, keepdims=True))[0] for c in classes])
    return classes, P


def fam_labels(classes, mapping):
    lab = np.empty(len(classes), dtype=object)
    for f, cs in mapping.items():
        for c in cs:
            lab[classes == c] = f
    return lab


def sil_p(P, lab, n_perm=N_PERM, seed=20260501):
    real = float(silhouette_score(P, lab, metric="cosine"))
    rng = np.random.default_rng(seed)
    null = np.array([float(silhouette_score(P, rng.permutation(lab), metric="cosine"))
                     for _ in range(n_perm)])
    p = float((1 + (null >= real).sum()) / (n_perm + 1))
    return real, p, float(np.quantile(null, .99)), float(null.mean())


def rand3_labels(classes, sizes=(2, 7, 6), seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(classes))
    lab = np.empty(len(classes), dtype=object)
    i = 0
    for gi, s in enumerate(sizes):
        lab[perm[i:i + s]] = f"g{gi}"
        i += s
    return lab


def main():
    t0 = time.time()
    feats = prepare_features()
    results = {"n_perm": N_PERM, "rand_seeds": RAND_SEEDS, "encoders": {}}

    for w in ORDER:
        if w not in feats or feats[w]["test"] is None:
            continue
        Xt, dft = feats[w]["test"]
        Xt = l2norm(Xt)
        classes, P = class_protos(Xt, dft["instrument_class"].to_numpy())
        rec = {}
        for name, mp in [("HS3", HS3), ("BIN2", BIN2), ("FINE5", FINE5)]:
            lab = fam_labels(classes, mp)
            real, p, q99, nm = sil_p(P, lab)
            rec[name] = {"sil": real, "p": p, "null_q99": q99, "null_mean": nm,
                         "n_family": len(mp)}
        # RAND3 negative control
        rands = []
        for s in RAND_SEEDS:
            lab = rand3_labels(classes, seed=s)
            real, p, _, _ = sil_p(P, lab, n_perm=500, seed=20260501)  # 500 permutations suffice for a control
            rands.append(real)
        rands = np.array(rands)
        rec["RAND3"] = {"sil_mean": float(rands.mean()), "sil_std": float(rands.std()),
                        "sil_max": float(rands.max()), "n_seed": len(rands)}
        results["encoders"][w] = rec

        print("\n" + "=" * 88)
        print(f"FAMILY PARTITION SENSITIVITY — {w}")
        print("=" * 88)
        for name in ("HS3", "BIN2", "FINE5"):
            r = rec[name]
            print(f"  {name:6s} sil={r['sil']:.4f}  p={r['p']:.4f}  "
                  f"null_q99={r['null_q99']:.4f}  ({r['n_family']} families)")
        r = rec["RAND3"]
        print(f"  RAND3  sil={r['sil_mean']:.4f}±{r['sil_std']:.4f} (max={r['sil_max']:.4f})"
              f"  -> HS3 {'>' if rec['HS3']['sil'] > r['sil_max'] else '<= max'} rand-max",
              flush=True)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    outp = OUT / "family_partitions.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {outp} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
