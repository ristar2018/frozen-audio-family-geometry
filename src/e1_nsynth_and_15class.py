#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analysis of the NSynth exploratory subset + FolkMusic 15-class fine-grained metrics
used for Table 2 (NSynth) and Table 3 (15-class) of the paper"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

PROJ = Path(__file__).resolve().parent.parent
EMB_DIR = PROJ / "results" / "e1" / "embeddings"
FOLK = PROJ / "results" / "e1" / "splits" / "folkmusic_dev_fragments.csv"
# NSynth metadata is loaded through load_nsynth_acoustic in the library; reproduce that here

sys.path.insert(0, str(PROJ / "code"))
from data_loader import load_nsynth_acoustic  # noqa

ENCODERS = ["beats", "mert", "ast", "audiomae"]
ENC_LABEL = {"beats": "BEATs", "mert": "MERT-v2", "ast": "AST", "audiomae": "AudioMAE"}


def loo_acc_family_level(X, y_fam):
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                weights="distance", algorithm="brute")
    knn.fit(X, y_fam)
    return (knn.predict(X) == y_fam).mean()


def loo_acc_class_level(X, y_cls):
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                weights="distance", algorithm="brute")
    knn.fit(X, y_cls)
    return (knn.predict(X) == y_cls).mean()


def main():
    SPLIT_SEED = 20250301
    PER_ENTITY_CAP = 40
    MIN_ENTITIES_PER_FAMILY = 5

    folk = pd.read_csv(FOLK).sort_values(["recording_id","fragment_id"]).reset_index(drop=True)
    nsynth = load_nsynth_acoustic(exclude_vocal=True,
                                   min_entities_per_family=MIN_ENTITIES_PER_FAMILY,
                                   splits=("train",))
    # reproduce the per-entity capped sampling of e1_embedding_analysis.py
    rng = np.random.default_rng(SPLIT_SEED)
    keep_idx = []
    for _, sub in nsynth.groupby("entity_id"):
        idx = sub.index.to_numpy()
        if len(idx) > PER_ENTITY_CAP:
            idx = rng.choice(idx, PER_ENTITY_CAP, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    nsynth = nsynth.loc[np.sort(keep_idx)].reset_index(drop=True)

    out = {}

    # ========== NSynth (7 families / 370 entities) ==========
    print("=" * 60)
    print(f"NSynth: n={len(nsynth)}, families={nsynth['instrument_family'].nunique()}, "
          f"entities={nsynth['entity_id'].nunique()}")
    nsynth_fam = nsynth["instrument_family"].to_numpy()

    nsynth_rows = []
    for enc in ENCODERS:
        fp = EMB_DIR / f"{enc}_nsynth.npz"
        X = np.load(fp)["embeddings"]
        assert X.shape == (len(nsynth), 768)

        # family-level silhouette
        sil_fam = silhouette_score(X, nsynth_fam, metric="cosine",
                                    sample_size=min(len(nsynth), 6000), random_state=0)
        # family-level leave-one-out accuracy
        loo_fam = loo_acc_family_level(X, nsynth_fam)
        # within vs between cos (sampled)
        rng = np.random.default_rng(42)
        fams = np.unique(nsynth_fam)
        in_list, out_list = [], []
        for fam in fams:
            idx = np.where(nsynth_fam == fam)[0]
            q = rng.choice(idx, size=min(200, len(idx)), replace=False)
            for i in q:
                same = idx[idx != i]
                diff = np.where(nsynth_fam != fam)[0]
                in_list.append((X[i] @ X[same].T).mean())
                out_list.append((X[i] @ X[diff].T).mean())
        within = np.mean(in_list); between = np.mean(out_list)
        nsynth_rows.append({
            "encoder": enc,
            "label": ENC_LABEL[enc],
            "sil_fam": sil_fam,
            "loo_fam": loo_fam,
            "within": within,
            "between": between,
            "delta": within - between,
        })
        print(f"  {enc:>8s}  sil_fam={sil_fam:.3f}  loo={loo_fam:.4f}  "
              f"within={within:.4f}  between={between:.4f}  Δ={within-between:.4f}")
    out["nsynth"] = nsynth_rows

    # ========== FolkMusic 15-class ==========
    print("=" * 60)
    print(f"FolkMusic 15-class: n={len(folk)}")
    folk_cls = folk["instrument_class"].to_numpy()
    folk_fam = folk["family_mapped"].to_numpy()

    cls_rows = []
    for enc in ENCODERS:
        fp = EMB_DIR / f"{enc}_folkmusic_dev.npz"
        X = np.load(fp)["embeddings"]
        assert X.shape == (len(folk), 768)

        sil_cls = silhouette_score(X, folk_cls, metric="cosine",
                                    sample_size=min(len(folk), 6000), random_state=0)
        loo_cls = loo_acc_class_level(X, folk_cls)
        cls_rows.append({"encoder": enc, "label": ENC_LABEL[enc],
                          "sil_15": sil_cls, "loo_15": loo_cls})
        print(f"  {enc:>8s}  sil_15class={sil_cls:.3f}  loo_15class={loo_cls:.4f}")
    out["folk15"] = cls_rows

    # ========== Per-class confusion (FolkMusic 15-class LOO) ==========
    print("=" * 60)
    print("Per-class LOO-acc (15-class, k=5, cosine):")
    per_class = []
    for enc in ENCODERS:
        fp = EMB_DIR / f"{enc}_folkmusic_dev.npz"
        X = np.load(fp)["embeddings"]
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine",
                                    weights="distance", algorithm="brute")
        knn.fit(X, folk_cls)
        pred = knn.predict(X)
        cm = confusion_matrix(folk_cls, pred, labels=sorted(np.unique(folk_cls)),
                               normalize="true")
        per_class.append({"encoder": enc, "cm": cm,
                           "classes": sorted(np.unique(folk_cls))})
        # diagonal of each class block
        diags = {c: cm[i, i] for i, c in enumerate(sorted(np.unique(folk_cls)))}
        print(f"  {enc:>8s}:")
        for c, v in sorted(diags.items(), key=lambda x: -x[1]):
            print(f"    {c:12s}  {v:.3f}")
    out["per_class"] = [{"encoder": r["encoder"],
                          "class_acc": {c: float(cm[i, i])
                                        for i, c in enumerate(r["classes"])}}
                         for r in per_class]

    # save JSON
    out_path = PROJ / "results" / "e1" / "nsynth_and_15class.json"
    # serialise ndarrays inside per_class -> list
    save = json.loads(json.dumps(out, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x.tolist() if isinstance(x, np.ndarray) else x))
    out_path.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
