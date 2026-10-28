# -*- coding: utf-8 -*-
"""
FAPN E1 data loader

Datasets:
  - FolkMusic: 11,966 fragments, 15 classes, families 6/2/7 (wind / bowed / plucked)
  - NSynth acoustic: exploratory cross-domain set (western instrument families), vocal excluded,
                      family-level labels; an entity is (family, local instrument_name index)

Leakage control:
  - FolkMusic is split dev/test by recording_id (fragments of one recording never straddle the split),
    stratified by instrument_class (StratifiedGroupKFold).
  - NSynth uses the official split: only train enters the exploratory E1 analysis; test/valid stay out.

Paths (the project root is two levels above this file):
  Every data, result and model path derives from PROJECT_ROOT, obtained by walking two levels up from
  src/data_loader.py. Moving the project directory (new disk or machine, same layout) needs no code change,
  and the file_path column is rebuilt as project-root plus relative rule at load time,
  so stale absolute paths recorded inside manifest or metadata files are ignored.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Project paths (single source of truth)
# ============================================================
# src/data_loader.py -> two levels up = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
FOLKMUSIC_ROOT = DATA_ROOT / "folkmusic"
NSYNTH_ROOT = DATA_ROOT / "nsynth"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"

FOLKMUSIC_MANIFEST = FOLKMUSIC_ROOT / "manifest.csv"
NSYNTH_METADATA = NSYNTH_ROOT / "nsynth_acoustic_metadata.csv"

DEV_TEST_SEED = 20250301  # split seed (fixed; the main script --split_seed defaults to this value)

# Family assignment (wind 6 / bowed 2 / plucked 7)
FAMILY_MAP = {
    # wind, 6 classes
    "dizi": "wind", "dongxiao": "wind", "sheng": "wind",
    "suona": "wind", "hulusi": "wind", "ba": "wind",
    # bowed, 2 classes
    "erhu": "bowed", "zhuiqin": "bowed",
    # plucked, 7 classes
    "pipa": "plucked", "guzheng": "plucked", "guqin": "plucked",
    "yangqin": "plucked", "zhongruan": "plucked", "liuqin": "plucked",
    "sanxian": "plucked",
}
FAMILY_ORDER = ["wind", "bowed", "plucked"]


# ============================================================
# FolkMusic
# ============================================================
def load_folkmusic_manifest(verify_files=False):
    """
    Load the FolkMusic manifest.csv (under data/folkmusic/ in the project).
    file_path is always rebuilt as project-root/data/folkmusic/<instrument_class>/<basename>
    rather than trusting a possibly stale absolute path in the manifest.
    """
    if not FOLKMUSIC_MANIFEST.exists():
        raise FileNotFoundError(f"FolkMusic manifest not found: {FOLKMUSIC_MANIFEST}")
    df = pd.read_csv(FOLKMUSIC_MANIFEST)

    required = ["recording_id", "fragment_id", "instrument_class",
                "family", "duration_sec", "file_path"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"FolkMusic manifest missing column: {col}")

    df["family_mapped"] = df["instrument_class"].map(FAMILY_MAP)
    if df["family_mapped"].isna().any():
        miss = df.loc[df["family_mapped"].isna(), "instrument_class"].unique()
        raise ValueError(f"Instruments missing family mapping: {miss}")

    # cross-check: the family column shipped with the manifest must agree with FAMILY_MAP
    mism = df[df["family"] != df["family_mapped"]]
    if len(mism) > 0:
        raise ValueError(
            f"family column conflicts with FAMILY_MAP for {len(mism)} rows, "
            f"e.g. {mism[['instrument_class','family']].drop_duplicates().head().to_dict('records')}"
        )

    # rebuild the project-relative path (forward slashes): folkmusic/<instrument>/<basename>
    def _rebuild(row):
        basename = os.path.basename(str(row["file_path"]).replace("\\", "/"))
        return (FOLKMUSIC_ROOT / str(row["instrument_class"]) / basename).as_posix()

    df["file_path"] = df.apply(_rebuild, axis=1)

    if verify_files:
        missing = [p for p in df["file_path"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} FolkMusic files missing, e.g. {missing[:3]}")

    return df.reset_index(drop=True)


def split_folkmusic_dev_test(df, test_size=0.2, seed=DEV_TEST_SEED):
    """
    Group by recording_id and stratify by instrument_class into an 80/20 split.
    All fragments of one recording fall on the same side (guards against audio-level leakage).

    Returns:
        dev_df, test_df (indices reset)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.arange(len(df))
    y = df["instrument_class"].values
    groups = df["recording_id"].values

    n_splits = int(round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # the first fold becomes test (about 20%); the rest is dev
    splits = sgkf.split(X, y, groups)
    dev_idx, test_idx = next(splits)

    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # assert that no recording appears on both sides
    overlap = set(dev_df["recording_id"]) & set(test_df["recording_id"])
    if overlap:
        raise AssertionError(f"recording_id leakage across dev/test: {len(overlap)}")

    return dev_df, test_df


# ============================================================
# NSynth acoustic
# ============================================================
def load_nsynth_acoustic(exclude_vocal=True, min_entities_per_family=None,
                         splits=("train", "valid", "test"), verify_files=False):
    """
    Load the rebuilt nsynth_acoustic_metadata.csv.

    Args:
        exclude_vocal: drop the vocal family (decision: voice is not an instrument here).
        min_entities_per_family: when given, drop families with fewer instrument entities than the threshold
            (under a class-level permutation a family with a single entity, e.g. bass, carries no information; 5 is a sensible exploratory value).
        splits: which official splits to keep.
    """
    meta = NSYNTH_METADATA
    if not meta.exists():
        raise FileNotFoundError(f"NSynth metadata not found: {meta}")
    df = pd.read_csv(meta)

    if "family" in df.columns and "instrument_family" not in df.columns:
        df["instrument_family"] = df["family"]

    # global instrument-entity id: instrument_name is a within-family local index, so it needs the family prefix
    df["entity_id"] = (
        df["instrument_family"].astype(str) + "__"
        + df["instrument_name"].astype(int).astype(str).str.zfill(3)
    )

    # audio path: rebuilt as project-root/data/nsynth/<split>/audio/<note_id>.wav,
    # ignoring stale absolute paths recorded in the metadata.
    df["file_path"] = df.apply(
        lambda r: (NSYNTH_ROOT / str(r["split"]) / "audio"
                   / f"{r['note_id']}.wav").as_posix(),
        axis=1,
    )

    if splits is not None:
        df = df[df["split"].isin(splits)].reset_index(drop=True)

    if exclude_vocal:
        df = df[df["instrument_family"] != "vocal"].reset_index(drop=True)

    if min_entities_per_family is not None:
        ent_cnt = df.groupby("instrument_family")["entity_id"].nunique()
        keep_fams = set(ent_cnt[ent_cnt >= min_entities_per_family].index)
        df = df[df["instrument_family"].isin(keep_fams)].reset_index(drop=True)

    if verify_files:
        missing = [p for p in df["file_path"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} NSynth files missing, e.g. {missing[:3]}")

    return df


def nsynth_family_entity_counts(df):
    """Number of instrument entities per family (family_sizes for the class-level permutation), sorted by family name."""
    cnt = df.groupby("instrument_family")["entity_id"].nunique()
    families = sorted(cnt.index)
    return families, [int(cnt[f]) for f in families]


# ============================================================
# Summary and self-check
# ============================================================
def load_e1_dev(test_size=0.2, seed=DEV_TEST_SEED, nsynth_min_entities=5):
    """
    Data actually used by E1:
      - FolkMusic dev (80%, recording-grouped and stratified); test is returned separately and kept out of E1
      - NSynth official train (exploratory), vocal excluded and families with fewer than 5 entities (bass, organ) dropped
    """
    fm = load_folkmusic_manifest()
    fm_dev, fm_test = split_folkmusic_dev_test(fm, test_size=test_size, seed=seed)

    ns_dev = load_nsynth_acoustic(
        exclude_vocal=True,
        min_entities_per_family=nsynth_min_entities,
        splits=("train",),
    )
    return {
        "folkmusic_dev": fm_dev,
        "folkmusic_test": fm_test,
        "nsynth_dev": ns_dev,
    }


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== FolkMusic ===")
    fm = load_folkmusic_manifest()
    print("N =", len(fm), "| classes =", fm["instrument_class"].nunique())
    print("fragments/family:", fm["family_mapped"].value_counts().to_dict())
    print("recordings =", fm["recording_id"].nunique(),
          "| frag per rec: max =", fm.groupby("recording_id").size().max())
    dev, test = split_folkmusic_dev_test(fm)
    print(f"dev={len(dev)}  test={len(test)}  (ratio {len(test)/len(fm):.3f})")
    print("dev family counts:", dev["family_mapped"].value_counts().to_dict())
    print("dev classes:", sorted(dev["instrument_class"].unique()).__len__(),
          "test classes:", sorted(test["instrument_class"].unique()).__len__())

    print("\n=== NSynth acoustic (dev=train, excl vocal, entity>=5) ===")
    ns = load_nsynth_acoustic(exclude_vocal=True, min_entities_per_family=5,
                              splits=("train",))
    print("N =", len(ns))
    fams, sizes = nsynth_family_entity_counts(ns)
    print("families:", fams)
    print("entity counts per family:", sizes)
    print("note counts per family:",
          ns["instrument_family"].value_counts().sort_index().to_dict())
