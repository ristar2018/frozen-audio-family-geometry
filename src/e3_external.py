# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening (E3-9): formal validation on external datasets

Answers R1/R2: "does the structure hold only on FolkMusic?"
  stage nsynth (CPU, reuses the existing embeddings): promote NSynth from "exploratory" to a formal external validation --
    entity-prototype family silhouette + permutation p, a genuine family-level LOO-kNN (corrects A7), Delta-cos (sampled),
    bootstrap 95% CI.
  stage irmas (GPU): the open-set IRMAS corpus (download first).

Usage:
  python e3_external.py --stage nsynth
  python e3_external.py --stage irmas-download
  python e3_external.py --stage irmas
Output: results/e3/external_nsynth.json / external_irmas.json
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from data_loader import load_nsynth_acoustic  # noqa: E402
from e2_supplement_analyze import l2norm  # noqa: E402

E1_EMB = PROJ / "results" / "e1" / "embeddings"
OUT = PROJ / "results" / "e3"
OUT.mkdir(parents=True, exist_ok=True)
ENCODERS = ["beats", "mert", "ast", "audiomae"]
KNN_K = 5
B = 500
N_PERM = 2000
SPLIT_SEED = 20250301
PER_ENTITY_CAP = 40
MIN_ENTITIES_PER_FAMILY = 5
# IRMAS 11 instrument classes -> timbre families (from Bosch et al. 2012 / the Zenodo 1290750 description)
IRMAS_FAMILY = {
    "cel": "bowed_strings", "vio": "bowed_strings",
    "gac": "plucked_strings", "gel": "plucked_strings",
    "cla": "woodwind", "flu": "woodwind", "sax": "woodwind",
    "tru": "brass", "org": "keys", "pia": "keys", "voi": "voice",
}
IRMAS_CODES = set(IRMAS_FAMILY)


def prototypes(X, y_c):
    classes = np.array(sorted(np.unique(y_c)))
    P = np.stack([l2norm(X[y_c == c].mean(0, keepdims=True))[0] for c in classes])
    return classes, P


def proto_sil(X, y_c, y_f):
    classes, P = prototypes(X, y_c)
    pf = np.array([np.unique(y_f[y_c == c])[0] for c in classes])
    return float(silhouette_score(P, pf, metric="cosine"))


def perm_p_labels(P, y, n_perm=N_PERM, seed=20260501):
    """family silhouette on the prototype matrix P + permutation p (prototypes fixed, labels shuffled)."""
    real = float(silhouette_score(P, y, metric="cosine"))
    rng = np.random.default_rng(seed)
    null = np.array([float(silhouette_score(P, rng.permutation(y), metric="cosine"))
                     for _ in range(n_perm)])
    return real, float((1 + (null >= real).sum()) / (n_perm + 1)), float(np.quantile(null, .99))


def proto_perm_p(X, y_c, y_f, n_perm=N_PERM, seed=20260501):
    """Group by y_c to obtain the prototypes -> permutation test on the family labels."""
    classes, P = prototypes(X, y_c)
    pf = np.array([np.unique(y_f[y_c == c])[0] for c in classes])
    return perm_p_labels(P, pf, n_perm=n_perm, seed=seed)


def loo_knn_acc(X, y, k=KNN_K):
    """True LOO: neighbours exclude self (corrects A7, where fit(X).predict(X) included self)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(X)
    _, ind = nn.kneighbors(X)
    nb = ind[:, 1:]
    correct = 0
    for i in range(len(X)):
        vals, cnt = np.unique(y[nb[i]], return_counts=True)
        correct += int(vals[cnt.argmax()] == y[i])
    return correct / len(X)


def dcos_sample(X, y, n_q=200, rng=None):
    rng = rng or np.random.default_rng(0)
    ins, outs = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        q = rng.choice(idx, size=min(n_q, len(idx)), replace=False)
        for i in q:
            same = idx[idx != i]
            diff = np.where(y != c)[0]
            ins.append(float(X[i] @ X[same].mean(0)))
            outs.append(float(X[i] @ X[diff].mean(0)))
    return float(np.mean(ins) - np.mean(outs))


def boot_ci(fn, n, rng, b=B):
    draws = []
    for _ in range(b):
        ix = rng.integers(0, n, n)
        try:
            draws.append(fn(ix))
        except Exception:
            draws.append(np.nan)
    v = np.array([x for x in draws if np.isfinite(x)])
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (np.nan, np.nan)


def stage_nsynth():
    t0 = time.time()
    nsynth = load_nsynth_acoustic(exclude_vocal=True,
                                  min_entities_per_family=MIN_ENTITIES_PER_FAMILY,
                                  splits=("train",))
    rng = np.random.default_rng(SPLIT_SEED)
    keep = []
    for _, sub in nsynth.groupby("entity_id"):
        idx = sub.index.to_numpy()
        if len(idx) > PER_ENTITY_CAP:
            idx = rng.choice(idx, PER_ENTITY_CAP, replace=False)
        keep.append(idx)
    nsynth = nsynth.loc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    y_f = nsynth["instrument_family"].to_numpy()
    y_e = nsynth["entity_id"].to_numpy()
    print(f"[nsynth] n={len(nsynth)} families={len(np.unique(y_f))} "
          f"entities={len(np.unique(y_e))}")

    res = {"n": len(nsynth), "n_family": int(len(np.unique(y_f))),
           "n_entity": int(len(np.unique(y_e))), "B": B, "n_perm": N_PERM, "encoders": {}}
    for enc in ENCODERS:
        p = E1_EMB / f"{enc}_nsynth.npz"
        if not p.exists():
            continue
        X = l2norm(np.load(p)["embeddings"])
        assert len(X) == len(nsynth), f"{enc} rows {len(X)} != {len(nsynth)}"
        sil_pt, sil_p, sil_q99 = proto_perm_p(X, y_e, y_f)
        r = np.random.default_rng(20260501)
        ci_sil = boot_ci(lambda ix: proto_sil(X[ix], y_e[ix], y_f[ix]), len(X), r)
        loo = loo_knn_acc(X, y_f)
        dcos = dcos_sample(X, y_f, n_q=50, rng=np.random.default_rng(20260501))
        res["encoders"][enc] = {
            "sil_proto": sil_pt, "sil_p": sil_p, "sil_null_q99": sil_q99,
            "sil_proto_ci": ci_sil, "loo_knn_family": loo, "dcos": dcos}
        print(f"  {enc:9s} sil_proto={sil_pt:.4f}(p={sil_p:.5f}) "
              f"[{ci_sil[0]:.3f},{ci_sil[1]:.3f}] LOO-kNN={loo:.4f} dcos={dcos:.4f}",
              flush=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "external_nsynth.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved -> results/e3/external_nsynth.json ({res['elapsed_sec']}s)")


def stage_irmas_download():
    dst = PROJ / "data" / "irmas"
    dst.mkdir(parents=True, exist_ok=True)
    zpath = dst / "IRMAS-TrainingData.zip"
    url = "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"
    extract = dst / "IRMAS-TrainingData"
    if extract.exists() and any(extract.rglob("*.wav")):
        print(f"[irmas] already extracted: {extract} (no archive needed, skipping the download)")
        return
    if not zpath.exists():
        print(f"[irmas] downloading {url} -> {zpath}")
        r = subprocess.run(["aria2c", "-x", "8", "-s", "8", "-d", str(dst), "-o",
                            "IRMAS-TrainingData.zip", url], check=False)
        if r.returncode != 0 or not zpath.exists():
            print("aria2c failed; download manually and place it at data/irmas/IRMAS-TrainingData.zip")
            return
    print(f"[irmas] extracting {zpath.name} -> {dst} (about 3 GB, several minutes)")
    import zipfile
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst)
    print(f"[irmas] extraction done, wav={len(list(extract.rglob('*.wav')))}")


def _irmas_class(w: Path):
    """Robust IRMAS class parsing: prefer the parent directory name, fall back to the __ field of the file name."""
    for tok in [w.parent.name] + w.stem.split("__"):
        t = tok.strip().lower()
        if t in IRMAS_CODES:
            return t
    return None


def stage_irmas():
    root = PROJ / "data" / "irmas" / "IRMAS-TrainingData"
    if not root.exists():
        print(f"[irmas] missing {root}; run --stage irmas-download first")
        return
    import torch
    from collections import Counter
    from encoders import create_encoder
    all_wav = sorted(root.rglob("*.wav"))
    wavs, cls = [], []
    for w in all_wav:
        c = _irmas_class(w)
        if c is not None:
            wavs.append(w); cls.append(c)
    print(f"[irmas] total wav={len(all_wav)} classes recognised={len(wavs)} "
          f"unrecognised={len(all_wav)-len(wavs)}")
    print("  class counts:", dict(Counter(cls)))
    if not wavs:
        print("  [!] no class recognised: check the extracted directory/file structure and rerun")
        return
    cls = np.array(cls)
    fam = np.array([IRMAS_FAMILY[c] for c in cls])
    # NOTE: one song is cut into 3 segments (...__1/__2/__3) and segments of the same song are highly correlated,
    groups = np.array([w.stem.rsplit("__", 1)[0] for w in wavs])
    print(f"  families={sorted(set(fam.tolist()))}  songs={len(set(groups.tolist()))}")
    emb_dir = PROJ / "results" / "e3" / "irmas_emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    res = {"n": len(wavs), "n_class": int(len(np.unique(cls))),
           "n_family": int(len(np.unique(fam))), "encoders": {}}
    for enc in ENCODERS:
        dst = emb_dir / f"{enc}_irmas.npz"
        if not dst.exists():
            e = create_encoder(enc)
            r = e.encode_dataset([str(w) for w in wavs], log_every=500)
            np.savez_compressed(dst, embeddings=r["embeddings"], ok_idx=np.array(r["ok_idx"]))
            del e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        z = np.load(dst, allow_pickle=True)
        ok = np.asarray(z["ok_idx"])
        X = l2norm(z["embeddings"])
        cls_ok, fam_ok, grp_ok = cls[ok], fam[ok], groups[ok]
        # song-level prototypes (independent unit): merge the segments of a song first, then run the family test and LOO
        gs = np.unique(grp_ok)
        Xg = np.stack([l2norm(X[grp_ok == g].mean(0, keepdims=True))[0] for g in gs])
        yg = np.array([np.unique(fam_ok[grp_ok == g])[0] for g in gs])
        sil_g, p_g, q99_g = perm_p_labels(Xg, yg, n_perm=2000)
        sil_c, p_c, _ = proto_perm_p(X, cls_ok, fam_ok, n_perm=2000)
        res["encoders"][enc] = {
            "n_group": int(len(gs)),
            "sil_family_song": sil_g, "sil_family_song_p": p_g,
            "sil_song_null_q99": q99_g, "loo_knn_family_song": loo_knn_acc(Xg, yg),
            "sil_proto_class": sil_c, "sil_proto_class_p": p_c,
            "loo_knn_family_sample": loo_knn_acc(X, fam_ok),
            "dcos_sample": dcos_sample(X, fam_ok)}
        print(f"  {enc:9s} song-level sil={sil_g:.4f}(p={p_g:.5f}) "
              f"LOO={res['encoders'][enc]['loo_knn_family_song']:.4f} | segment-level "
              f"sil={sil_c:.4f}(p={p_c:.5f}) "
              f"LOO={res['encoders'][enc]['loo_knn_family_sample']:.4f}", flush=True)
    (OUT / "external_irmas.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved -> results/e3/external_irmas.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["nsynth", "irmas", "irmas-download"])
    a = ap.parse_args()
    {"nsynth": stage_nsynth, "irmas": stage_irmas,
     "irmas-download": stage_irmas_download}[a.stage]()


if __name__ == "__main__":
    main()
