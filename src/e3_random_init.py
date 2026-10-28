# -*- coding: utf-8 -*-
"""
FAPN E3, review strengthening (E3-8, GPU): random-initialisation control (answers the core concern of R1)

The central claim "frozen self-supervised encoders recover instrument-family structure" must rule out two trivial explanations:
  (a) architectural or front-end priors -- a ViT/Transformer with a mel/fbank front end may already produce separable embeddings;
  (b) what self-supervised pretraining actually learned.
A random-initialisation control (same architecture, same front end, pretrained weights NOT loaded) tests (a) directly: if random
weights also recover family structure, the conclusion comes from the architectural prior rather than from self-supervised learning. >=3 seeds.

Two stages (extraction needs GPU, analysis is CPU only):
  python e3_random_init.py --stage extract --encoders ast,audiomae,mert,beats --seeds 0,1,2
  python e3_random_init.py --stage analyze

Output: results/e3/random_init/embeddings/<enc>_seed<k>_{dev,test}.npz
      results/e3/random_init.json
"""
import argparse, json, sys, time, gc
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "code"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402
import encoders as E  # noqa: E402
from encoders import _REGISTRY, _default_beats_source, _LOCAL_BEATS_CODE  # noqa: E402
from data_loader import load_folkmusic_manifest, split_folkmusic_dev_test, DEV_TEST_SEED  # noqa: E402
from e2_supplement_analyze import l2norm, proto_predict, knn_predict  # noqa: E402

E1_EMB = PROJ / "results" / "e1" / "embeddings"
E2_EMB = PROJ / "results" / "e2_supplement" / "embeddings"
OUT = PROJ / "results" / "e3" / "random_init"
EMB = OUT / "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PERM = 2000


# ============================================================
# randomly initialised models (same architecture, same front end, pretrained weights not loaded)
# ============================================================
def make_random(name, seed, device=DEVICE):
    enc = _REGISTRY[name](device=device)          # __init__ sets cfg/hf_id/sr/dim
    torch.manual_seed(seed); np.random.seed(seed)
    if name == "beats":
        code_dir = str(_LOCAL_BEATS_CODE)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from BEATs import BEATs, BEATsConfig
        ckpt = torch.load(_default_beats_source(), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        enc.model = BEATs(cfg)                    # random initialisation, load_state_dict not called
        enc._BEATsCls = BEATs
        enc.dim = cfg.encoder_embed_dim
    elif name == "audiomae":
        import timm
        enc.model = timm.create_model(f"hf_hub:{enc.hf_id}", pretrained=False)
    elif name == "ast":
        from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
        conf = AutoConfig.from_pretrained(enc.hf_id)
        enc.processor = AutoFeatureExtractor.from_pretrained(enc.hf_id)
        enc.model = AutoModel.from_config(conf)
    elif name == "mert":
        from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
        conf = AutoConfig.from_pretrained(enc.hf_id, trust_remote_code=True)
        enc.processor = AutoFeatureExtractor.from_pretrained(enc.hf_id, trust_remote_code=True)
        enc.model = AutoModel.from_config(conf, trust_remote_code=True)
        hid = enc.cfg.get("hidden_dim", 1024)
        if hid != enc.dim:
            gen = torch.Generator().manual_seed(20260501 + 330)
            Q, _ = torch.linalg.qr(torch.randn(hid, enc.dim, generator=gen))
            enc._proj = Q.to(device)
        else:
            enc._proj = None
    enc.model.eval().to(device)
    return enc


def extract(enc_names, seeds):
    fm = load_folkmusic_manifest(verify_files=False)
    fm_dev, fm_test = split_folkmusic_dev_test(fm, test_size=0.2, seed=DEV_TEST_SEED)
    fm_dev, fm_test = fm_dev.reset_index(drop=True), fm_test.reset_index(drop=True)
    EMB.mkdir(parents=True, exist_ok=True)
    for name in enc_names:
        for seed in seeds:
            enc = make_random(name, seed)
            for tag, df in [("dev", fm_dev), ("test", fm_test)]:
                dst = EMB / f"{name}_seed{seed}_{tag}.npz"
                if dst.exists():
                    print(f"[skip] {dst.name}"); continue
                t0 = time.time()
                r = enc.encode_dataset(df["file_path"].tolist(), log_every=200)
                ok = np.array(r["ok_idx"])
                sub = df.iloc[ok].reset_index(drop=True)
                np.savez_compressed(
                    dst, embeddings=r["embeddings"],
                    fragment_id=sub["fragment_id"].to_numpy(),
                    instrument_class=sub["instrument_class"].to_numpy(),
                    family=sub["family_mapped"].to_numpy())
                print(f"[extract] {dst.name} {r['embeddings'].shape} "
                      f"failed={len(r['failed'])} {time.time()-t0:.0f}s", flush=True)
            del enc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================
# analysis (CPU only): random vs pretrained triad
# ============================================================
def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return (z["embeddings"], z["instrument_class"], z["family"])


def proto_sil(X, y_c, y_f):
    from sklearn.metrics import silhouette_score
    classes = np.array(sorted(np.unique(y_c)))
    P, pf = [], []
    for c in classes:
        idx = np.where(y_c == c)[0]
        P.append(l2norm(X[idx].mean(0, keepdims=True))[0])
        pf.append(np.unique(y_f[idx])[0])
    return float(silhouette_score(np.stack(P), np.array(pf), metric="cosine"))


def dcos_vec(S, y):
    ds = []
    for c in np.unique(y):
        m, o = np.where(y == c)[0], np.where(y != c)[0]
        if len(m) < 2 or len(o) == 0:
            continue
        blk = S[np.ix_(m, m)]
        ds.append(float(((blk.sum(1) - 1.0) / (len(m) - 1)).mean()
                        - S[np.ix_(m, o)].sum(1).mean() / len(o)))
    return float(np.mean(ds)) if ds else float("nan")


def triad(Xd, yd_c, yd_f, Xt, yt_c, yt_f, rng, n_rep=10):
    Xd, Xt = l2norm(Xd), l2norm(Xt)
    out = {"sil_proto": proto_sil(Xt, yt_c, yt_f),
           "dcos": dcos_vec(Xt @ Xt.T, yt_f),
           "proto_acc": float((proto_predict(Xd, yd_f, Xt) == yt_f).mean()),
           "knn_acc": float((knn_predict(Xd, yd_f, Xt) == yt_f).mean())}
    for ns in (1, 8):
        accs = []
        for _ in range(n_rep):
            sel = []
            for f in np.unique(yd_f):
                idx = np.where(yd_f == f)[0]
                sel += list(rng.choice(idx, size=min(ns, len(idx)), replace=False))
            sel = np.array(sel)
            accs.append(float((proto_predict(Xd[sel], yd_f[sel], Xt) == yt_f).mean()))
        out[f"few{ns}"] = float(np.mean(accs))
    out["r1_ins"] = float((yd_c[np.argmax(Xt @ Xd.T, 1)] == yt_c).mean())
    return out


def analyze(enc_names, seeds):
    res = {"encoders": {}}
    for name in enc_names:
        # pretrained: dev=E1, test=E2
        Xd_p, yd_c, yd_f = load_npz(E1_EMB / f"{name}_folkmusic_dev.npz")
        Xt_p, yt_c, yt_f = load_npz(E2_EMB / f"{name}_folkmusic_test.npz")
        rec = {"pretrained": triad(Xd_p, yd_c, yd_f, Xt_p, yt_c, yt_f,
                                  np.random.default_rng(20260501))}
        rand_runs = []
        for seed in seeds:
            pd_ = EMB / f"{name}_seed{seed}_dev.npz"
            pt_ = EMB / f"{name}_seed{seed}_test.npz"
            if not (pd_.exists() and pt_.exists()):
                continue
            Xd_r, yd2_c, yd2_f = load_npz(pd_)
            Xt_r, yt2_c, yt2_f = load_npz(pt_)
            rand_runs.append(triad(Xd_r, yd2_c, yd2_f, Xt_r, yt2_c, yt2_f,
                                   np.random.default_rng(20260501)))
        if rand_runs:
            keys = list(rand_runs[0])
            rec["random_mean"] = {k: float(np.mean([r[k] for r in rand_runs])) for k in keys}
            rec["random_std"] = {k: float(np.std([r[k] for r in rand_runs])) for k in keys}
            rec["n_seeds"] = len(rand_runs)
        res["encoders"][name] = rec
        print(f"\n=== {name} ===")
        for k in rec["pretrained"]:
            pre = rec["pretrained"][k]
            rm = rec.get("random_mean", {}).get(k, float("nan"))
            print(f"  {k:10s} pretrained={pre:+.4f}  random={rm:+.4f}")
    (PROJ / "results" / "e3" / "random_init.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved -> results/e3/random_init.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "analyze"], required=True)
    ap.add_argument("--encoders", default="ast,audiomae,mert,beats")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    encs = a.encoders.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.stage == "extract":
        extract(encs, seeds)
    else:
        analyze(encs, seeds)


if __name__ == "__main__":
    main()
