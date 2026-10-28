# -*- coding: utf-8 -*-
"""
Route A: cross-cultural zero-shot probe (Saraga 1.5 Hindustani).

Pipeline:
  1) Load Saraga manifest -> assign raag to coarse family (10-thaat inspired).
  2) 4 encoders (ast / mert / audiomae / beats) -> 108 x 768 embeddings each,
     cached to results/saraga/embeddings/<encoder>.npy.
  3) UMAP + t-SNE per encoder, cosine metric.
  4) Metrics: family silhouette; same-raag-different-artist avg cosine.
  5) HTML report with Plotly 4-panel interactive scatter.
"""
import os, sys, re, json, html as html_mod, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def _vram(tag: str = ""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved()   / 1024**2
    print(f"[vram{': '+tag if tag else ''}] alloc={a:.0f} MB  reserved={r:.0f} MB")


def _release_encoder(enc):
    """Move model to CPU, drop references, empty CUDA cache."""
    try:
        if hasattr(enc, "model") and enc.model is not None:
            enc.model.to("cpu")
        enc.model = None
    except Exception:
        pass
    for attr in ("processor", "feature_extractor", "tokenizer"):
        if hasattr(enc, attr):
            try: setattr(enc, attr, None)
            except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    _vram("after release")

ROOT      = Path(__file__).resolve().parent.parent
SARAGA    = ROOT / "data" / "openethnic" / "saraga"
HINDU     = SARAGA / "hindustani"
MANIFEST  = ROOT / "manifests" / "saraga_hindustani_manifest.csv"
OUT_DIR   = ROOT / "results" / "saraga"
EMB_DIR   = OUT_DIR / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoders as E

# ===== 1) raag -> family (coarse, 10-thaat inspired) ===================
# Order matters: specific patterns first. "meta" = non-musical entries.
_FAMILY_PATTERNS = [
    ("meta",     re.compile(r"^(intro|lyrics|speech|concept|shrutinandan|taal)")),
    ("bhairav",  re.compile(r"bhairav|ahir bhair|nat bhairav|bhairabi|bhatiyar")),
    ("kalyan",   re.compile(r"kalyan|jait kalyan|shuddha kalyan|shuddh kalyan|yaman")),
    ("malhar",   re.compile(r"malhar|megh")),
    ("marwa",    re.compile(r"^marwa|puriya|^shree|bibhas|lalit|lalat")),
    ("todi",     re.compile(r"todi|multani|asavari|komal rishabh|bilaskhani")),
    ("kafi",     re.compile(r"kafi|khamaj|bahar|maajh khamaj|mishra piloo|des\b|jhinjhoti")),
    ("bhairavi", re.compile(r"bhairavi|kauns|malkauns|chandrakauns|madhukauns")),
    ("bilawal",  re.compile(r"bilawal|kedar|basanti|bageshree|hameer|kamod|sohini|suha|paraj|sawani|sohani")),
]

def raag_to_family(name: str) -> str:
    s = (name or "").strip().lower()
    for fam, pat in _FAMILY_PATTERNS:
        if pat.search(s):
            return fam
    return "standalone"

# ===== 2) manifest + family column =====================================
df = pd.read_csv(MANIFEST, encoding="utf-8-sig")
df["raag_family"] = df["raag_common"].map(raag_to_family)
print("=== raag family counts ===")
print(df["raag_family"].value_counts().to_string())
print(f"families: {df['raag_family'].nunique()} | unique raag_common: {df['raag_common'].nunique()} | rows: {len(df)}")

# ===== 3) feature extraction (per encoder, cached to npy) ==============
audio_paths = [str(HINDU / p) for p in df["audio_relpath"].tolist()]
# sanity
miss = [p for p in audio_paths if not Path(p).exists()]
if miss:
    print(f"!!! {len(miss)} audio files MISSING; first 3:")
    for m in miss[:3]: print("   ", m)
    raise SystemExit(1)
ENCODERS = ["ast", "audiomae", "mert", "beats"]

embs = {}
_vram("start")
for name in ENCODERS:
    cache = EMB_DIR / f"{name}.npy"
    if cache.exists():
        X = np.load(cache)
        print(f"[{name}] loaded cache {cache} shape={X.shape}")
    else:
        _vram(f"before load {name}")
        print(f"[{name}] loading encoder...")
        enc = E.create_encoder(name)
        _vram(f"after load {name}")
        out = enc.encode_dataset(audio_paths, batch_size=2, log_every=8)
        _vram(f"after encode {name}")
        N = len(audio_paths)
        X = np.zeros((N, out["embeddings"].shape[1]), dtype=np.float32)
        for i, gi in enumerate(out["ok_idx"]):
            X[gi] = out["embeddings"][i]
        np.save(cache, X)
        print(f"[{name}] saved {cache} shape={X.shape} failed={len(out['failed'])}")
        if out["failed"]:
            for gi, p, err in out["failed"][:5]:
                print(f"    fail idx={gi} {Path(p).name}: {err}")
        # 🔴 release GPU resources for this encoder before moving to next
        _release_encoder(enc)
        del enc, out
        gc.collect()
    embs[name] = X
    assert X.shape == (len(df), 768), f"{name} shape mismatch {X.shape}"

# ===== 4) UMAP / t-SNE =================================================
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed; pip install umap-learn (fallback to t-SNE only)")

family_labels = df["raag_family"].values
raag_labels   = df["raag_common"].values

projections = {}
for name, X in embs.items():
    proj = {}
    if HAS_UMAP:
        u = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=20260501)
        proj["umap"] = u.fit_transform(X)
    t = TSNE(n_components=2, metric="cosine", random_state=20260501,
             init="pca", learning_rate="auto")
    proj["tsne"] = t.fit_transform(X)
    projections[name] = proj

# ===== 5) metrics ======================================================
# silhouette by family (cosine)
fams_all = list(df["raag_family"].unique())
fams_multi = [f for f in fams_all if (family_labels == f).sum() >= 2]

metrics = {}
for name, X in embs.items():
    sil = silhouette_score(X, family_labels, metric="cosine") if len(fams_multi) >= 2 else float("nan")
    # same-raag-different-artist agreement: for raags with >1 recording
    sims = []
    for rg, grp in df.groupby("raag_common"):
        if len(grp) < 2: continue
        idx = grp.index.tolist()
        v = X[idx]
        cos = v @ v.T
        iu = np.triu_indices(len(idx), k=1)
        sims.append(cos[iu].mean())
    agreement = float(np.mean(sims)) if sims else float("nan")
    metrics[name] = {"silhouette_family": sil, "same_raag_cos": agreement}
print("\n=== metrics ===")
for k, v in metrics.items():
    print(f"  {k:9s}  sil_family={v['silhouette_family']:+.3f}  same_raag_cos={v['same_raag_cos']:.3f}")

# ===== 6) HTML report (Plotly) =========================================
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

fam_unique = sorted(set(family_labels))
cmap = plt.get_cmap("tab20")
color_of = {f: to_hex(cmap(i / max(1, len(fam_unique) - 1))) for i, f in enumerate(fam_unique)}

def build_card(name, method, Y, sil, agree):
    pts = []
    for y, rl, fl, p, a in zip(Y, raag_labels, family_labels,
                                df["performance"], df["lead_artist"]):
        pts.append({
            "x": float(y[0]), "y": float(y[1]),
            "raag": rl, "family": fl, "perf": p, "artist": a,
            "color": color_of[fl],
        })
    pts_json = json.dumps(pts, ensure_ascii=False)
    pid = f"plot-{name}-{method}"
    return f"""
    <div class='card'>
      <h3>{html_mod.escape(name)} · {method.upper()}
          <small>(sil_family={sil:+.3f}, same_raag_cos={agree:.3f})</small>
      </h3>
      <div id='{pid}' style='width:100%;height:640px;background:#fafafa'></div>
      <script>
      (function(){{
        const pts = {pts_json};
        const el = document.getElementById('{pid}');
        const data = [{{
          x: pts.map(p=>p.x), y: pts.map(p=>p.y),
          mode: 'markers', type: 'scatter',
          marker: {{size:8, color: pts.map(p=>p.color), line:{{width:0.5,color:'#333'}}}},
          text: pts.map(p=> p.raag + '<br>' + p.artist + '<br>' + p.perf),
          hoverinfo: 'text'
        }}];
        Plotly.newPlot(el, data, {{margin:{{t:20}}, hovermode:'closest'}},
                       {{responsive:true, displaylogo:false}});
      }})();
      </script>
    </div>"""

cards = []
for name in ENCODERS:
    proj = projections[name]
    sil, agree = metrics[name]["silhouette_family"], metrics[name]["same_raag_cos"]
    for method in ("umap", "tsne"):
        if method not in proj: continue
        cards.append(build_card(name, method, proj[method], sil, agree))

legend = "\n".join(
    f'<span style="display:inline-block;width:12px;height:12px;background:{color_of[f]};margin-right:6px;vertical-align:middle"></span>{html_mod.escape(f)}'
    for f in fam_unique
)

HTML_PATH = OUT_DIR / "cross_cultural_probe.html"
HTML_PATH.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Saraga Route-A Probe</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;margin:24px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px}}
h3{{margin:.2em 0}}small{{color:#666;font-weight:normal;margin-left:12px}}
.legend{{display:flex;flex-wrap:wrap;gap:12px 18px;padding:12px;border:1px solid #eee;border-radius:8px;margin-bottom:16px}}
</style></head><body>
<h2>Saraga 1.5 Hindustani · Cross-cultural Zero-shot Probe</h2>
<p>108 Raag recordings × 4 encoders (768-d, L2). Color = raag family (coarse 10-thaat grouping).
   <b>sil_family</b> = silhouette by family in 768-d cosine space;
   <b>same_raag_cos</b> = avg within-raag cosine when same raag performed by different artists (higher = more artist-invariant).
</p>
<div class='legend'><b>Family:</b>&nbsp;{legend}</div>
{"".join(cards)}
</body></html>""", encoding="utf-8")
print(f"\nHTML report -> {HTML_PATH}")
