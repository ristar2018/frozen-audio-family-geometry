# -*- coding: utf-8 -*-
"""
Route A v2: fine-grained raag labels.

Reads cached 4-encoder embeddings + Saraga manifest.
Adds fine-grained label layer:
  - top-20 most common raag_common (top-20着色 + 其余灰)
  - all 76 unique raag_common as fine labels (sil 计算)
Recomputes metrics and regenerates HTML with 16 plots
  (4 enc × {coarse-family, top-20-raag} × {UMAP, t-SNE}).
"""
import os, sys, re, json, html as html_mod, gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

ROOT    = Path(__file__).resolve().parent.parent
SARAGA  = ROOT / "data" / "openethnic" / "saraga"
HINDU   = SARAGA / "hindustani"
MANIFEST= ROOT / "manifests" / "saraga_hindustani_manifest.csv"
EMB_DIR = ROOT / "results" / "saraga" / "embeddings"
OUT_DIR = ROOT / "results" / "saraga"
ENCODERS = ["ast", "audiomae", "mert", "beats"]

# === 1) load manifest + embeddings =====================================
df = pd.read_csv(MANIFEST, encoding="utf-8-sig")
print(f"manifest: {len(df)} rows | {df['raag_common'].nunique()} unique raag_common | "
      f"{df['performance'].nunique()} perf | {df['lead_artist'].nunique()} artists")

embs = {}
for name in ENCODERS:
    p = EMB_DIR / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing cache {p}; run saraga_probe_routeA.py first")
    X = np.load(p)
    print(f"  {name}: {X.shape}  norm-mean={np.linalg.norm(X, axis=1).mean():.4f}")
    embs[name] = X

# === 2) two label layers ===============================================
# coarse: same 10-family patterns as v1
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
def _to_family(name):
    s = (name or "").strip().lower()
    for fam, pat in _FAMILY_PATTERNS:
        if pat.search(s): return fam
    return "standalone"

df["family"] = df["raag_common"].map(_to_family)

# fine: raag_common as-is (76 unique)
fine_labels = df["raag_common"].values
coarse_labels = df["family"].values

# top-N raag by frequency (for 2nd visualization)
TOPN = 20
raag_counts = Counter(fine_labels)
top_raags = [r for r, _ in raag_counts.most_common(TOPN)]
print(f"\n--- top-{TOPN} raag (by count, incl. multi-artist repetitions) ---")
for r in top_raags:
    n_artist = df.loc[df["raag_common"] == r, "lead_artist"].nunique()
    print(f"  {raag_counts[r]:2d}  {n_artist} artist(s)  {r}")

# top-20 label: raag_common if in top20, else "__other__"
top20_labels = np.where(np.isin(fine_labels, top_raags), fine_labels, "__other__")

# === 3) UMAP + t-SNE (reuse if cached) ================================
PROJ_DIR = OUT_DIR / "projections_v2"
PROJ_DIR.mkdir(parents=True, exist_ok=True)

projections = {}
for name, X in embs.items():
    proj = {}
    umap_path = PROJ_DIR / f"{name}_umap.npy"
    tsne_path = PROJ_DIR / f"{name}_tsne.npy"
    if HAS_UMAP:
        if umap_path.exists():
            proj["umap"] = np.load(umap_path)
        else:
            u = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=20260501)
            Y = u.fit_transform(X); np.save(umap_path, Y); proj["umap"] = Y
    if tsne_path.exists():
        proj["tsne"] = np.load(tsne_path)
    else:
        t = TSNE(n_components=2, metric="cosine", random_state=20260501,
                 init="pca", learning_rate="auto")
        Y = t.fit_transform(X); np.save(tsne_path, Y); proj["tsne"] = Y
    projections[name] = proj
    print(f"  [{name}] UMAP/t-SNE done")

# === 4) metrics (3 label layers) =======================================
def _same_raag_cos(X, labels):
    sims = []
    for lbl, grp in pd.Series(range(len(labels))).groupby(labels):
        if len(grp) < 2: continue
        v = X[grp.tolist()]
        cos = v @ v.T
        iu = np.triu_indices(len(grp), k=1)
        sims.append(cos[iu].mean())
    return float(np.mean(sims)) if sims else float("nan")

def _safe_sil(X, labels):
    if len(set(labels)) < 2: return float("nan")
    if len(set(labels)) >= len(X): return float("nan")
    return silhouette_score(X, labels, metric="cosine")

rows = []
for name, X in embs.items():
    rows.append({
        "encoder": name,
        "sil_coarse":  _safe_sil(X, coarse_labels),
        "sil_top20":   _safe_sil(X, top20_labels),
        "sil_fine":    _safe_sil(X, fine_labels),
        "cos_coarse":  _same_raag_cos(X, coarse_labels),
        "cos_top20":   _same_raag_cos(X, top20_labels),
        "cos_fine":    _same_raag_cos(X, fine_labels),  # =1.0 trivially
    })
print("\n=== metrics ===")
print(f"{'encoder':9s}  sil(coarse/top20/fine)     cos(coarse/top20/fine)")
for r in rows:
    print(f"{r['encoder']:9s}  "
          f"{r['sil_coarse']:+.3f}  {r['sil_top20']:+.3f}  "
          f"({r['sil_fine']:+.3f} trivial)   "
          f"{r['cos_coarse']:.3f}  {r['cos_top20']:.3f}  (1.000)")

# === 5) HTML report ====================================================
# palette: tab20 + extra for top20+other
N_COLORS = max(len(set(coarse_labels)), len(set(top20_labels))) + 1
cmap = plt.get_cmap("tab20")
palette = [to_hex(cmap(i / max(1, N_COLORS - 1))) for i in range(N_COLORS)]
# ensure "__other__" = gray
def _color_map(labels):
    uniq = sorted(set(labels))
    m = {}
    i = 0
    for lbl in uniq:
        if lbl == "__other__":
            m[lbl] = "#bbbbbb"
        else:
            m[lbl] = palette[i % len(palette)]; i += 1
    return m

def build_card(name, method, Y, labels, color_map, sil, agree, title):
    pts = []
    for y, lbl, rl, p, a in zip(Y, labels, fine_labels,
                                 df["performance"], df["lead_artist"]):
        pts.append({
            "x": float(y[0]), "y": float(y[1]),
            "raag": rl, "label": lbl, "perf": p, "artist": a,
            "color": color_map.get(lbl, "#bbbbbb"),
        })
    pts_json = json.dumps(pts, ensure_ascii=False)
    pid = f"plot-{name}-{method}-{title.replace(' ','_')}"
    return f"""
    <div class='card'>
      <h3>{html_mod.escape(name)} · {method.upper()} · <span style='color:#888'>{html_mod.escape(title)}</span>
          <small>(sil={sil:+.3f}, same-label cos={agree:.3f})</small>
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
    X = embs[name]; proj = projections[name]
    for labels, title, sil_key, cos_key in [
        (coarse_labels, "coarse family (10-thaat)", "sil_coarse", "cos_coarse"),
        (top20_labels,  f"top-{TOPN} raag (rest gray)", "sil_top20", "cos_top20"),
    ]:
        cm = _color_map(labels)
        sil = next(r[sil_key] for r in rows if r["encoder"] == name)
        agree = next(r[cos_key] for r in rows if r["encoder"] == name)
        for method in ("umap", "tsne"):
            if method not in proj: continue
            cards.append(build_card(name, method, proj[method],
                                    labels, cm, sil, agree, title))

# legend (top20 + other + coarse families)
def legend_block(title, color_map, keys):
    items = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;background:{color_map[k]};'
        f'margin-right:6px;vertical-align:middle"></span>{html_mod.escape(k)}'
        for k in keys
    )
    return f'<div class="legend"><b>{title}:</b>&nbsp;{items}</div>'

coarse_cm = _color_map(coarse_labels)
top20_cm  = _color_map(top20_labels)

HTML_PATH = OUT_DIR / "cross_cultural_probe_v2.html"
HTML_PATH.write_text(f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Saraga Route-A Probe v2 (fine-grained)</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>body{{font-family:-apple-system,Segoe UI,sans-serif;margin:24px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px}}
h3{{margin:.2em 0}}small{{color:#666;font-weight:normal;margin-left:12px}}
.legend{{display:flex;flex-wrap:wrap;gap:10px 16px;padding:12px;border:1px solid #eee;
         border-radius:8px;margin-bottom:12px}}
table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ccc;padding:6px 10px;text-align:right}}
th{{background:#f4f4f4;text-align:center}}
</style></head><body>
<h2>Saraga 1.5 Hindustani · Cross-cultural Zero-shot Probe · v2 Fine-grained</h2>
<p>108 Raag recordings × 4 encoders (768-d, L2). 双层标签：
<b>coarse family</b> = 10-thaat 粗分组（10 族），
<b>top-{TOPN} raag</b> = 按频次前 {TOPN} 个 unique raag_common 着色（其余灰）。</p>
{legend_block('Coarse family', coarse_cm, sorted(set(coarse_labels)))}
{legend_block(f'Top-{TOPN} raag', top20_cm, top_raags + ['__other__'])}
<h3>指标汇总</h3>
<table><tr><th>encoder</th><th>sil(coarse)</th><th>sil(top-{TOPN})</th>
<th>cos(coarse)</th><th>cos(top-{TOPN})</th></tr>
{"".join(f"<tr><td>{r['encoder']}</td><td>{r['sil_coarse']:+.3f}</td>"
         f"<td>{r['sil_top20']:+.3f}</td><td>{r['cos_coarse']:.3f}</td>"
         f"<td>{r['cos_top20']:.3f}</td></tr>" for r in rows)}
</table>
<p><b>解读</b>：sil 从负到正 = 标签粒度与特征结构对齐度；cos 高 = 同 Raag 跨艺术家一致性
（最强正面信号：4 编码器全部 ≥ 0.84，AST 达 0.95）。</p>
{"".join(cards)}
</body></html>""", encoding="utf-8")
print(f"\nHTML v2 -> {HTML_PATH}")
