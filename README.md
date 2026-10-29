# Boundary Conditions of Frozen Audio Encoders for Recovering Instrument-Family Structure

Reproduction code for *"Boundary Conditions of Frozen Audio Encoders for Recovering
Instrument-Family Structure: A Cross-Corpus Geometric Diagnosis with
Random-Initialisation Controls"* (submitted to *Neurocomputing*).

The repository diagnoses **embedding geometry**, not just classifier accuracy:
whether frozen pretrained audio encoders (**BEATs, MERT, AST, AudioMAE**) organise
musical instruments into family-level structure, measured with a permutation-calibrated
family silhouette, leave-one-out *k*-NN accuracy, and the within/between cosine gap,
with random-initialisation, cross-dataset, paired-stem, and same-family controls.

> The archive ships the **analysis code, the split manifests and every JSON output** the
> manuscript reports, so each table can be checked against the file that produced it.

---

## 1. Repository layout

```
.
├── src/                         # analysis pipeline (all paths derive from this folder's parent)
│   ├── data_loader.py           #   FolkMusic/NSynth loaders, 15->3 family map, recording-isolated split
│   ├── encoders.py              #   BEATs / MERT / AST / AudioMAE wrappers, L2-normalised
│   ├── analysis.py              #   silhouette, exact permutation test, cosine gap, CKA
│   ├── e1_*.py                  #   stage E1: embedding extraction + primary analysis
│   ├── e2_supplement_*.py       #   stage E2: held-out confirmatory split + MFCC
│   ├── e3_*.py                  #   stage E3: permutation/CI, external, random-init, Saraga controls ...
│   ├── e4_*.py                  #   stage E4: Carnatic paired stems + solo/same-family corpora
│   ├── build_saraga_manifest.py #   Saraga Hindustani manifest builder
│   ├── saraga_probe_routeA.py    #   Saraga embeddings (cached .npy) + coarse raag-family report
│   ├── saraga_probe_routeA_v2.py #   fine-grained raag layer (76 labels, 16-panel report)
│   ├── fetch_supp_datasets.py   #   downloader + md5 for the supplementary corpora
│   ├── download_beats.ps1       #   BEATs checkpoint download (official then mirror)
│   └── setup_env.ps1            #   conda environment creation
├── scripts/
│   ├── e1_visualization_pack.py #   overview figures + HTML report from cached results
│   ├── e3_fig_umap_tsne_fix.py  #   main-text UMAP / t-SNE panels
│   ├── e3_fig_fix.py            #   confusion-matrix and encoder-comparison panels
│   └── e3_fig_familybar_fix.py  #   within/between cosine bar panel
├── results/                     # the JSON of every script (cached embeddings not shipped)
├── manifests/                   # sample lists that define every split and sampling rule
│   ├── splits/                  #   FolkMusic dev/test recording-isolated fragments
│   ├── supplementary/           #   Medley-solos-DB, BRID, Beijing-opera, Mridangam, Carnatic stems
│   └── saraga_hindustani_manifest.csv
├── config/seeds.env             # fixed seeds used throughout
├── requirements.txt
├── MANIFEST_SHA256.txt          # SHA-256 of every file (verify: Section 8)
└── LICENSE
```

All scripts resolve the project root as the **parent of the `src/` directory**
(`Path(__file__).resolve().parent.parent`). Clone this repository, place the
downloaded datasets under `data/` next to `src/` as described below, and no
absolute path needs editing.

## 2. Environment

```powershell
# create the dedicated conda environment (Windows; PowerShell)
powershell -ExecutionPolicy Bypass -File src/setup_env.ps1
conda activate fapn
```
or manually:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install a CUDA-matched torch separately, e.g.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Set a Hugging Face mirror where needed:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 3. Model checkpoints

* **MERT, AST, AudioMAE** load through `transformers`/`timm` from the Hugging Face Hub.
* **BEATs is not released inside `transformers`**. It uses the Microsoft `unilm`
  custom code (`BEATs.py`, `backbone.py`) plus a `.pt` checkpoint. Download it with

  ```powershell
  powershell -ExecutionPolicy Bypass -File src/download_beats.ps1
  ```

  and place the checkpoint under `models/beats/`. Smoke-test all encoders first:

  ```bash
  python src/e1_smoke_test.py --encoders ast mert audiomae beats
  ```

## 4. Data

Audio is **not** included. Download each corpus from its distributor and keep the
licence terms. `src/fetch_supp_datasets.py` lists URLs, sizes, md5 hashes, and licences
for the supplementary corpora and verifies downloads:

```bash
python src/fetch_supp_datasets.py --list                 # show all candidates
python src/fetch_supp_datasets.py --dataset medley-solos-db
python src/fetch_supp_datasets.py --all-open             # every directly downloadable corpus
```

Expected layout (only the manifests are committed; audio is git-ignored):

```
data/
├── folkmusic/            # FolkMusic (Li et al., 2023; CC BY 4.0; Zenodo 10.5281/zenodo.8012071)
├── nsynth/               # NSynth acoustic entities + nsynth_acoustic_metadata.csv
├── openethnic/saraga/    # Saraga 1.5 Hindustani/Carnatic (CC BY-NC-SA 4.0)
└── supp/                 # medley-solos-db, brid, beijing-opera-percussion,
                          # mridangam-stroke, saraga-carnatic
```

The `manifests/` directory commits the exact fragment lists, split identifiers and
sampling rules used in the paper (FolkMusic dev/test; 3,200 Medley clips across
227 songs; 274 BRID clips; 235 Beijing-opera clips; 1,200 mridangam clips;
106 Carnatic performances).

## 5. Reproducing the analyses

Run from the repository root. Outputs are written to `results/<stage>/`.

```bash
# E1  primary corpus: embedding extraction (cache) + family geometry
python src/e1_embedding_analysis.py
# E2  confirmatory held-out split + MFCC extraction/analysis
python src/e2_supplement_extract.py
python src/e2_supplement_analyze.py
# E3  exact permutation test, bootstrap CIs, external corpora, random init, Saraga
python src/e3_exact_permutation.py
python src/e3_bootstrap_ci.py
python src/e3_external.py
python src/e3_random_init.py
python src/e3_mfcc_baseline.py
python src/e3_saraga_controls.py
python src/e3_geometry_intervention.py
python src/e3_random_init_pvalues.py   # exact p for each random-initialisation seed
python src/e3_irmas_exact.py           # enumerated family-label null on IRMAS (831,600)
python src/e3_saraga_twoway.py         # two-way cluster bootstrap + per-query chance
# E4  Carnatic paired-stem control, then solo / same-family corpora
python src/e4_carnatic_stem_manifest.py
python src/e4_carnatic_mix_vs_stem.py
python src/e4_supp_prepare.py
python src/e4_supp_stats.py
# supplementary analyses and back-fills (each reads the caches above)
python src/e1_nsynth_and_15class.py
python src/build_saraga_manifest.py
python src/saraga_probe_routeA.py       # Saraga embeddings + report (needs the audio under data/)
python src/saraga_probe_routeA_v2.py    # fine-grained raag layer on the cached embeddings
python src/e3_audit_fix.py
python src/e3_dev_clip_knn.py
python src/e3_family_partitions.py
python src/e3_folk15_perclass.py
python src/e3_folk15_sil.py
python src/e3_retrieval_fix.py
# figures (written to results/e1/figures/)
python scripts/e1_visualization_pack.py
python scripts/e3_fig_umap_tsne_fix.py
python scripts/e3_fig_fix.py
python scripts/e3_fig_familybar_fix.py
```

Headline statistic definitions (in `src/analysis.py`):

* **family silhouette** at clip level against an exact instrument-label null
  (permutation-calibrated, reported with the null 95th percentile);
* **leave-one-out 1-NN** on cosine similarity, excluding near-duplicates of the same
  recording/artist/tonic group;
* **within/between cosine gap** Δcos with equal label weighting;
* a **group-disjoint silhouette** variant for the supplementary corpora (pairs from
  the same song, performer, or tonic are masked; the mask depends only on grouping
  and is therefore valid under permutation).

## 6. Result files

`results/` ships the JSON written by each script — the exact state behind the submitted
numbers — plus the split manifests. The cached embeddings are omitted to keep the archive
small; the end of this section lists what writes each cache. The JSONs map to the
manuscript as follows:

| file | manuscript content |
|---|---|
| `e1/*.json` | primary family geometry, 15-class silhouette, clip kNN |
| `e2_supplement/e2_supplement_results.json` | confirmatory held-out split, MFCC baseline |
| `e3/exact_permutation.json` | exact permutation test on the held-out split |
| `e3/bootstrap_ci.json` | paired bootstrap confidence intervals |
| `e3/external_nsynth.json`, `e3/external_irmas.json` | cross-dataset transfer |
| `e3/irmas_exact_family.json` | enumerated family-label null on IRMAS |
| `e3/random_init.json`, `e3/random_init_pvalues.json` | random-initialisation control and its per-seed exact p |
| `e3/mfcc_baseline.json` | MFCC reference |
| `e3/saraga_controls.json`, `e3/saraga_twoway.json` | Hindustani negative control, 2x2 pair design, two-way cluster bootstrap, per-query chance |
| `e3/geometry_intervention.json` | anisotropy and covariance interventions (Table 7, Tables A.2–A.3) |
| `e3/fig_familybar.json` | within/between cosine means behind the family-similarity figure |
| `e4/carnatic_mix_vs_stem.json` | paired-stem control |
| `e4/carnatic_stems_summary.json` | Carnatic stem inventory: 249 items, 168 with stems, 106 paired performances |
| `e4/*supp*.json` | solo-recording and same-family controls |
| `e4/data_quality.json` | corpus QA: durations, sample rates, and the two clips that fail MFCC deltas |

Re-running a script overwrites its JSON, so the shipped files are the exact state behind
the submitted numbers.

### Regenerating the cached embeddings

No array has to be downloaded: every cache is written by one of the scripts above, from
the corpora of Section 4 and the checkpoints of Section 3.

| cache | written by | size |
|---|---|---|
| `e1/embeddings/*.npz` (8) | `e1_embedding_analysis.py` | 265 MB |
| `e2_supplement/embeddings/*.npz` (4), `mfcc/*` (3) | `e2_supplement_extract.py` | 37 MB |
| `e3/random_init/embeddings/*.npz` (24) | `e3_random_init.py` | 391 MB |
| `e3/irmas_emb/*.npz` (4) | `e3_external.py` | 73 MB |
| `e4/supp/*.npy` (40), `e4/embeddings/*`, `e4/mfcc/*` | `e4_supp_prepare.py`, `e4_carnatic_mix_vs_stem.py` | 67 MB |
| `saraga/embeddings/*.npy` (4), `projections_v2/*` | `saraga_probe_routeA.py`, `saraga_probe_routeA_v2.py` | 1 MB |
| `e3/_proj_coords.npz` (1) | `scripts/e3_fig_umap_tsne_fix.py` | 0.5 MB |
| **total** | 126 arrays | **833 MB** |

Each stage writes its cache before analysing it, so running the commands of Section 5 in
order rebuilds the arrays and, with them, the JSON. Feature extraction is the only
expensive part (GPU recommended for the four encoders; the random-initialisation stage
runs twelve additional models over the same two splits).

## 7. Notes on scope and licences

* The code is released under the MIT licence (`LICENSE`). This covers **code only**;
  it does not change the licences of the datasets or checkpoints.
* Saraga corpora are CC BY-NC-SA 4.0 and the mridangam stroke dataset is CC BY-NC 4.0;
  treat their outputs as non-commercial and share-alike where required.
* The Medley-solos-DB, BRID, and Beijing-opera percussion datasets are CC BY 4.0.
* Random seeds are fixed (`config/seeds.env`); permutation counts are 2,000 for the
  main/supplementary tests and 500 for the mridangam stroke analysis.
* Two extremely short clips fail MFCC delta extraction and are reported in the logs
  (1 of 236 Beijing-opera clips; 1 of 1,200 mridangam clips); all statistics use the
  common valid rows documented in the paper.

## 8. Availability

The archive is 4 MB in 92 files: code, split manifests, the result JSON and the
SHA-256 manifest, with no audio, checkpoints, figures or feature caches (Section 6
lists what rebuilds each cache from the public corpora).

### Verifying the download

The manifest is byte-exact, so the hashes hold only if no line-ending conversion
happened on the way — this is why `.gitattributes` sets `* -text`. After cloning:

```bash
# Linux / macOS
sha256sum -c MANIFEST_SHA256.txt
```

```powershell
# Windows PowerShell (no sha256sum available)
Get-Content MANIFEST_SHA256.txt | Where-Object { $_ -and $_ -notmatch '^#' } | ForEach-Object {
  $hash, $path = $_ -split '  ', 2
  if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $hash.ToUpper()) { "BAD: $path" }
}
```

## 9. Citation

A BibTeX entry will be provided upon publication. For now, please cite the manuscript
title above and the dataset DOIs recorded in `src/fetch_supp_datasets.py`.
