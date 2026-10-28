# =====================================================================
# Fetch BEATs for FAPN E1 (Microsoft unilm custom-code route)
#
# IMPORTANT (verified 2026-10-20):
#   * BeatsModel / BeatsFeatureExtractor are NOT in any released
#     transformers wheel (checked 4.44-4.53 and 5.14). Do NOT import them.
#   * BEATs ships as custom code in github.com/microsoft/unilm (beats/):
#       BEATs.py (imports backbone), backbone.py, modules.py,
#       quantizer.py, Tokenizers.py
#   * The official Azure checkpoints (valle.blob.core.windows.net/...) now
#     return 409 "Public access is not permitted". Identical official-format
#     .pt files are mirrored on Hugging Face and reachable via hf-mirror.
#
# This script downloads:
#   models\beats\code\  <- the 5 unilm .py files (from raw.githubusercontent)
#   models\beats\BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt  <- weight
#
# The encoders.py BEATsEncoder loads models\beats\code as a package, builds
# BEATsConfig(checkpoint['cfg']) and BEATs(cfg), load_state_dict(checkpoint['model']).
#
# ASCII-only. Run:
#   powershell -ExecutionPolicy Bypass -File download_beats.ps1
# =====================================================================
$ErrorActionPreference = 'Stop'

$ROOT      = Split-Path $PSScriptRoot -Parent
$DEST      = Join-Path $ROOT 'models\beats'
$CODE_DIR  = Join-Path $DEST 'code'
New-Item -ItemType Directory -Force -Path $CODE_DIR | Out-Null

# Mirror base can be overridden:  $env:HF_BASE = "https://huggingface.co"
$HF_BASE = if ($env:HF_BASE) { $env:HF_BASE } else { 'https://hf-mirror.com' }

# Weight choice. AS2M-finetuned cpt2 is the best general audio representation
# and matches "AST AudioSet / AudioMAE AS2M" for a fair encoder comparison.
# Override with:  $env:BEATS_WEIGHT = "iter3"  (self-supervised pre-only)
$WEIGHT_CHOICE = if ($env:BEATS_WEIGHT) { $env:BEATS_WEIGHT } else { 'cpt2' }

switch ($WEIGHT_CHOICE) {
    'iter3' {
        $REPO = 'lpepino/beats_ckpts'
        $FILE = 'BEATs_iter3.pt'
    }
    'as2m' {
        $REPO = 'lpepino/beats_ckpts'
        $FILE = 'BEATs_iter3_plus_AS2M.pt'
    }
    default {
        $REPO = 'WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2'
        $FILE = 'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    }
}
$WEIGHT_URL = "$HF_BASE/$REPO/resolve/main/$FILE"
$WEIGHT_DST = Join-Path $DEST $FILE

Write-Host '=== [1/2] Download Microsoft unilm BEATs source code ===' -ForegroundColor Cyan
$codeFiles = @('BEATs.py', 'backbone.py', 'modules.py', 'quantizer.py', 'Tokenizers.py')
foreach ($f in $codeFiles) {
    $u = "https://raw.githubusercontent.com/microsoft/unilm/master/beats/$f"
    $o = Join-Path $CODE_DIR $f
    Write-Host "  $f"
    curl.exe -sS -L -m 120 --retry 3 --ssl-no-revoke -o "$o" "$u"
    if (-not (Test-Path $o) -or ((Get-Item $o).Length -lt 100)) {
        throw "failed to download $u (GitHub raw may be blocked; use a proxy)."
    }
}
# make code dir an importable package
$init = Join-Path $CODE_DIR '__init__.py'
if (-not (Test-Path $init)) { Set-Content -Path $init -Value '' -Encoding ascii }

Write-Host '=== [2/2] Download BEATs checkpoint (.pt) ===' -ForegroundColor Cyan
Write-Host "  choice=$WEIGHT_CHOICE  repo=$REPO  file=$FILE"
if (Test-Path $WEIGHT_DST) {
    $mb = [math]::Round((Get-Item $WEIGHT_DST).Length / 1MB, 1)
    Write-Host "  already present ($mb MB): $WEIGHT_DST  (delete to re-download)"
} else {
    # aria2c if available (16 conn + resume), else curl with resume
    $aria = Get-Command aria2c -ErrorAction SilentlyContinue
    if ($aria) {
        & aria2c -c -x16 -s16 -k1M --file-allocation=none -d "$DEST" -o "$FILE" "$WEIGHT_URL"
    } else {
        curl.exe -L -C - --retry 3 -o "$WEIGHT_DST" "$WEIGHT_URL"
    }
    if (-not (Test-Path $WEIGHT_DST) -or ((Get-Item $WEIGHT_DST).Length -lt 100MB)) {
        throw "weight download looks incomplete: $WEIGHT_DST. Retry or set BEATS_WEIGHT/use a proxy."
    }
}

# quick load sanity check inside the fapn env
Write-Host '=== Verify checkpoint loads with unilm code ===' -ForegroundColor Cyan
$verify = @'
import os, sys, torch
# __file__ is NOT in globals() when running a script; resolve directly.
root = r"MODELS_BEATS"
code = os.path.join(root, "code")
sys.path.insert(0, code)
from BEATs import BEATs, BEATsConfig
import glob
pts = glob.glob(os.path.join(root, "*.pt"))
assert pts, "no .pt found"
p = pts[0]
ckpt = torch.load(p, map_location="cpu", weights_only=False)
cfg = BEATsConfig(ckpt["cfg"])
model = BEATs(cfg)
model.load_state_dict(ckpt["model"])
print("loaded:", os.path.basename(p), "| embed_dim =", cfg.cfg.encoder_embed_dim)
print("BEATS_READY")
'@
# replace placeholder with the real models\beats path (avoid quoting pain)
$verify = $verify -replace 'r"MODELS_BEATS"', ('r"' + ($DEST -replace '\\','\\') + '"')
$tmp = Join-Path $env:TEMP '_verify_beats.py'
$verify | Out-File -FilePath $tmp -Encoding ascii
conda run -n fapn --live-stream python -X utf8 "$tmp"
$rc = $LASTEXITCODE
if ($tmp -and (Test-Path $tmp)) { Remove-Item -Force $tmp -ErrorAction SilentlyContinue }
if ($rc -ne 0) { throw "BEATs verify failed (exit $rc)." }

Write-Host ''
Write-Host 'DONE. BEATs code + weight are under:' -ForegroundColor Green
Write-Host "  $DEST"
