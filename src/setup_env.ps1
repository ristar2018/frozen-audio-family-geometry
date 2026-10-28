# =====================================================================
# FAPN conda environment setup (reference: RTX 5060 Ti 16GB, Blackwell)
# Run manually in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
# Idempotent: reuses env "fapn"; downgrades/repairs a polluted env as needed.
# Does NOT touch base / chartdetect / smartagri.
#
# HARDWARE: RTX 5060 Ti is Blackwell sm_120. It needs the CUDA 12.8 wheels:
#   torch / torchvision / torchaudio 2.11.0+cu128
#   (older cu124 / torch 2.4 installs load but emit "sm_120 not compatible"
#    and cannot run kernels on this GPU.)
#
# TRANSFORMERS NOTE (verified 2026-10-20):
#   BeatsModel / BeatsFeatureExtractor are NOT shipped in ANY released
#   transformers wheel (checked 4.44 -> 4.53 and 5.14: no models/beats).
#   BEATs is used via Microsoft unilm's custom code (github microsoft/unilm,
#   beats/BEATs.py + backbone.py) and a .pt checkpoint -- NOT via transformers.
#   So E1 pins transformers 4.46.x only because MERT/AST/AudioMAE need a stable
#   transformers, and the self-check below does NOT import BeatsModel.
#   See download_beats.ps1 for fetching the unilm code + .pt weight.
#
# timm pulls torchvision; pinning the matched triplet + a constraints file
# keeps pip from silently swapping the CUDA torch for a newer CPU build.
#
# ASCII-only so PowerShell 5.1 parses it safely (no BOM reliance).
# =====================================================================
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'fapn'
$PY_VER   = '3.10'
$CODE_DIR = $PSScriptRoot
$REQ      = Join-Path $CODE_DIR 'requirements_e1.txt'

Write-Host '=== [1/4] Ensure conda env fapn (Python 3.10) ===' -ForegroundColor Cyan
$envList = conda env list
if ($envList -match ('^' + [regex]::Escape($ENV_NAME) + '\s')) {
    Write-Host "Env '$ENV_NAME' already exists -> skip create."
} else {
    conda create -y -n $ENV_NAME python=$PY_VER
}
conda run -n $ENV_NAME --live-stream python -c "import sys; print('prefix:', sys.prefix)"

Write-Host '=== [2/4] CUDA 12.8 triplet 2.11.0 (Blackwell sm_120) ===' -ForegroundColor Cyan
conda run -n $ENV_NAME --live-stream python -m pip install --no-input `
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host '=== [3/4] Remaining deps (Tsinghua mirror), torch triplet constrained ===' -ForegroundColor Cyan
if (-not (Test-Path $REQ)) { throw "requirements file not found: $REQ" }
$con = Join-Path $env:TEMP '_fapn_constraints.txt'
@(
    'torch==2.11.0'
    'torchvision==0.26.0'
    'torchaudio==2.11.0'
) | Set-Content -Path $con -Encoding ascii
conda run -n $ENV_NAME --live-stream python -m pip install --no-input `
    -r "$REQ" -c "$con" -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host '=== [4/4] Self-check (external file; conda run cannot take multiline -c) ===' -ForegroundColor Cyan
$check = @'
import torch, transformers
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available - torch is a CPU build"
cap = torch.cuda.get_device_capability(0)
print("gpu", torch.cuda.get_device_name(0), "capability sm_%d%d" % cap)
print("torchvision", __import__("torchvision").__version__)
print("torchaudio", __import__("torchaudio").__version__)
print("transformers", transformers.__version__)
# NOTE: BEATs is NOT part of transformers (no BeatsModel in any release).
# It loads via Microsoft unilm custom code; checked separately in the smoke test
# once download_beats.ps1 has fetched beats/ + the .pt checkpoint.
from transformers import AutoModel, AutoFeatureExtractor  # MERT/AST path
print("transformers AutoModel/AutoFeatureExtractor: OK")
import timm, librosa, soundfile
print("timm", timm.__version__, "| librosa", librosa.__version__, "| soundfile", soundfile.__version__)
x = torch.randn(64, 64, device="cuda")
y = (x @ x).sum().item()
print("GPU matmul smoke: OK (sum=%.1f)" % y)
print("ALL_CHECKS_PASSED")
'@
$tmpPy = Join-Path $env:TEMP '_fapn_check.py'
$check | Out-File -FilePath $tmpPy -Encoding ascii
conda run -n $ENV_NAME --live-stream python -X utf8 "$tmpPy"
$rc = $LASTEXITCODE
foreach ($f in @($tmpPy, $con, $reqNoTimm)) {
    if ($f -and (Test-Path $f)) { Remove-Item -Force $f -ErrorAction SilentlyContinue }
}
if ($rc -ne 0) { throw "self-check failed (exit $rc). See messages above." }

$smoke = Join-Path $CODE_DIR 'e1_smoke_test.py'
Write-Host ''
Write-Host 'ENVIRONMENT READY. Next steps:' -ForegroundColor Green
Write-Host "  conda activate $ENV_NAME"
Write-Host '  $env:HF_ENDPOINT = "https://hf-mirror.com"'
Write-Host "  python `"$smoke`""
Write-Host ''
Write-Host 'BEATs is NOT in transformers: run download_beats.ps1 to fetch the' -ForegroundColor Yellow
Write-Host 'Microsoft unilm beats/ code + a .pt checkpoint, then smoke-test beats.' -ForegroundColor Yellow
