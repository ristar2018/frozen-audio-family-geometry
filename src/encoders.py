# -*- coding: utf-8 -*-
"""
FAPN E1 encoders (BEATs iter3, MERT v1-330M, AST, AudioMAE)

Shared conventions:
  * audio is resampled to each encoder target sample rate; mono;
  * output phi = 768-dimensional vector after masked-mean pooling over time frames or patches;
  * phi is L2-normalised before it leaves the encoder (unit sphere, matching the cosine geometry used by the silhouette);
  * true batched forward pass (padded within the batch); failures are recorded one by one and never silently pollute the embeddings.

Runtime (dedicated conda environment fapn; do not reuse another environment transformers build):
  torch/torchvision/torchaudio 2.11.0+cu128 (RTX 5060 Ti is Blackwell sm_120),
  transformers 4.46.x (for MERT and AST), timm (AudioMAE), soundfile/librosa.
  AudioMAE uses the timm port of the official weights with the torchaudio kaldi mel front end.

BEATs (important; verified by direct inspection):
  BeatsModel / BeatsFeatureExtractor are not part of any released transformers wheel
  (4.44 to 4.53 and 5.14 all lack models/beats). BEATs therefore requires the Microsoft unilm
  custom code: models/beats/code/{BEATs.py, backbone.py, modules.py,
  quantizer.py, Tokenizers.py} plus an official-format .pt checkpoint, fetched by download_beats.ps1.
  The front end is torchaudio kaldi fbank (same settings as AudioMAE: 16 kHz / 128 mel / 1024 frames).
"""
import os
import io
import sys
import wave
import glob
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"
# BEATs custom code and .pt checkpoint (download_beats.ps1 writes them to models/beats/)
_LOCAL_BEATS_DIR = MODELS_ROOT / "beats"
_LOCAL_BEATS_CODE = _LOCAL_BEATS_DIR / "code"

from transformers import AutoModel, AutoFeatureExtractor, AutoConfig


# ============================================================
# Configuration
# ============================================================
def _default_beats_source():
    """
    BEATs has no Hugging Face model id, so this returns the local .pt checkpoint path.
    Priority: FAPN_BEATS_PATH (pointing at a .pt) > models/beats/*.pt (first match) > None.
    When None, create_encoder raises a clear "run download_beats.ps1 first" error.
    """
    env = os.environ.get("FAPN_BEATS_PATH")
    if env and Path(env).exists():
        return env
    if _LOCAL_BEATS_DIR.exists():
        pts = sorted(glob.glob(str(_LOCAL_BEATS_DIR / "*.pt")))
        if pts:
            return pts[0]
    return None


ENCODER_CONFIGS = {
    "beats": {
        "hf_id": None,  # BEATs has no HF id; the weight path is resolved at runtime (see _default_beats_source)
        "sr": 16000, "dim": 768, "frontend": "beats_kaldi", "default_bs": 1,
        "trust_remote_code": False,
        "n_frames": 1024, "n_mels": 128,
        "desc": "BEATs iter3+ (Microsoft unilm code, AudioSet; .pt checkpoint)",
    },
    "mert": {
        "hf_id": "m-a-p/MERT-v1-330M",
        "sr": 24000, "dim": 768, "hidden_dim": 1024,  # MERT-330M hidden size is 1024, projected to 768
        "frontend": "waveform", "default_bs": 1,
        "trust_remote_code": True,   # the repository ships modeling_MERT.py
        "desc": "MERT v1-330M (m-a-p, music SSL; 1024 -> linear proj -> 768)",
    },
    "ast": {
        "hf_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "sr": 16000, "dim": 768, "frontend": "melfbank", "default_bs": 1,
        "trust_remote_code": False,
        "desc": "AST (MIT, AudioSet finetuned)",
    },
    "audiomae": {
        "hf_id": "gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        "sr": 16000, "dim": 768, "frontend": "audiomae_mel", "default_bs": 1,
        "trust_remote_code": False,
        "desc": "AudioMAE ViT-B (AS-2M MAE pretrain), timm port",
    },
}


# ============================================================
# Dependency-free WAV decoding (FolkMusic ships WAV content under a .mp3 extension)
# ============================================================
def load_audio_mono(path, target_sr, max_sec=20.0):
    """
    Read audio, convert to float32 mono and resample to target_sr.
    Important guard: large corpora such as Saraga contain MP3 files up to 71 minutes long.
    Decoding such a file into memory exhausts RAM (the audioread/librosa internal buffer exceeds 4 GB).
    ffmpeg therefore decodes only the first max_sec seconds into a temporary WAV, keeping memory bounded.
    """
    path = str(path)
    try:
        file_bytes = Path(path).stat().st_size
    except Exception:
        file_bytes = 0
    # coarse length guard: audio above 1 MB is almost certainly longer than 20 s, so decode only the head with ffmpeg
    if file_bytes > 1_000_000:
        return _load_via_ffmpeg_crop(path, target_sr, max_sec)
    try:
        with wave.open(path, "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        audio = _pcm_to_float(raw, sw)
        if nch > 1:
            audio = audio.reshape(-1, nch).mean(axis=1)
        if sr != target_sr:
            audio = _resample_linear(audio, sr, target_sr)
        return audio.astype(np.float32)
    except Exception:
        return _load_with_backend(path, target_sr, max_sec=max_sec)


def _load_via_ffmpeg_crop(path, target_sr, max_sec):
    """Decode only the first max_sec seconds with ffmpeg (avoids whole-file decoding of long audio).
    stderr goes to a temporary log file rather than capture_output, so subprocess error messages are not
    buffered inside Python memory; stdout is discarded to a file as well. The child process only holds the decode buffer.
    """
    import subprocess, tempfile
    tmp = Path(tempfile.gettempdir()) / f"_fapn_crop_{os.getpid()}_{hash(path) & 0xffffffff:x}.wav"
    err_log = tmp.with_suffix(".err")
    try:
        with open(err_log, "wb") as ef:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", path,
                "-t", str(max_sec),
                "-ar", str(target_sr), "-ac", "1",
                "-f", "wav", str(tmp),
            ]
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=ef, timeout=60)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 44:
            err_msg = err_log.read_text(errors="ignore")[:200] if err_log.exists() else "no tmp"
            raise RuntimeError(f"ffmpeg crop failed (rc={r.returncode}): {err_msg}")
        return load_audio_mono(str(tmp), target_sr, max_sec=max_sec * 2)
    finally:
        for f in (tmp, err_log):
            try: f.unlink()
            except Exception: pass


def _pcm_to_float(raw, sampwidth):
    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as32 = (b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16))
        as32 = np.where(as32 & 0x800000, as32 - 0x1000000, as32)
        x = as32.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width {sampwidth}")
    return x


def _resample_linear(x, sr_in, sr_out):
    if len(x) == 0 or sr_in == sr_out:
        return x
    duration = len(x) / sr_in
    n_out = int(round(duration * sr_out))
    if n_out <= 1:
        return x[:1].astype(np.float32)
    idx = np.linspace(0.0, 1.0, n_out, endpoint=False) * (len(x) - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = (idx - i0).astype(np.float32)
    return x[i0] * (1 - frac) + x[i1] * frac


def _load_with_backend(path, target_sr, max_sec=20.0):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = _resample_linear(audio, sr, target_sr)
        # truncate at max_sec to keep long audio from exhausting memory
        max_n = int(max_sec * target_sr)
        if len(audio) > max_n:
            audio = audio[:max_n]
        return audio.astype(np.float32)
    except Exception:
        import librosa
        audio, _ = librosa.load(path, sr=target_sr, mono=True, duration=max_sec)
        return audio.astype(np.float32)


def _masked_mean(seq, mask):
    """seq:(B,T,D), mask:(B,T) with 1 = valid frame. Returns (B,D)."""
    m = mask.unsqueeze(-1).to(seq.dtype)
    s = (seq * m).sum(dim=1)
    cnt = m.sum(dim=1).clamp(min=1.0)
    return s / cnt


def _l2(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


# ============================================================
# Base class
# ============================================================
class BaseEncoder:
    name = "base"

    def __init__(self, model_path=None, device=None):
        cfg = ENCODER_CONFIGS[self.name]
        self.cfg = cfg
        self.hf_id = model_path or cfg["hf_id"]
        self.sr = cfg["sr"]
        self.dim = cfg["dim"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    @torch.no_grad()
    def encode_batch(self, paths):
        """Read a batch of audio -> (B,768) L2-normalised phi; returns (emb, ok_flags)."""
        raise NotImplementedError

    @torch.no_grad()
    def encode_dataset(self, paths, batch_size=None, log_every=50, on_fail="collect"):
        """
        Full extraction.
        on_fail: "collect" records failing indices and continues; "raise" propagates the error.
        Returns dict(embeddings np.ndarray[N_ok,768], ok_idx list[int], failed list[(idx,path,err)])
        """
        bs = batch_size or self.cfg["default_bs"]
        embs, ok_idx, failed = [], [], []
        n = len(paths)
        for s in range(0, n, bs):
            batch = list(range(s, min(s + bs, n)))
            try:
                e, flags = self.encode_batch([paths[i] for i in batch])
            except Exception as ex:
                # if a whole batch fails, retry item by item to locate the bad file
                for gi in batch:
                    try:
                        e1, f1 = self.encode_batch([paths[gi]])
                        embs.append(e1.cpu()); ok_idx.append(gi)
                    except Exception as ex1:
                        failed.append((gi, paths[gi], repr(ex1)))
                        if on_fail == "raise":
                            raise
                continue
            for k, gi in enumerate(batch):
                if flags[k]:
                    embs.append(e[k:k + 1].cpu()); ok_idx.append(gi)
                else:
                    failed.append((gi, paths[gi], "encode returned invalid"))
            if ((s // bs) + 1) % log_every == 0:
                print(f"    [{self.name}] {min(s+bs,n)}/{n}  failed={len(failed)}", flush=True)
        if not embs:
            raise RuntimeError(f"{self.name}: no embeddings extracted")
        emb = torch.cat(embs, dim=0).numpy().astype(np.float32)
        order = np.argsort(ok_idx)
        emb = emb[order]; ok_idx = [ok_idx[i] for i in order]
        return {"embeddings": emb, "ok_idx": ok_idx, "failed": failed}


# ============================================================
# Waveform front-end encoders: BEATs and MERT
# ============================================================
class WaveformEncoder(BaseEncoder):
    def load(self):
        kwargs = {"trust_remote_code": self.cfg["trust_remote_code"]}
        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id, **kwargs)
        self.model = AutoModel.from_pretrained(self.hf_id, **kwargs)
        self.model.eval().to(self.device)
        return self

    @torch.no_grad()
    def encode_batch(self, paths):
        wavs, flags = [], []
        for p in paths:
            try:
                wavs.append(load_audio_mono(p, self.sr)); flags.append(True)
            except Exception:
                wavs.append(np.zeros(self.sr, dtype=np.float32)); flags.append(False)
        # MERT Wav2Vec2FeatureExtractor requires an explicit max_length when truncation=True;
        # cap at 10 s at the model sample rate (MERT was trained on 240000 samples = 10 s at 24 kHz).
        max_len = max(10 * self.sr, max(len(w) for w in wavs))
        inputs = self.processor(
            wavs, sampling_rate=self.sr, return_tensors="pt",
            padding=True, truncation=True, max_length=max_len,
        )
        input_values = inputs["input_values"].to(self.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        out = self.model(input_values, attention_mask=attn,
                         output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]                      # (B,T,D)
        if attn is not None:
            # the waveform sampling mask may have a different frame count than T; align by proportional mapping
            if attn.shape[1] != hs.shape[1]:
                B = attn.shape[0]
                T = hs.shape[1]
                ratio = hs.shape[1] / attn.shape[1]
                pos = (torch.arange(T, device=self.device).float() / ratio).long()
                frame_mask = attn[:, pos.clamp(max=attn.shape[1] - 1)]
            else:
                frame_mask = attn
            pooled = _masked_mean(hs, frame_mask)
        else:
            pooled = hs.mean(dim=1)
        pooled = _l2(pooled)
        return pooled, flags


class BEATsEncoder(BaseEncoder):
    """
    BEATs through the Microsoft unilm custom code (not transformers).
    Weights are the official-format .pt (dict with keys cfg and model); the code lives in models/beats/code.
    The 768-dimensional embedding is taken by bypassing the AudioSet classification head (finetuned checkpoints set cfg.finetuned_model=True).
    """
    name = "beats"
    WIN_SEC = 10.0

    def load(self):
        ckpt_path = self.hf_id or _default_beats_source()
        if not ckpt_path or not Path(ckpt_path).exists():
            raise FileNotFoundError(
                "No BEATs .pt checkpoint found. Run src/download_beats.ps1 first: "
                "it fetches the unilm code into models/beats/code and the weights into models/beats/*.pt; "
                "alternatively pass --model_path pointing at a .pt file, or set FAPN_BEATS_PATH."
            )
        code_dir = str(_LOCAL_BEATS_CODE)
        if not (_LOCAL_BEATS_CODE / "BEATs.py").exists():
            raise FileNotFoundError(
                f"Missing the BEATs custom code directory {code_dir}; run src/download_beats.ps1 first."
            )
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from BEATs import BEATs, BEATsConfig  # unilm top-level module (not transformers)
        self._BEATsCls = BEATs
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval().to(self.device)
        self.model = model
        self.ckpt_name = Path(ckpt_path).name
        self.dim = cfg.encoder_embed_dim
        return self

    @torch.no_grad()
    def encode_batch(self, paths):
        target_len = int(self.sr * self.WIN_SEC)
        wavs, flags = [], []
        for p in paths:
            try:
                wav = load_audio_mono(p, self.sr)        # float32 [-1,1], 16k mono
                wav = torch.from_numpy(wav)
                if wav.numel() < target_len:             # zero-pad short audio at the tail
                    wav = F.pad(wav, (0, target_len - wav.numel()))
                else:                                    # otherwise truncate to 10 s so the batch is uniform
                    wav = wav[:target_len]
                wavs.append(wav)
                flags.append(True)
            except Exception:
                wavs.append(torch.zeros(target_len))
                flags.append(False)
        source = torch.stack(wavs, dim=0).to(self.device)
        model = self.model
        # extract_features on a finetuned checkpoint returns 527-dim logits; bypass the head to take frame-level x
        saved_predictor = model.predictor
        model.predictor = None
        try:
            x, padding_mask = model.extract_features(source, padding_mask=None)
        finally:
            model.predictor = saved_predictor
        # x:(B,T,768); padding_mask is None because the input is already a uniform 10 s, so take the time mean
        pooled = x.mean(dim=1)
        pooled = _l2(pooled)
        return pooled, flags


class MERTEncoder(WaveformEncoder):
    name = "mert"

    def load(self):
        super().load()
        # The MERT-v1-330M hidden size is 1024 while the E1 protocol asks for 768 dimensions plus L2.
        # Generate a projection matrix P (1024, 768) with orthonormal columns from a fixed seed,
        # so that ||P @ x|| = ||x|| (length preserved, hence the L2 normalisation is unchanged);
        # this keeps every run identical (deterministic reduction that preserves cosine geometry).
        hid = self.cfg.get("hidden_dim", 1024)
        if hid != self.dim:
            gen = torch.Generator(device="cpu").manual_seed(20260501 + 330)
            A = torch.randn(hid, self.dim, device="cpu", generator=gen)
            Q, _ = torch.linalg.qr(A)          # Q: (1024, 768) with orthonormal cols
            self._proj = Q.to(self.device)      # move to GPU for the matrix product
        else:
            self._proj = None
        return self

    @torch.no_grad()
    def encode_batch(self, paths):
        pooled, flags = super().encode_batch(paths)
        if self._proj is not None:
            pooled = pooled @ self._proj              # (B,768)
            pooled = _l2(pooled)
        return pooled, flags


# ============================================================
# AST: the processor builds the log-mel internally
# ============================================================
class ASTEncoder(BaseEncoder):
    name = "ast"

    def load(self):
        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id)
        self.model.eval().to(self.device)
        return self

    @torch.no_grad()
    def encode_batch(self, paths):
        wavs, flags = [], []
        for p in paths:
            try:
                wavs.append(load_audio_mono(p, self.sr)); flags.append(True)
            except Exception:
                wavs.append(np.zeros(self.sr, dtype=np.float32)); flags.append(False)
        inputs = self.processor(
            wavs, sampling_rate=self.sr, return_tensors="pt",
            padding=True, truncation=True,
        ).to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        pooled = _l2(hs.mean(dim=1))      # mean-pool the AST patch tokens
        return pooled, flags


# ============================================================
# AudioMAE: timm ViT + torchaudio kaldi mel (fixed 1024 frames x 128 mel)
# ============================================================
class AudioMAEEncoder(BaseEncoder):
    name = "audiomae"
    WIN_SEC = 10.0
    N_FRAMES = 1024
    N_MELS = 128
    MEAN = -4.2677393
    STD = 4.5689974

    def load(self):
        import timm
        self.model = timm.create_model(f"hf_hub:{self.hf_id}", pretrained=True)
        self.model.eval().to(self.device)
        return self

    def _melspec(self, wav_16k_tensor):
        # Force CPU: torchaudio.compliance.kaldi.fbank runs on GPU by default and a 10 s clip
        # accumulates intermediate buffers that can trigger a Windows TDR (2 s GPU hang reset) or OOM.
        # The mel spectrogram itself is small (1024 x 128), so CPU computation costs almost nothing.
        from torchaudio.compliance import kaldi
        wav_cpu = wav_16k_tensor.detach().cpu().unsqueeze(0)
        melspec = kaldi.fbank(
            wav_cpu, htk_compat=True,
            window_type="hanning", num_mel_bins=self.N_MELS,
        )
        if melspec.shape[0] < self.N_FRAMES:
            melspec = F.pad(melspec, (0, 0, 0, self.N_FRAMES - melspec.shape[0]))
        else:
            melspec = melspec[:self.N_FRAMES]
        melspec = (melspec - self.MEAN) / (self.STD * 2)
        return melspec.view(1, 1, self.N_FRAMES, self.N_MELS)

    @torch.no_grad()
    def encode_batch(self, paths):
        specs, flags = [], []
        for p in paths:
            try:
                wav = load_audio_mono(p, self.sr)
                specs.append(self._melspec(torch.from_numpy(wav)))
                flags.append(True)
            except Exception:
                specs.append(torch.zeros(1, 1, self.N_FRAMES, self.N_MELS))
                flags.append(False)
        x = torch.cat(specs, dim=0).to(self.device)
        feats = self.model.forward_features(x)     # (B, 1+512, 768)
        feats = feats[:, 1:, :]                    # drop the CLS token
        # 2D patches: (B,64,8,768) -> mean over the mel dimension -> (B,64,768) -> mean over time
        B = feats.shape[0]
        feats = feats.unflatten(1, (self.N_FRAMES // 16, self.N_MELS // 16))
        pooled = feats.mean(dim=2).mean(dim=1)     # (B,768)
        pooled = _l2(pooled)
        return pooled, flags


# ============================================================
# Factory
# ============================================================
_REGISTRY = {
    "beats": BEATsEncoder,
    "mert": MERTEncoder,
    "ast": ASTEncoder,
    "audiomae": AudioMAEEncoder,
}


def create_encoder(name, model_path=None, device=None):
    if name not in _REGISTRY:
        raise ValueError(f"unknown encoder {name}; choose from {list(_REGISTRY)}")
    enc = _REGISTRY[name](model_path=model_path, device=device)
    enc.load()
    return enc


if __name__ == "__main__":
    import sys
    print("encoder configs:")
    for k, v in ENCODER_CONFIGS.items():
        src = _default_beats_source() if k == "beats" else v["hf_id"]
        print(f"  {k:9s} sr={v['sr']:5d} dim={v['dim']} bs={v['default_bs']}  {src}")
    if len(sys.argv) > 1 and sys.argv[1] == "loadtest":
        name = sys.argv[2] if len(sys.argv) > 2 else "ast"
        enc = create_encoder(name)
        print("loaded", name, "on", enc.device)
