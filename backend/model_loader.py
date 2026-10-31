"""
Model Loader — Multi-Model Manager
===================================
Loads segmentation models lazily and keeps a small LRU cache of them resident
in memory. Each quality tier can request a different model/resolution and the
manager will load it on first use and reuse it afterwards.

Key improvements over the previous single-global design:
  * A real registry of models that exist on the Hugging Face Hub (the old
    "birefnet-general" repo 404s and was silently falling back).
  * Per-model input normalization. BiRefNet wants ImageNet stats; RMBG-1.4
    wants mean 0.5 / std 1.0. Using the wrong stats quietly degrades results.
  * Configurable device: auto / cpu / cuda / dml (DirectML for AMD/Intel GPUs
    on native Windows via the optional `torch-directml` package).
  * An LRU cache so switching tiers doesn't thrash, but RAM stays bounded.
"""

import sys
import threading
from collections import OrderedDict

import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import MODEL_NAME, MODEL_RESOLUTION, DEVICE, MODEL_CACHE_SIZE, \
    QUALITY_TIERS, DEFAULT_QUALITY_TIER

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# norm: which normalization the model expects.
#   "imagenet" -> mean [0.485,0.456,0.406], std [0.229,0.224,0.225]  (BiRefNet)
#   "half"     -> mean [0.5,0.5,0.5],       std [1.0,1.0,1.0]        (RMBG-1.4)
# default_res: a sensible inference resolution if a tier doesn't override it.
MODEL_REGISTRY = {
    "birefnet":            {"repo": "ZhengPeng7/BiRefNet",           "norm": "imagenet", "default_res": 1024},
    "birefnet_lite":       {"repo": "ZhengPeng7/BiRefNet_lite",      "norm": "imagenet", "default_res": 1024},
    "birefnet_hr":         {"repo": "ZhengPeng7/BiRefNet_HR",        "norm": "imagenet", "default_res": 2048},
    "birefnet_matting":    {"repo": "ZhengPeng7/BiRefNet-matting",   "norm": "imagenet", "default_res": 1024},
    "birefnet_hr_matting": {"repo": "ZhengPeng7/BiRefNet_HR-matting", "norm": "imagenet", "default_res": 2048},
    "birefnet_portrait":   {"repo": "ZhengPeng7/BiRefNet-portrait",  "norm": "imagenet", "default_res": 1024},
    "rmbg14":              {"repo": "briaai/RMBG-1.4",               "norm": "half",     "default_res": 1024},
    # Backwards-compatible aliases (old config/env values keep working)
    "rmbg-1.4":            {"repo": "briaai/RMBG-1.4",               "norm": "half",     "default_res": 1024},
}

_NORM_STATS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "half":     ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
}

PRIMARY_FALLBACK = "birefnet"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def _resolve_device() -> torch.device:
    """Pick the compute device based on config.DEVICE and what's available."""
    pref = DEVICE

    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    # ROCm builds of torch expose CUDA-style APIs
    if pref in ("auto", "cuda") and hasattr(torch, "hip") and getattr(torch, "hip", None) \
            and torch.hip.is_available():
        return torch.device("cuda")

    if pref in ("auto", "dml"):
        try:
            import torch_directml  # optional, Windows only
            dev = torch_directml.device()
            print("🔧 DirectML device available (AMD/Intel GPU acceleration).")
            return dev
        except Exception:
            if pref == "dml":
                print("⚠️ STIX_DEVICE=dml but torch-directml is not installed; falling back to CPU.")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------
class ModelManager:
    """Thread-safe lazy loader + LRU cache of segmentation models."""

    def __init__(self, cache_size: int = 2):
        self._cache_size = max(1, cache_size)
        self._models: "OrderedDict[str, torch.nn.Module]" = OrderedDict()
        self._lock = threading.Lock()
        self._device = _resolve_device()
        self._startup_model = None   # key loaded at startup / current default
        self._startup_res = None

    # -- device -----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    # -- loading ----------------------------------------------------------
    def _load(self, key: str) -> torch.nn.Module:
        entry = MODEL_REGISTRY.get(key)
        if entry is None:
            raise KeyError(f"Unknown model '{key}'. Known: {list(MODEL_REGISTRY)}")

        repo_id = entry["repo"]
        print(f"📦 Loading model '{key}' from {repo_id} ...")
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                repo_id, trust_remote_code=True
            )
        except Exception as e:
            print(f"⚠️ Failed to load '{key}' ({repo_id}): {e}")
            if key != PRIMARY_FALLBACK:
                print(f"🔄 Falling back to '{PRIMARY_FALLBACK}'...")
                return self._load(PRIMARY_FALLBACK)
            raise

        model.to(self._device)
        # FP32 everywhere on CPU to avoid float16/float32 mismatch errors.
        if self._device.type == "cpu":
            model.float()
        model.eval()
        return model

    def get(self, key: str | None = None) -> tuple[torch.nn.Module, dict]:
        """Return (model, registry_entry), loading + caching as needed."""
        key = key or self._startup_model or MODEL_NAME
        if key not in MODEL_REGISTRY:
            print(f"⚠️ Unknown model '{key}', using '{PRIMARY_FALLBACK}'.")
            key = PRIMARY_FALLBACK

        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key], MODEL_REGISTRY[key]

        # Load outside the lock (can be slow / download), then insert.
        model = self._load(key)
        with self._lock:
            self._models[key] = model
            self._models.move_to_end(key)
            while len(self._models) > self._cache_size:
                evicted, _ = self._models.popitem(last=False)
                print(f"🗑️ Evicting cached model '{evicted}' (cache size {self._cache_size}).")
            return model, MODEL_REGISTRY[key]

    # -- transforms -------------------------------------------------------
    def build_transform(self, resolution: int, norm: str = "imagenet"):
        mean, std = _NORM_STATS.get(norm, _NORM_STATS["imagenet"])
        return transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # -- startup / info ---------------------------------------------------
    def warmup(self, key: str | None = None, resolution: int | None = None):
        key = key or MODEL_NAME
        self._startup_model = key
        self._startup_res = resolution or MODEL_RESOLUTION
        print(f"🔧 Device: {self._device}")
        self.get(key)
        return self

    def info(self) -> dict:
        with self._lock:
            loaded = list(self._models.keys())
        return {
            "name": self._startup_model or MODEL_NAME,
            "resolution": self._startup_res or MODEL_RESOLUTION,
            "device": str(self._device),
            "loaded_models": loaded,
            "cache_size": self._cache_size,
            "available_models": [k for k in MODEL_REGISTRY if k != "rmbg-1.4"],
            "quality_tiers": QUALITY_TIERS,
            "default_tier": DEFAULT_QUALITY_TIER,
        }


# ---------------------------------------------------------------------------
# Global instance + backwards-compatible module API
# ---------------------------------------------------------------------------
_manager = ModelManager(cache_size=MODEL_CACHE_SIZE)


def load_model(model_key: str | None = None, resolution: int | None = None):
    """Eagerly load the startup/default model. Called once at startup."""
    _manager.warmup(model_key, resolution)
    return _manager.get(model_key)[0]


def get_model(model_key: str | None = None):
    """Return the model for `model_key` (or the default), loading if needed."""
    return _manager.get(model_key)[0]


def get_model_and_meta(model_key: str | None = None):
    """Return (model, registry_entry) for the given key."""
    return _manager.get(model_key)


def get_transform(resolution: int | None = None, norm: str = "imagenet"):
    """Build an input transform. Defaults to the startup resolution."""
    res = resolution or _manager._startup_res or MODEL_RESOLUTION
    return _manager.build_transform(res, norm)


def get_device_instance():
    return _manager.device


def get_model_info() -> dict:
    return _manager.info()
