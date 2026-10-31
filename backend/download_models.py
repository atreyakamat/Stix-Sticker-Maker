"""
Pre-download segmentation models into the Hugging Face cache.

Run this to warm the cache so the first real request isn't slowed by a
multi-hundred-MB download. In Docker this is typically run once against the
mounted model-cache volume:

    docker compose run --rm backend python download_models.py
    # or a subset:
    docker compose run --rm backend python download_models.py birefnet birefnet_lite

With no arguments it downloads every model referenced by the configured
quality tiers (plus the startup model).
"""

import sys

from transformers import AutoModelForImageSegmentation

from config import QUALITY_TIERS, MODEL_NAME
from model_loader import MODEL_REGISTRY


def models_for_tiers() -> list[str]:
    keys = {MODEL_NAME}
    for tier in QUALITY_TIERS.values():
        if tier.get("model"):
            keys.add(tier["model"])
    return [k for k in keys if k in MODEL_REGISTRY]


def main(argv: list[str]) -> int:
    keys = argv[1:] or models_for_tiers()
    print(f"Pre-downloading {len(keys)} model(s): {', '.join(keys)}")

    failures = []
    for key in keys:
        entry = MODEL_REGISTRY.get(key)
        if not entry:
            print(f"  ⚠️  Unknown model key '{key}', skipping.")
            continue
        repo = entry["repo"]
        print(f"  ⬇️  {key} -> {repo}")
        try:
            AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
            print(f"  ✅  {key} cached.")
        except Exception as e:  # noqa: BLE001
            print(f"  ❌  {key} failed: {e}")
            failures.append(key)

    if failures:
        print(f"\nDone with errors. Failed: {', '.join(failures)}")
        return 1
    print("\nAll requested models are cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
