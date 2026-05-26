import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def config_hash(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def architecture_config_hash(config: Mapping[str, object]) -> str:
    """Hash a resolved architecture/training config independent of seed and run metadata."""
    payload = _strip_run_specific_fields(deepcopy(dict(config)))
    return config_hash(payload)


def _strip_run_specific_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "architecture_config_hash",
                "config_hash",
                "run_id",
                "device",
                "git_commit",
                "status",
                "package_versions",
                "best_epoch",
                "peak_gpu_memory_mb",
                "predictions_path",
            }:
                continue
            if key == "seed":
                continue
            cleaned[key] = _strip_run_specific_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_run_specific_fields(item) for item in value]
    return value
