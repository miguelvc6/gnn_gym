from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
