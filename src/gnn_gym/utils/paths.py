from datetime import UTC, datetime
from pathlib import Path


def make_run_dir(
    root: str | Path,
    model: str,
    dataset: str,
    seed: int,
    config_hash: str,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}__{model}__{dataset}__seed-{seed}__{config_hash}"
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path
