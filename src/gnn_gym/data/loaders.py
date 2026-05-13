from gnn_gym.data.adapters import DatasetBundle
from gnn_gym.registry import DATASET_REGISTRY, ensure_registrations


def load_dataset(name: str, config: dict[str, object]) -> DatasetBundle:
    ensure_registrations()
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](config)  # type: ignore[return-value]
