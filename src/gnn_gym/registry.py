from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MODEL_REGISTRY: dict[str, Callable[..., object]] = {}
DATASET_REGISTRY: dict[str, Callable[..., object]] = {}
TRAINER_REGISTRY: dict[str, Callable[..., object]] = {}
EVALUATOR_REGISTRY: dict[str, Callable[..., object]] = {}


def _register(registry: dict[str, Callable[..., object]], kind: str, name: str) -> Callable[[T], T]:
    def decorator(obj: T) -> T:
        if name in registry:
            raise ValueError(f"{kind} already registered: {name}")
        registry[name] = obj  # type: ignore[assignment]
        return obj

    return decorator


def register_model(name: str) -> Callable[[T], T]:
    return _register(MODEL_REGISTRY, "Model", name)


def register_dataset(name: str) -> Callable[[T], T]:
    return _register(DATASET_REGISTRY, "Dataset", name)


def register_trainer(name: str) -> Callable[[T], T]:
    return _register(TRAINER_REGISTRY, "Trainer", name)


def register_evaluator(name: str) -> Callable[[T], T]:
    return _register(EVALUATOR_REGISTRY, "Evaluator", name)


def ensure_registrations() -> None:
    import gnn_gym.data.catalog  # noqa: F401
    import gnn_gym.evaluation.evaluators  # noqa: F401
    import gnn_gym.models.gat  # noqa: F401
    import gnn_gym.models.gcn  # noqa: F401
    import gnn_gym.models.gin  # noqa: F401
    import gnn_gym.models.mlp  # noqa: F401
    import gnn_gym.training.graph_trainer  # noqa: F401
    import gnn_gym.training.link_trainer  # noqa: F401
    import gnn_gym.training.node_trainer  # noqa: F401
