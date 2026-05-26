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
    import gnn_gym.models.appnp_net  # noqa: F401
    import gnn_gym.models.bethe_gnn  # noqa: F401
    import gnn_gym.models.cavity_gnn  # noqa: F401
    import gnn_gym.models.confidence_appnp_net  # noqa: F401
    import gnn_gym.models.decimation_gnn  # noqa: F401
    import gnn_gym.models.dual_primal_gnn  # noqa: F401
    import gnn_gym.models.entropy_gated_gnn  # noqa: F401
    import gnn_gym.models.equilibrium_belief_gnn  # noqa: F401
    import gnn_gym.models.frustration_gnn  # noqa: F401
    import gnn_gym.models.gat  # noqa: F401
    import gnn_gym.models.gated_appnp_net  # noqa: F401
    import gnn_gym.models.gatv2  # noqa: F401
    import gnn_gym.models.gcn  # noqa: F401
    import gnn_gym.models.gcn2_net  # noqa: F401
    import gnn_gym.models.gin  # noqa: F401
    import gnn_gym.models.gpr_gnn  # noqa: F401
    import gnn_gym.models.jk_gcn  # noqa: F401
    import gnn_gym.models.kikuchi_gnn  # noqa: F401
    import gnn_gym.models.loop_corrected_gnn  # noqa: F401
    import gnn_gym.models.mlp  # noqa: F401
    import gnn_gym.models.nb_appnp_net  # noqa: F401
    import gnn_gym.models.nb_belief_gnn  # noqa: F401
    import gnn_gym.models.nb_light_gnn  # noqa: F401
    import gnn_gym.models.region_collapse_gnn  # noqa: F401
    import gnn_gym.models.res_appnp_net  # noqa: F401
    import gnn_gym.models.res_gin  # noqa: F401
    import gnn_gym.models.revision_gnn  # noqa: F401
    import gnn_gym.models.rign_gnn  # noqa: F401
    import gnn_gym.models.survey_gnn  # noqa: F401
    import gnn_gym.models.temp_ladder_gnn  # noqa: F401
    import gnn_gym.models.walk_belief_transformer  # noqa: F401
    import gnn_gym.training.graph_trainer  # noqa: F401
    import gnn_gym.training.link_trainer  # noqa: F401
    import gnn_gym.training.node_trainer  # noqa: F401
