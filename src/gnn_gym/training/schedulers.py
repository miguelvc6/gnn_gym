from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR


def build_scheduler(optimizer: Optimizer, config: dict[str, object]) -> LRScheduler | None:
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    name = str(training.get("scheduler", "none")).lower()
    if name in {"none", "null"}:
        return None
    if name == "cosine":
        max_epochs = int(training.get("max_epochs", 100))
        return CosineAnnealingLR(optimizer, T_max=max(max_epochs, 1))
    if name == "step":
        step_size = int(training.get("step_size", 50))
        gamma = float(training.get("gamma", 0.5))
        return StepLR(optimizer, step_size=max(step_size, 1), gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {name}")
