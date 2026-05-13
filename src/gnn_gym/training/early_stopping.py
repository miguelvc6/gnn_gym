from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    higher_is_better: bool = True
    best: float | None = None
    best_epoch: int = 0

    def update(self, value: float, epoch: int) -> bool:
        if self.best is None:
            improved = True
        elif self.higher_is_better:
            improved = value > self.best
        else:
            improved = value < self.best
        if improved:
            self.best = value
            self.best_epoch = epoch
        return improved

    def should_stop(self, epoch: int) -> bool:
        return epoch - self.best_epoch >= self.patience
