from dataclasses import dataclass
from typing import Any

from torch_geometric.data import Data


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: str
    metric: str
    trainer: str
    num_features: int
    num_outputs: int
    data: Data | None = None
    dataset: Any | None = None
    split_idx: Any | None = None
    evaluator: str = "accuracy"
    higher_is_better: bool = True
    metadata: dict[str, Any] | None = None
