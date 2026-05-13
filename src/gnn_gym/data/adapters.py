from dataclasses import dataclass
from typing import Any

from torch_geometric.data import Data


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: str
    metric: str
    trainer: str
    data: Data
    num_features: int
    num_outputs: int
    evaluator: str = "accuracy"
    metadata: dict[str, Any] | None = None
