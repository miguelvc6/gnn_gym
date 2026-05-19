from typing import Any

import torch


class OgbEvaluatorAdapter:
    def __init__(self, name: str) -> None:
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        from ogb.nodeproppred import Evaluator as NodeEvaluator

        if name.startswith("ogbn-"):
            self.evaluator = NodeEvaluator(name)
        elif name.startswith("ogbg-"):
            self.evaluator = GraphEvaluator(name)
        elif name.startswith("ogbl-"):
            self.evaluator = LinkEvaluator(name)
        else:
            raise ValueError(f"Unsupported OGB dataset: {name}")

    def eval(self, payload: dict[str, Any]) -> dict[str, float]:
        normalized = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in payload.items()
        }
        return self.evaluator.eval(normalized)
