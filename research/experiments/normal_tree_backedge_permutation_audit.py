from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_gym.data.loaders import load_dataset
from gnn_gym.evaluation.metrics import average_precision
from gnn_gym.models import build_model
from gnn_gym.registry import ensure_registrations
from gnn_gym.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-relabels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_registrations()
    config = load_config(args.run_dir)
    device = get_device(args.device)
    dataset_name = str(config["dataset"]["name"])
    model_name = str(config["model"]["name"])
    bundle = load_dataset(dataset_name, config)
    model = build_model(
        name=model_name,
        in_channels=bundle.num_features,
        out_channels=bundle.num_outputs,
        task=bundle.task,
        config=config,
    ).to(device)
    checkpoint = torch.load(args.run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_graphs = select_split(bundle.dataset, bundle.split_idx, "test")
    original_logits, labels = predict_graphs(model, test_graphs, args.batch_size, device)
    original_ap = average_precision(original_logits, labels)
    prevalence = float(labels.float().mean().item())

    generator = torch.Generator().manual_seed(args.seed)
    relabeled_probs: list[torch.Tensor] = []
    relabeled_aps: list[float] = []
    for _ in range(args.num_relabels):
        relabeled = [randomly_relabel_graph(graph, generator) for graph in test_graphs]
        logits, relabel_labels = predict_graphs(model, relabeled, args.batch_size, device)
        if not torch.equal(labels.view(-1), relabel_labels.view(-1)):
            raise RuntimeError("Relabeling changed graph label order")
        relabeled_probs.append(torch.sigmoid(logits.view(-1)))
        relabeled_aps.append(average_precision(logits, relabel_labels))

    stacked_probs = torch.stack(relabeled_probs, dim=0)
    result = {
        "run_dir": str(args.run_dir),
        "num_graphs": len(test_graphs),
        "num_relabels": args.num_relabels,
        "class_prevalence_random_ap": prevalence,
        "original_ap": original_ap,
        "relabeled_ap_mean": float(torch.tensor(relabeled_aps).mean().item()),
        "relabeled_ap_std": float(torch.tensor(relabeled_aps).std(unbiased=False).item()),
        "mean_prediction_variance": float(stacked_probs.var(dim=0, unbiased=False).mean().item()),
        "max_prediction_range": float(
            (stacked_probs.max(dim=0).values - stacked_probs.min(dim=0).values).max().item()
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def load_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_split(dataset: Any, split_idx: dict[str, Any], split: str) -> list[Data]:
    indices = split_idx[split]
    if split == "valid" and indices is None:
        indices = split_idx["val"]
    return [dataset[int(idx)] for idx in indices]


@torch.no_grad()
def predict_graphs(
    model: torch.nn.Module,
    graphs: list[Data],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        pred = model(
            batch.x.float(),
            batch.edge_index,
            batch=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
        )
        preds.append(pred.detach().cpu())
        labels.append(batch.y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def randomly_relabel_graph(graph: Data, generator: torch.Generator) -> Data:
    num_nodes = int(graph.num_nodes)
    perm = torch.randperm(num_nodes, generator=generator)
    relabeled = graph.clone()
    relabeled.x = graph.x[torch.argsort(perm)]
    relabeled.edge_index = perm[graph.edge_index]
    return relabeled


if __name__ == "__main__":
    main()
