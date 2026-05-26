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
from gnn_gym.models import build_model
from gnn_gym.registry import ensure_registrations
from gnn_gym.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_registrations()
    config = load_config(args.run_dir)
    device = get_device(args.device)
    bundle = load_dataset(str(config["dataset"]["name"]), config)
    model = build_model(
        name=str(config["model"]["name"]),
        in_channels=bundle.num_features,
        out_channels=bundle.num_outputs,
        task=bundle.task,
        config=config,
    ).to(device)
    checkpoint = torch.load(args.run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graphs = select_split(bundle.dataset, bundle.split_idx, args.split)
    layer_weights: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=args.batch_size, shuffle=False):
            batch = batch.to(device)
            model(
                batch.x.float(),
                batch.edge_index,
                batch=batch.batch,
                edge_attr=getattr(batch, "edge_attr", None),
            )
            for layer_idx, weights in enumerate(model.encoder.last_gate_weights):
                layer_weights.setdefault(layer_idx, []).append(weights)

    result = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "num_graphs": len(graphs),
        "layers": summarize_layers(layer_weights),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def load_config(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_split(dataset: Any, split_idx: dict[str, Any], split: str) -> list[Data]:
    key = "valid" if split == "val" else split
    indices = split_idx[key]
    if key == "valid" and indices is None:
        indices = split_idx["val"]
    return [dataset[int(idx)] for idx in indices]


def summarize_layers(layer_weights: dict[int, list[torch.Tensor]]) -> dict[str, dict[str, Any]]:
    summaries = {}
    for layer_idx, values in sorted(layer_weights.items()):
        weights = torch.cat(values, dim=0)
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=1)
        summaries[str(layer_idx)] = {
            "mean_weights": [float(value) for value in weights.mean(dim=0).tolist()],
            "std_weights": [float(value) for value in weights.std(dim=0, unbiased=False).tolist()],
            "mean_entropy": float(entropy.mean().item()),
            "max_entropy": float(torch.log(torch.tensor(float(weights.size(1)))).item()),
        }
    return summaries


if __name__ == "__main__":
    main()
