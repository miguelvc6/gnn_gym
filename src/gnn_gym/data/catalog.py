import torch
from torch_geometric.data import Data

from gnn_gym.data.adapters import DatasetBundle
from gnn_gym.registry import register_dataset


@register_dataset("toy-node")
def load_toy_node_dataset(config: dict[str, object] | None = None) -> DatasetBundle:
    dataset_config = (config or {}).get("dataset", {}) if config else {}
    num_nodes = int(dataset_config.get("num_nodes", 48))  # type: ignore[union-attr]
    num_features = int(dataset_config.get("num_features", 8))  # type: ignore[union-attr]
    num_classes = int(dataset_config.get("num_classes", 3))  # type: ignore[union-attr]

    generator = torch.Generator().manual_seed(12345)
    x = torch.randn((num_nodes, num_features), generator=generator)
    weights = torch.randn((num_features, num_classes), generator=generator)
    y = (x @ weights).argmax(dim=1)

    source = torch.arange(num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
    skip = (source + 3) % num_nodes
    edge_index = torch.stack(
        [
            torch.cat([source, target, source, skip]),
            torch.cat([target, source, skip, source]),
        ],
        dim=0,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes) : int(0.8 * num_nodes)] = True
    test_mask[int(0.8 * num_nodes) :] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return DatasetBundle(
        name="toy-node",
        task="node_classification",
        metric="accuracy",
        trainer="full_batch_node",
        evaluator="accuracy",
        data=data,
        num_features=num_features,
        num_outputs=num_classes,
    )
