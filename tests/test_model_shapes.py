import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from gnn_gym.data.adapters import DatasetBundle
from gnn_gym.models import build_model
from gnn_gym.models.base import NodeModel, TaskModel
from gnn_gym.training.graph_trainer import GraphPredictionTrainer


@pytest.mark.parametrize(
    "model_name",
    [
        "mlp",
        "gcn",
        "gat",
        "gin",
        "bethe_gnn",
        "confidence_appnp_net",
        "decimation_gnn",
        "dual_primal_gnn",
        "equilibrium_belief_gnn",
        "revision_gnn",
        "res_gin",
        "jk_gcn",
        "appnp_net",
        "gcn2_net",
        "gpr_gnn",
        "kikuchi_gnn",
        "loop_corrected_gnn",
        "gatv2",
        "gated_appnp_net",
        "cavity_gnn",
        "frustration_gnn",
        "nb_appnp_net",
        "nb_belief_gnn",
        "nb_light_gnn",
        "normal_tree_backedge_gnn",
        "tree_pack_gnn",
        "res_appnp_net",
        "region_collapse_gnn",
        "sep_bottleneck_gnn",
        "entropy_gated_gnn",
        "rign_gnn",
        "temp_ladder_gnn",
        "survey_gnn",
        "walk_belief_transformer",
    ],
)
def test_model_output_shape(model_name: str) -> None:
    x = torch.randn(12, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
        ],
        dtype=torch.long,
    )
    config = {
        "model": {
            "name": model_name,
            "hidden_channels": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "activation": "elu" if model_name in {"gat", "gatv2"} else "relu",
            "norm": "none",
            "heads": 2,
            "attention_dropout": 0.0,
            "temperatures": [0.5, 1.0],
            "num_particles": 2,
            "residual_weight": 0.5,
            "nb_steps": 2,
            "propagation_steps": 3,
            "alpha": 0.1,
            "gate_init": -4.0,
            "max_steps": 2,
            "num_regions": 4,
            "max_triangles": 8,
            "num_rounds": 2,
            "walk_length": 3,
            "gate_hidden_channels": 4,
            "separator_residual_init": 0.001,
            "separator_token_init": 0.001,
            "separator_max_scale": 0.1,
            "num_tree_orders": 2,
            "num_trees": 4,
            "tree_start_idx": 0,
            "tree_pooling": "gated",
            "use_graph_channel": True,
            "use_tree_channel": True,
        }
    }
    model = build_model(
        model_name,
        in_channels=5,
        out_channels=3,
        task="node_classification",
        config=config,
    )
    model.eval()

    logits = model(x, edge_index)

    assert logits.shape == (12, 3)


def test_mean_max_add_graph_pooling_shape() -> None:
    x = torch.randn(12, 5)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    config = {
        "model": {
            "name": "gcn",
            "hidden_channels": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "activation": "relu",
            "norm": "none",
            "pooling": "mean_max_add",
            "head_hidden_channels": 8,
        }
    }
    model = build_model(
        "gcn",
        in_channels=5,
        out_channels=2,
        task="graph_binary_classification",
        config=config,
    )
    model.eval()

    logits = model(x, edge_index, batch)

    assert logits.shape == (2, 2)


class EdgeAttrRecorder(NodeModel):
    def __init__(self) -> None:
        super().__init__()
        self.output_channels = 4
        self.linear = nn.Linear(5, 4)
        self.saw_edge_attr = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.saw_edge_attr = edge_attr is not None
        return self.linear(x)


def test_graph_trainer_passes_edge_attr(tmp_path) -> None:
    graphs = []
    for idx in range(6):
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(edge_index.size(1), 3)
        y = torch.tensor([[float(idx % 2)]])
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    dataset = DatasetBundle(
        name="edge-attr-toy-graph",
        task="graph_binary_classification",
        metric="average_precision",
        trainer="graph_prediction",
        evaluator="average_precision",
        dataset={"train": graphs[:4], "val": graphs[4:5], "test": graphs[5:]},
        num_features=5,
        num_outputs=1,
    )
    encoder = EdgeAttrRecorder()
    model = TaskModel(encoder, out_channels=1, task="graph_binary_classification")
    trainer = GraphPredictionTrainer(
        model=model,
        dataset=dataset,
        config={
            "model": {"name": "edge_attr_recorder"},
            "training": {"seed": 0, "lr": 0.01, "weight_decay": 0.0, "scheduler": "none"},
            "trainer": {"name": "graph_prediction", "batch_size": 2},
        },
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )

    trainer.train_epoch(1)

    assert encoder.saw_edge_attr
