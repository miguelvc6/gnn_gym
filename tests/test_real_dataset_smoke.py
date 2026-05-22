import os
from pathlib import Path

import pytest

from gnn_gym.data.loaders import load_dataset
from gnn_gym.utils.config import load_run_config

pytestmark = pytest.mark.skipif(
    os.environ.get("GNN_GYM_RUN_REAL_DATASET_TESTS") != "1",
    reason="set GNN_GYM_RUN_REAL_DATASET_TESTS=1 to run downloading dataset smoke tests",
)


@pytest.mark.parametrize("model,dataset", [("gcn", "cora"), ("gat", "pubmed")])
def test_planetoid_dataset_smoke(tmp_path: Path, model: str, dataset: str) -> None:
    config = load_run_config(model, dataset, [f"dataset.root={tmp_path}"])
    try:
        bundle = load_dataset(dataset, config)
    except Exception as error:
        pytest.skip(f"dataset unavailable in this environment: {error}")

    assert bundle.num_features > 0
    assert bundle.num_outputs > 0
    assert bundle.data is not None
