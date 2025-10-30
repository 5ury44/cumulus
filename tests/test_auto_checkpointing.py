#!/usr/bin/env python3
"""
Smoke tests that exercise the automatic checkpointing hooks across all supported
frameworks. Each test submits a small training job through the Cumulus client
and verifies that the AutoCheckpointManager produced a checkpoint without any
explicit save calls inside the training code.
"""

import os
from typing import Dict, Any

from sdk import CumulusClient


TEST_SERVER_URL = os.getenv("CUMULUS_TEST_SERVER", "http://localhost:8084")


def _client() -> CumulusClient:
    return CumulusClient(TEST_SERVER_URL)


def _assert_checkpoint(result: Dict[str, Any], expected_extension: str) -> None:
    assert result["status"] == "ok", f"Job failed: {result}"
    checkpoint = result.get("checkpoint")
    assert checkpoint, f"Missing checkpoint info: {result}"
    assert checkpoint.get("checkpoint_id"), "Checkpoint missing identifier"
    local_path = checkpoint.get("local_path")
    assert local_path, "Checkpoint missing local path"
    if expected_extension:
        assert local_path.endswith(expected_extension), (
            f"Expected {expected_extension} checkpoint, got {local_path}"
        )
    assert result.get("path_exists"), f"Checkpoint path not found: {local_path}"


def auto_checkpoint_pytorch_job():
    import os
    import torch
    import runtime

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for step in range(120):
        inputs = torch.randn(64, 16)
        targets = torch.randn(64, 8)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    return {"status": "ok", "framework": "pytorch", "checkpoint": checkpoint, "path_exists": exists}


def auto_checkpoint_tensorflow_job():
    import os
    import numpy as np
    import tensorflow as tf
    import runtime

    tf.random.set_seed(42)
    np.random.seed(42)

    x = np.random.randn(1024, 12).astype("float32")
    y = np.random.randint(0, 2, size=(1024, 1)).astype("float32")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation="relu", input_shape=(12,)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x, y, batch_size=8, epochs=2, verbose=0)

    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    return {"status": "ok", "framework": "tensorflow", "checkpoint": checkpoint, "path_exists": exists}


def auto_checkpoint_sklearn_job():
    import os
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    import runtime

    np.random.seed(7)
    x = np.random.randn(200, 10)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    model = LogisticRegression(max_iter=100)
    model.fit(x, y)

    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    return {"status": "ok", "framework": "sklearn", "checkpoint": checkpoint, "path_exists": exists}


def auto_checkpoint_xgboost_job():
    import os
    import numpy as np
    import xgboost as xgb
    import runtime

    np.random.seed(13)
    x = np.random.randn(256, 12)
    y = (x[:, 0] > 0).astype(int)
    dtrain = xgb.DMatrix(x, label=y)

    params = {"objective": "binary:logistic", "tree_method": "hist"}
    xgb.train(params, dtrain, num_boost_round=16)

    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    return {"status": "ok", "framework": "xgboost", "checkpoint": checkpoint, "path_exists": exists}


def auto_checkpoint_lightgbm_job():
    import os
    import numpy as np
    import lightgbm as lgb
    import runtime

    np.random.seed(29)
    x = np.random.randn(256, 12)
    y = (x[:, 0] > 0).astype(int)

    dataset = lgb.Dataset(x, label=y, free_raw_data=True)
    params = {"objective": "binary", "verbosity": -1}
    lgb.train(params, dataset, num_boost_round=16)

    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    return {"status": "ok", "framework": "lightgbm", "checkpoint": checkpoint, "path_exists": exists}


def test_auto_checkpoint_pytorch():
    client = _client()
    result = client.run(
        func=auto_checkpoint_pytorch_job,
        gpu_memory=0.3,
        duration=600,
        requirements=["torch"],
    )
    _assert_checkpoint(result, ".pt")


def test_auto_checkpoint_tensorflow():
    client = _client()
    result = client.run(
        func=auto_checkpoint_tensorflow_job,
        gpu_memory=0.4,
        duration=600,
        requirements=["tensorflow"],
    )
    _assert_checkpoint(result, ".weights.h5")


def test_auto_checkpoint_sklearn():
    client = _client()
    result = client.run(
        func=auto_checkpoint_sklearn_job,
        gpu_memory=0.1,
        duration=300,
        requirements=["scikit-learn"],
    )
    _assert_checkpoint(result, ".pkl")


def test_auto_checkpoint_xgboost():
    client = _client()
    result = client.run(
        func=auto_checkpoint_xgboost_job,
        gpu_memory=0.3,
        duration=600,
        requirements=["xgboost"],
    )
    _assert_checkpoint(result, ".json")


def test_auto_checkpoint_lightgbm():
    client = _client()
    result = client.run(
        func=auto_checkpoint_lightgbm_job,
        gpu_memory=0.3,
        duration=600,
        requirements=["lightgbm"],
    )
    _assert_checkpoint(result, ".txt")


