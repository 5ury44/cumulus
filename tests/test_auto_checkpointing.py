#!/usr/bin/env python3
"""
Smoke tests that exercise the automatic checkpointing hooks across all supported
frameworks. Each test submits a small training job through the Cumulus client
and verifies that the AutoCheckpointManager produced a checkpoint without any
explicit save calls inside the training code.
"""

import os
from typing import Dict, Any, TYPE_CHECKING

from sdk import CumulusClient

if TYPE_CHECKING:  # pragma: no cover - import hints for linters/mypy
    import torch  # type: ignore[import]
    import tensorflow  # type: ignore[import]
    import xgboost  # type: ignore[import]
    import lightgbm  # type: ignore[import]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import]
    import runtime  # type: ignore[import]


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
    import runtime  # type: ignore[import]
    runtime.get_auto_checkpoint_manager()
    os.environ.setdefault("CUMULUS_CHECKPOINT_EVERY_STEPS", "20")
    import torch  # type: ignore[import]

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(200):
        batch = torch.randn(32, 32)
        targets = torch.randn(32, 4)
        logits = model(batch)
        loss = torch.nn.functional.mse_loss(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    manager = runtime.get_auto_checkpoint_manager()
    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    optimizer_states = []
    for optimizer_state in manager._torch_optimizer_state.values():
        optimizer_states.append(
            {
                "step": optimizer_state.get("step"),
                "has_model": bool(optimizer_state.get("model")),
            }
        )

    debug = {
        "torch_hook": getattr(torch.optim.Optimizer, "_cumulus_auto_ckpt", False),
        "optimizer_states": optimizer_states,
        "framework_status": manager.get_framework_status(),
        "manager_enabled": manager.enabled,
    }
    return {
        "status": "ok",
        "framework": "pytorch",
        "checkpoint": checkpoint,
        "path_exists": exists,
        "debug": debug,
    }


def auto_checkpoint_tensorflow_job():
    import os
    import numpy as np  # type: ignore[import]
    import tensorflow as tf  # type: ignore[import]
    import runtime  # type: ignore[import]

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
    import runtime  # type: ignore[import]
    os.environ.setdefault("CUMULUS_CHECKPOINT_EVERY_STEPS", "5")
    import numpy as np  # type: ignore[import]
    from sklearn.linear_model import SGDClassifier  # type: ignore[import]

    np.random.seed(7)
    x = np.random.randn(256, 10)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    model = SGDClassifier(loss="log_loss", max_iter=5, warm_start=True)
    for _ in range(120):
        model.fit(x, y)

    manager = runtime.get_auto_checkpoint_manager()
    checkpoint = runtime.get_last_checkpoint_info()
    exists = bool(checkpoint and os.path.exists(checkpoint["local_path"]))
    debug = manager.framework_state.get("sklearn", {}).copy()
    debug["framework_status"] = manager.get_framework_status()
    debug["manager_enabled"] = manager.enabled
    debug["auto_enabled"] = manager.enabled
    return {
        "status": "ok",
        "framework": "sklearn",
        "checkpoint": checkpoint,
        "path_exists": exists,
        "debug": debug,
    }


def auto_checkpoint_xgboost_job():
    import os
    import numpy as np  # type: ignore[import]
    import xgboost as xgb  # type: ignore[import]
    import runtime  # type: ignore[import]

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
    import numpy as np  # type: ignore[import]
    import lightgbm as lgb  # type: ignore[import]
    import runtime  # type: ignore[import]

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
        requirements=["torch", "boto3"],
    )
    _assert_checkpoint(result, ".pt")


def test_auto_checkpoint_tensorflow():
    client = _client()
    result = client.run(
        func=auto_checkpoint_tensorflow_job,
        gpu_memory=0.4,
        duration=600,
        requirements=["tensorflow", "boto3"],
    )
    _assert_checkpoint(result, ".weights.h5")


def test_auto_checkpoint_sklearn():
    client = _client()
    result = client.run(
        func=auto_checkpoint_sklearn_job,
        gpu_memory=0.1,
        duration=300,
        requirements=["scikit-learn", "boto3"],
    )
    _assert_checkpoint(result, ".pkl")


def test_auto_checkpoint_xgboost():
    client = _client()
    result = client.run(
        func=auto_checkpoint_xgboost_job,
        gpu_memory=0.3,
        duration=600,
        requirements=["xgboost", "boto3"],
    )
    _assert_checkpoint(result, ".json")


def test_auto_checkpoint_lightgbm():
    client = _client()
    result = client.run(
        func=auto_checkpoint_lightgbm_job,
        gpu_memory=0.3,
        duration=600,
        requirements=["lightgbm", "boto3"],
    )
    _assert_checkpoint(result, ".txt")


if __name__ == "__main__":
    TESTS = [
        ("pytorch", test_auto_checkpoint_pytorch),
        ("tensorflow", test_auto_checkpoint_tensorflow),
        ("sklearn", test_auto_checkpoint_sklearn),
        ("xgboost", test_auto_checkpoint_xgboost),
        ("lightgbm", test_auto_checkpoint_lightgbm),
    ]

    failures = []
    for name, test_func in TESTS:
        try:
            print(f"Running auto checkpoint smoke test for {name}...")
            test_func()
            print(f"✅ {name} checkpointing passed")
        except AssertionError as exc:
            print(f"❌ {name} checkpointing failed: {exc}")
            failures.append(name)
        except Exception as exc:  # pragma: no cover - debugging aid
            print(f"❌ {name} checkpointing error: {exc}")
            failures.append(name)

    if failures:
        raise SystemExit(f"Auto-checkpoint tests failed for: {', '.join(failures)}")

    print("🎉 All auto-checkpoint smoke tests passed")

