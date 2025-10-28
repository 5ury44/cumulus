from __future__ import annotations

from typing import Any, Optional
import importlib


class BaseAdapter:
    name: str
    ext: str

    def save(self, model: Any, dst_path: str, optimizer: Any = None) -> None:
        raise NotImplementedError

    def load(self, model: Any, src_path: str, optimizer: Any = None) -> Any:
        raise NotImplementedError


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


class TorchAdapter(BaseAdapter):
    name = "pytorch"
    ext = "pt"

    def save(self, model: Any, dst_path: str, optimizer: Any = None) -> None:
        import torch
        state = {
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'rng_cpu': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, dst_path)

    def load(self, model: Any, src_path: str, optimizer: Any = None) -> Any:
        import torch
        state = torch.load(src_path, map_location='cpu')
        if model is not None and 'model' in state:
            model.load_state_dict(state['model'])
        if optimizer is not None and state.get('optimizer') is not None:
            optimizer.load_state_dict(state['optimizer'])
        if 'rng_cpu' in state:
            torch.random.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        return state


class TFAdapter(BaseAdapter):
    name = "tensorflow"
    ext = "weights.h5"

    def save(self, model: Any, dst_path: str, optimizer: Any = None) -> None:
        model.save_weights(dst_path)

    def load(self, model: Any, src_path: str, optimizer: Any = None) -> Any:
        model.load_weights(src_path)
        return {'model': 'loaded'}


class SklearnAdapter(BaseAdapter):
    name = "sklearn"
    ext = "pkl"

    def save(self, model: Any, dst_path: str, optimizer: Any = None) -> None:
        import joblib
        joblib.dump(model, dst_path)

    def load(self, model: Any, src_path: str, optimizer: Any = None) -> Any:
        import joblib
        return joblib.load(src_path)


class XGBoostAdapter(BaseAdapter):
    name = "xgboost"
    ext = "json"

    def save(self, booster: Any, dst_path: str, optimizer: Any = None) -> None:
        booster.save_model(dst_path)

    def load(self, booster: Any, src_path: str, optimizer: Any = None) -> Any:
        import xgboost as xgb
        if booster is None:
            booster = xgb.Booster()
        booster.load_model(src_path)
        return booster


class LightGBMAdapter(BaseAdapter):
    name = "lightgbm"
    ext = "txt"

    def save(self, booster: Any, dst_path: str, optimizer: Any = None) -> None:
        booster.save_model(dst_path)

    def load(self, booster: Any, src_path: str, optimizer: Any = None) -> Any:
        import lightgbm as lgb
        if booster is None:
            return lgb.Booster(model_file=src_path)
        booster.load_model(src_path)
        return booster


def pick_adapter(model: Any, framework: Optional[str] = None) -> BaseAdapter:
    name = (framework or "").lower().strip()
    # Explicit choice
    if name in ("torch", "pytorch"):
        return TorchAdapter()
    if name in ("tf", "tensorflow", "keras"):
        return TFAdapter()
    if name in ("sklearn", "scikit", "scikit-learn"):
        return SklearnAdapter()
    if name in ("xgb", "xgboost"):
        return XGBoostAdapter()
    if name in ("lgbm", "lightgbm"):
        return LightGBMAdapter()

    # Auto-detect by type module path
    module_name = type(model).__module__ if model is not None else ""
    if module_name.startswith("torch."):
        return TorchAdapter()
    if module_name.startswith(("tensorflow.", "keras.")):
        return TFAdapter()
    if module_name.startswith("sklearn."):
        return SklearnAdapter()
    if module_name.startswith("xgboost."):
        return XGBoostAdapter()
    if module_name.startswith("lightgbm."):
        return LightGBMAdapter()

    # Fallback to PyTorch if installed
    if _module_available("torch"):
        return TorchAdapter()
    raise RuntimeError("Could not determine framework adapter; please pass framework=...")


