import os
import sys
from pathlib import Path
from typing import Any


PATCHED_SGLANG_PATH_ENV = "VERL_SGLANG_PATCHED_PATH"
PATCHED_SGLANG_PATH_KEY = "patched_sglang_path"


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    get = getattr(config, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except TypeError:
            return get(key)
    return getattr(config, key, default)


def get_patched_sglang_path(config_or_engine_kwargs: Any = None) -> str | None:
    """Return an opt-in patched SGLang import root from rollout config or env."""
    candidate = _config_get(config_or_engine_kwargs, PATCHED_SGLANG_PATH_KEY)
    if candidate is None:
        engine_kwargs = _config_get(config_or_engine_kwargs, "engine_kwargs", {}) or {}
        sglang_kwargs = _config_get(engine_kwargs, "sglang", {}) or {}
        candidate = _config_get(sglang_kwargs, PATCHED_SGLANG_PATH_KEY)
    if candidate is None:
        candidate = os.environ.get(PATCHED_SGLANG_PATH_ENV)
    if candidate in (None, ""):
        return None
    return str(candidate)


def configure_sglang_import(config_or_engine_kwargs: Any = None) -> str | None:
    """Put an opt-in patched SGLang tree first on sys.path before importing sglang."""
    patched_path = get_patched_sglang_path(config_or_engine_kwargs)
    if patched_path is None:
        return None

    import_root = str(Path(patched_path).expanduser().resolve())
    if not Path(import_root, "sglang").is_dir():
        raise FileNotFoundError(
            f"{PATCHED_SGLANG_PATH_KEY} must point to a directory containing an sglang package: {import_root}"
        )

    loaded_sglang = sys.modules.get("sglang")
    if loaded_sglang is not None:
        loaded_file = getattr(loaded_sglang, "__file__", "") or ""
        if not str(Path(loaded_file).resolve()).startswith(import_root):
            raise RuntimeError(
                "Patched SGLang was requested after another sglang package was already imported "
                f"from {loaded_file}. Set {PATCHED_SGLANG_PATH_ENV} before launching verl, or ensure "
                f"{PATCHED_SGLANG_PATH_KEY} is configured before SGLang rollout actors are imported."
            )

    if import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)
    os.environ[PATCHED_SGLANG_PATH_ENV] = import_root
    return import_root
