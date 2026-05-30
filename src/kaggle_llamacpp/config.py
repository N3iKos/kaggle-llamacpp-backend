from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULT_CONFIG: dict[str, Any] = {
    "api": {
        "host": "0.0.0.0",
        "port": 8080,
        "alias": "gemma-4-local",
        "api_key": "local",
        "api_key_env": "LLAMA_API_KEY",
        "metrics": True,
    },
    "model": {
        "mode": "text",  # text | vision
        "model_url": "",
        "model_file": "",
        "mmproj_url": "",
        "mmproj_file": "",
        "mmproj_offload": True,
    },
    "runtime": {
        "profile": "balanced",
        "ctx_size": 8192,
        "batch_size": 256,
        "ubatch_size": 64,
        "gpu_layers": 999,
        "split_mode": "layer",
        "tensor_split": "1,1",
        "threads": 4,
        "parallel": 1,
        "warmup": True,
        "allow_unsafe_context": False,
    },
    "reasoning": {
        "mode": "off",  # off | raw | parsed
        "reasoning_format": "none",
        "enable_thinking": False,
        "clean_gemma4_tags": True,
        "prefer_verbose_content": True,
    },
    "health": {
        "timeout_s": 900,
        "test_prompt": "Say hello in one short sentence. Do not explain your reasoning.",
        "test_max_tokens": 64,
        "check_models_endpoint": True,
    },
    "download": {
        "backend": "aria2",
        "hf_token_env": "HF_TOKEN",
    },
}

PROFILES: dict[str, dict[str, Any]] = {
    "safe": {"runtime": {"ctx_size": 4096, "batch_size": 256, "ubatch_size": 64, "parallel": 1}},
    "balanced": {"runtime": {"ctx_size": 8192, "batch_size": 256, "ubatch_size": 64, "parallel": 1}},
    "long_context_20k": {"runtime": {"ctx_size": 20000, "batch_size": 256, "ubatch_size": 64, "parallel": 1}},
    "vision_safe": {"model": {"mode": "vision"}, "runtime": {"ctx_size": 4096, "batch_size": 256, "ubatch_size": 64, "parallel": 1}},
}


def validate_config(cfg: dict[str, Any]) -> None:
    mode = cfg.get("model", {}).get("mode", "text")
    if mode not in {"text", "vision"}:
        raise ValueError("model.mode harus 'text' atau 'vision'.")
    if not cfg.get("model", {}).get("model_url") and not cfg.get("model", {}).get("model_file"):
        raise ValueError("model.model_url atau model.model_file wajib diisi.")
    if mode == "vision" and not cfg["model"].get("mmproj_url") and not cfg["model"].get("mmproj_file"):
        raise ValueError("mode vision butuh model.mmproj_url atau model.mmproj_file.")
    if not cfg.get("api", {}).get("alias"):
        raise ValueError("api.alias wajib diisi supaya /v1/models bisa dideteksi aplikasi.")
    ctx = int(cfg.get("runtime", {}).get("ctx_size", 0))
    if ctx > 20000 and not cfg.get("runtime", {}).get("allow_unsafe_context", False):
        raise ValueError(f"ctx_size={ctx} terlalu agresif. Pakai <=20000 atau set runtime.allow_unsafe_context=True.")


def load_config(path: str | Path | None = None, *, profile: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        cfg = deep_merge(cfg, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    selected = profile or cfg.get("runtime", {}).get("profile")
    if selected:
        cfg = deep_merge(cfg, PROFILES.get(selected, {}))
        cfg["runtime"]["profile"] = selected
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    validate_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def print_effective_config(cfg: dict[str, Any]) -> None:
    safe = copy.deepcopy(cfg)
    if safe.get("api", {}).get("api_key"):
        safe["api"]["api_key"] = "<set>"
    print(yaml.safe_dump(safe, sort_keys=False))
