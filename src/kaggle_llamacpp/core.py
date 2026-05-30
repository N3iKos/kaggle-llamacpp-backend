from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from .config import validate_config
from .download import download_file, extract_archive, filename_from_url
from .utils import chmod_exec, clean_gemma4_text, find_first, kill_pidfile, read_state, tail, write_state


def ensure_aria2c() -> None:
    if shutil.which("aria2c"):
        print("aria2c:", shutil.which("aria2c"))
        return
    print("aria2c belum ada. Install via apt...")
    subprocess.run(["bash", "-lc", "export DEBIAN_FRONTEND=noninteractive; sudo apt-get update -y -qq && sudo apt-get install -y -qq aria2"], check=True)
    print("aria2c:", shutil.which("aria2c"))


def _github_latest_asset(repo: str, *, cuda_version: str = "12.8", arch: str = "amd64") -> str:
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    r = requests.get(api, timeout=30)
    r.raise_for_status()
    matches = []
    for a in r.json().get("assets", []):
        name = a.get("name", "")
        url = a.get("browser_download_url", "")
        if name.endswith(".tar.gz") and f"cuda-{cuda_version}" in name and arch in name:
            matches.append((name, url))
    if not matches:
        raise RuntimeError(f"Tidak menemukan asset CUDA {cuda_version} {arch} di latest release {repo}.")
    return sorted(matches)[0][1]


def _runtime_env(llama_home: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    llama_home = str(llama_home)
    paths = [llama_home, "/usr/local/cuda-12.8/lib64", "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64", env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join([p for p in paths if p])
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    return env


def ensure_llamacpp_cuda(
    *,
    install_dir: str | Path = "/kaggle/working/llama.cpp-cuda",
    downloads_dir: str | Path = "/kaggle/working/downloads",
    repo: str = "ai-dock/llama.cpp-cuda",
    cuda_version: str = "12.8",
    force: bool = False,
) -> dict[str, Any]:
    ensure_aria2c()
    install_dir = Path(install_dir)
    downloads_dir = Path(downloads_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    existing = find_first([install_dir], "llama-server")
    if existing and not force:
        llama_bin = existing
    else:
        asset_url = _github_latest_asset(repo, cuda_version=cuda_version)
        archive = downloads_dir / Path(asset_url).name
        download_file(asset_url, archive, force=force, backend="aria2")
        if install_dir.exists():
            for child in install_dir.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        extract_archive(archive, install_dir)
        llama_bin = find_first([install_dir], "llama-server")
        if not llama_bin:
            raise FileNotFoundError("llama-server tidak ditemukan setelah extract.")

    llama_home = llama_bin.parent
    llama_cli = find_first([install_dir], "llama-cli")
    for p in [llama_bin, llama_cli]:
        if p:
            chmod_exec(p)

    env = _runtime_env(llama_home)
    print("===== LLAMA VERSION =====")
    subprocess.run([str(llama_bin), "--version"], env=env, check=False)
    print("\n===== CUDA DEVICES =====")
    subprocess.run([str(llama_bin), "--list-devices"], env=env, check=False)

    return write_state(llama_home=str(llama_home), llama_bin=str(llama_bin), llama_cli=str(llama_cli or ""), model_dir="/kaggle/working/models", port=8080)


def _output_name(url: str, configured_file: str = "") -> str:
    return configured_file or filename_from_url(url)


def download_assets(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    validate_config(cfg)
    ensure_aria2c()
    model_dir = Path("/kaggle/working/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    backend = cfg.get("download", {}).get("backend", "aria2")
    token_env = cfg.get("download", {}).get("hf_token_env", "HF_TOKEN")
    token = os.environ.get(token_env, "").strip() or None

    if model_cfg.get("model_url"):
        model_path = model_dir / _output_name(model_cfg["model_url"], model_cfg.get("model_file", ""))
        download_file(model_cfg["model_url"], model_path, token=token, force=force, backend=backend)
    else:
        model_path = Path(model_cfg["model_file"])

    updates: dict[str, Any] = {"model_path": str(model_path), "model_mode": model_cfg.get("mode", "text")}

    if model_cfg.get("mode") == "vision":
        if model_cfg.get("mmproj_url"):
            mmproj_path = model_dir / _output_name(model_cfg["mmproj_url"], model_cfg.get("mmproj_file", ""))
            download_file(model_cfg["mmproj_url"], mmproj_path, token=token, force=force, backend=backend)
        else:
            mmproj_path = Path(model_cfg["mmproj_file"])
        updates["mmproj_path"] = str(mmproj_path)

    return write_state(**updates)


def _api_key(cfg: dict[str, Any]) -> str:
    api = cfg.get("api", {})
    return os.environ.get(api.get("api_key_env", "LLAMA_API_KEY"), "").strip() or api.get("api_key", "")


def _auth_headers(cfg: dict[str, Any]) -> dict[str, str]:
    key = _api_key(cfg)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _reasoning_settings(cfg: dict[str, Any]) -> tuple[str, bool]:
    r = cfg.get("reasoning", {})
    mode = r.get("mode", "off")
    if mode == "off":
        return "none", False
    if mode == "parsed":
        return r.get("reasoning_format", "deepseek"), True
    if mode == "raw":
        return "none", True
    return r.get("reasoning_format", "none"), bool(r.get("enable_thinking", False))


def start_from_config(cfg: dict[str, Any]) -> int:
    validate_config(cfg)
    state = read_state()
    llama_bin = Path(state.get("llama_bin", ""))
    model_path = Path(state.get("model_path", ""))
    if not llama_bin.exists():
        raise FileNotFoundError("llama_bin tidak valid. Jalankan ensure_llamacpp_cuda().")
    if not model_path.exists():
        raise FileNotFoundError("model_path tidak valid. Jalankan download_assets(cfg).")

    log_path = Path("/kaggle/working/llama-server.log")
    pid_path = Path("/kaggle/working/llama-server.pid")
    kill_pidfile(pid_path)
    subprocess.run(["bash", "-lc", "pkill -f llama-server 2>/dev/null || true"], check=False)
    log_path.unlink(missing_ok=True)

    api = cfg["api"]
    rt = cfg["runtime"]
    model_cfg = cfg["model"]
    reasoning_format, enable_thinking = _reasoning_settings(cfg)
    cmd = [
        str(llama_bin),
        "--model", str(model_path),
        "--alias", str(api["alias"]),
        "--host", str(api.get("host", "0.0.0.0")),
        "--port", str(api.get("port", 8080)),
        "--ctx-size", str(rt["ctx_size"]),
        "--batch-size", str(rt["batch_size"]),
        "--ubatch-size", str(rt["ubatch_size"]),
        "--n-gpu-layers", str(rt["gpu_layers"]),
        "--split-mode", str(rt["split_mode"]),
        "--tensor-split", str(rt["tensor_split"]),
        "--threads", str(rt["threads"]),
        "--parallel", str(rt["parallel"]),
        "--cont-batching",
        "--reasoning-format", reasoning_format,
        "--chat-template-kwargs", json.dumps({"enable_thinking": enable_thinking}),
    ]
    api_key = _api_key(cfg)
    if api_key:
        cmd += ["--api-key", api_key]
    if api.get("metrics", True):
        cmd.append("--metrics")
    if not rt.get("warmup", True):
        cmd.append("--no-warmup")
    if model_cfg.get("mode") == "vision":
        mmproj_path = Path(state.get("mmproj_path", ""))
        if not mmproj_path.exists():
            raise FileNotFoundError("mmproj_path tidak valid untuk vision mode.")
        cmd += ["--mmproj", str(mmproj_path)]
        if not model_cfg.get("mmproj_offload", True):
            cmd.append("--no-mmproj-offload")
    cmd.append("--verbose")

    env = _runtime_env(llama_bin.parent)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    write_state(port=api.get("port", 8080), alias=api["alias"], api_key_set=bool(api_key), ctx_size=rt["ctx_size"], llama_pid=proc.pid, llama_log=str(log_path), command=" ".join(cmd))
    print(f"Started llama-server PID={proc.pid}")
    print(f"Alias: {api['alias']}")
    print(f"Port: {api.get('port', 8080)}")
    print(f"Log: {log_path}")
    time.sleep(5)
    print_status(lines=80)
    return proc.pid


def wait_until_ready(*, cfg: dict[str, Any], timeout_s: int | None = None) -> bool:
    port = int(cfg["api"].get("port", 8080))
    timeout_s = timeout_s or int(cfg.get("health", {}).get("timeout_s", 900))
    start = time.time()
    pid_path = Path("/kaggle/working/llama-server.pid")
    log_path = Path("/kaggle/working/llama-server.log")
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            print("health:", r.text)
            if r.status_code == 200 and '"status":"ok"' in r.text:
                return True
        except Exception as exc:
            print("health error:", exc)
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            alive = subprocess.run(["bash", "-lc", f"kill -0 {pid} 2>/dev/null"], check=False).returncode == 0
            if not alive:
                print("llama-server process sudah mati.")
                print(tail(log_path, 240))
                return False
        print(tail(log_path, 30))
        time.sleep(10)
    print("Timeout menunggu server ready.")
    print(tail(log_path, 240))
    return False


def test_models_endpoint(cfg: dict[str, Any]) -> dict[str, Any]:
    port = int(cfg["api"].get("port", 8080))
    alias = cfg["api"]["alias"]
    r = requests.get(f"http://127.0.0.1:{port}/v1/models", headers=_auth_headers(cfg), timeout=30)
    print("HTTP", r.status_code)
    print(r.text[:2000])
    r.raise_for_status()
    data = r.json()
    ids = [m.get("id") for m in data.get("data", [])]
    if alias not in ids:
        print(f"Warning: alias {alias!r} tidak muncul di /v1/models. ids={ids}")
    return data


def test_chat_completion(cfg: dict[str, Any], prompt: str | None = None, *, max_tokens: int | None = None, temperature: float = 0.7, clean: bool = True) -> str:
    port = int(cfg["api"].get("port", 8080))
    prompt = prompt or cfg.get("health", {}).get("test_prompt", "Say hello.")
    max_tokens = max_tokens or int(cfg.get("health", {}).get("test_max_tokens", 64))
    payload = {"model": cfg["api"]["alias"], "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, headers=_auth_headers(cfg), timeout=180)
    print("HTTP", r.status_code)
    print(r.text[:3000])
    r.raise_for_status()
    data = r.json()
    choice = data.get("choices", [{}])[0].get("message", {})
    raw = data.get("__verbose", {}).get("content") or choice.get("content") or choice.get("reasoning_content") or ""
    out = clean_gemma4_text(raw) if clean else raw
    print("\n===== CLEAN TEXT =====")
    print(out)
    return out


def test_vision_completion(cfg: dict[str, Any], image_path: str | Path, prompt: str = "Describe this image.", *, mime_type: str = "image/jpeg", max_tokens: int = 512) -> str:
    if cfg.get("model", {}).get("mode") != "vision":
        raise ValueError("Config bukan mode vision.")
    port = int(cfg["api"].get("port", 8080))
    image_path = Path(image_path)
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": cfg["api"]["alias"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}]}],
        "max_tokens": max_tokens,
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, headers=_auth_headers(cfg), timeout=300)
    print("HTTP", r.status_code)
    print(r.text[:4000])
    r.raise_for_status()
    data = r.json()
    choice = data.get("choices", [{}])[0].get("message", {})
    raw = data.get("__verbose", {}).get("content") or choice.get("content") or choice.get("reasoning_content") or ""
    out = clean_gemma4_text(raw)
    print("\n===== CLEAN VISION TEXT =====")
    print(out)
    return out


def print_status(*, lines: int = 80) -> None:
    print("===== PROCESS =====")
    subprocess.run(["bash", "-lc", "ps -eo pid,ppid,pcpu,pmem,etime,cmd | grep -E 'llama-server|PID' | grep -v grep || true"], check=False)
    print("\n===== GPU =====")
    subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true"], check=False)
    print("\n===== LOG TAIL =====")
    print(tail("/kaggle/working/llama-server.log", lines))


def stop_llama_server() -> None:
    kill_pidfile("/kaggle/working/llama-server.pid")
    subprocess.run(["bash", "-lc", "pkill -f llama-server 2>/dev/null || true"], check=False)
    print("llama-server stopped.")
