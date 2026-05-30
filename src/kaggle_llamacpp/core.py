from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .download import download_file, extract_archive, filename_from_url
from .utils import chmod_exec, clean_gemma4_text, find_first, kill_pidfile, read_state, tail, write_state


@dataclass
class RuntimePaths:
    base_dir: Path = Path("/kaggle/working")
    downloads_dir: Path = Path("/kaggle/working/downloads")
    llama_dir: Path = Path("/kaggle/working/llama.cpp-cuda")
    model_dir: Path = Path("/kaggle/working/models")
    log_path: Path = Path("/kaggle/working/llama-server.log")
    pid_path: Path = Path("/kaggle/working/llama-server.pid")

    def make_dirs(self) -> None:
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.llama_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ServerConfig:
    ctx_size: int = 2048
    batch_size: int = 256
    ubatch_size: int = 64
    port: int = 8080
    gpu_layers: int = 999
    split_mode: str = "layer"
    tensor_split: str = "1,1"
    threads: int = 4
    parallel: int = 1
    reasoning_format: str = "none"
    enable_thinking: bool = False
    verbose: bool = True


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
    data = r.json()
    assets = data.get("assets", [])
    matches: list[tuple[str, str]] = []
    for a in assets:
        name = a.get("name", "")
        url = a.get("browser_download_url", "")
        if name.endswith(".tar.gz") and f"cuda-{cuda_version}" in name and arch in name:
            matches.append((name, url))
    if not matches:
        raise RuntimeError(f"Tidak menemukan asset CUDA {cuda_version} {arch} di latest release {repo}.")
    matches.sort()
    return matches[0][1]


def ensure_llamacpp_cuda(
    *,
    paths: RuntimePaths | None = None,
    repo: str = "ai-dock/llama.cpp-cuda",
    cuda_version: str = "12.8",
    force: bool = False,
) -> dict:
    paths = paths or RuntimePaths()
    paths.make_dirs()
    ensure_aria2c()

    existing = find_first([paths.llama_dir], "llama-server")
    if existing and not force:
        llama_bin = existing
        llama_home = llama_bin.parent
    else:
        asset_url = _github_latest_asset(repo, cuda_version=cuda_version)
        archive = paths.downloads_dir / Path(asset_url).name
        download_file(asset_url, archive, force=force, backend="aria2")
        if paths.llama_dir.exists():
            for child in paths.llama_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        extract_archive(archive, paths.llama_dir)
        llama_bin = find_first([paths.llama_dir], "llama-server")
        if not llama_bin:
            raise FileNotFoundError("llama-server tidak ditemukan setelah extract.")
        llama_home = llama_bin.parent

    llama_cli = find_first([paths.llama_dir], "llama-cli")
    for p in [llama_bin, llama_cli]:
        if p:
            chmod_exec(p)

    env = _runtime_env(str(llama_home))
    print("===== LLAMA VERSION =====")
    subprocess.run([str(llama_bin), "--version"], env=env, check=False)
    print("\n===== CUDA DEVICES =====")
    subprocess.run([str(llama_bin), "--list-devices"], env=env, check=False)

    state = write_state(
        llama_home=str(llama_home),
        llama_bin=str(llama_bin),
        llama_cli=str(llama_cli) if llama_cli else "",
        model_dir=str(paths.model_dir),
        port=8080,
    )
    return state


def download_model(
    model_url: str,
    *,
    output_name: str | None = None,
    hf_token: str | None = None,
    paths: RuntimePaths | None = None,
    force: bool = False,
    backend: str = "aria2",
) -> Path:
    paths = paths or RuntimePaths()
    paths.make_dirs()
    ensure_aria2c()

    output_name = output_name or filename_from_url(model_url)
    dest = paths.model_dir / output_name

    token = hf_token or os.environ.get("HF_TOKEN", "").strip() or None
    path = download_file(model_url, dest, token=token, force=force, backend=backend)

    if path.suffix.lower() != ".gguf":
        print(f"Warning: file bukan .gguf: {path.name}")

    write_state(model_path=str(path), model_url=model_url, model_dir=str(paths.model_dir))
    print(f"MODEL_PATH={path}")
    return path


def _runtime_env(llama_home: str | Path) -> dict[str, str]:
    env = os.environ.copy()
    llama_home = str(llama_home)
    lib_paths = [
        llama_home,
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda/lib64",
        "/usr/local/nvidia/lib64",
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join([p for p in lib_paths if p])
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    return env


def start_llama_server(config: ServerConfig | None = None, *, paths: RuntimePaths | None = None) -> int:
    config = config or ServerConfig()
    paths = paths or RuntimePaths()
    state = read_state()

    llama_bin = Path(state.get("llama_bin", ""))
    model_path = Path(state.get("model_path", ""))

    if not llama_bin.exists():
        raise FileNotFoundError("llama_bin tidak valid. Jalankan ensure_llamacpp_cuda() dulu.")
    if not model_path.exists():
        raise FileNotFoundError("model_path tidak valid. Jalankan download_model() dulu.")

    kill_pidfile(paths.pid_path)
    subprocess.run(["bash", "-lc", "pkill -f llama-server 2>/dev/null || true"], check=False)
    paths.log_path.unlink(missing_ok=True)

    llama_home = llama_bin.parent
    env = _runtime_env(llama_home)

    cmd = [
        str(llama_bin),
        "--model", str(model_path),
        "--host", "0.0.0.0",
        "--port", str(config.port),
        "--ctx-size", str(config.ctx_size),
        "--batch-size", str(config.batch_size),
        "--ubatch-size", str(config.ubatch_size),
        "--n-gpu-layers", str(config.gpu_layers),
        "--split-mode", config.split_mode,
        "--tensor-split", config.tensor_split,
        "--threads", str(config.threads),
        "--parallel", str(config.parallel),
        "--cont-batching",
        "--metrics",
        "--reasoning-format", config.reasoning_format,
        "--chat-template-kwargs", json.dumps({"enable_thinking": config.enable_thinking}),
    ]
    if config.verbose:
        cmd.append("--verbose")

    log_f = open(paths.log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    paths.pid_path.write_text(str(proc.pid), encoding="utf-8")

    write_state(
        port=config.port,
        ctx_size=config.ctx_size,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        reasoning_format=config.reasoning_format,
        enable_thinking=config.enable_thinking,
        llama_pid=proc.pid,
        llama_log=str(paths.log_path),
    )

    print(f"Started llama-server PID={proc.pid}")
    print(f"Log: {paths.log_path}")
    time.sleep(5)
    print_status(paths=paths, lines=80)
    return proc.pid


def wait_until_ready(*, port: int | None = None, timeout_s: int = 900, paths: RuntimePaths | None = None) -> bool:
    paths = paths or RuntimePaths()
    state = read_state()
    port = port or int(state.get("port", 8080))
    start = time.time()

    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            text = r.text
            print("health:", text)
            if r.status_code == 200 and '"status":"ok"' in text:
                return True
        except Exception as exc:
            print("health error:", exc)

        if paths.pid_path.exists():
            pid = int(paths.pid_path.read_text().strip())
            alive = subprocess.run(["bash", "-lc", f"kill -0 {pid} 2>/dev/null"], check=False).returncode == 0
            if not alive:
                print("llama-server process sudah mati.")
                print(tail(paths.log_path, 240))
                return False

        print(tail(paths.log_path, 30))
        time.sleep(10)

    print("Timeout menunggu server ready.")
    print(tail(paths.log_path, 240))
    return False


def test_chat_completion(
    prompt: str = "Say hello in one short sentence.",
    *,
    port: int | None = None,
    max_tokens: int = 64,
    temperature: float = 0.7,
    clean: bool = True,
) -> str:
    state = read_state()
    port = port or int(state.get("port", 8080))

    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=180)
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


def print_status(*, paths: RuntimePaths | None = None, lines: int = 80) -> None:
    paths = paths or RuntimePaths()
    print("===== PROCESS =====")
    subprocess.run(["bash", "-lc", "ps -eo pid,ppid,pcpu,pmem,etime,cmd | grep -E 'llama-server|PID' | grep -v grep || true"], check=False)
    print("\n===== GPU =====")
    subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true"], check=False)
    print("\n===== LOG TAIL =====")
    print(tail(paths.log_path, lines))


def stop_llama_server(*, paths: RuntimePaths | None = None) -> None:
    paths = paths or RuntimePaths()
    kill_pidfile(paths.pid_path)
    subprocess.run(["bash", "-lc", "pkill -f llama-server 2>/dev/null || true"], check=False)
    print("llama-server stopped.")
