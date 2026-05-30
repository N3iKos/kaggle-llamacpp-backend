
import base64, json, os, shutil, subprocess, time
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
    mmproj_dir: Path = Path("/kaggle/working/mmproj")
    log_path: Path = Path("/kaggle/working/llama-server.log")
    pid_path: Path = Path("/kaggle/working/llama-server.pid")
    def make_dirs(self):
        for p in [self.downloads_dir,self.llama_dir,self.model_dir,self.mmproj_dir]: p.mkdir(parents=True, exist_ok=True)

@dataclass
class ServerConfig:
    model_alias: str = "local"
    host: str = "0.0.0.0"
    port: int = 8080
    ctx_size: int = 8192
    batch_size: int = 256
    ubatch_size: int = 64
    gpu_layers: int = 999
    split_mode: str = "layer"
    tensor_split: str = "1,1"
    cuda_visible_devices: str = "0,1"
    threads: int = 4
    parallel: int = 1
    reasoning_format: str = "none"
    enable_thinking: bool = False
    enable_vision: object = "auto"
    mmproj_offload: bool = True
    image_min_tokens: int | None = None
    image_max_tokens: int | None = None
    cont_batching: bool = True
    metrics: bool = True
    verbose: bool = True
    no_warmup: bool = False
    api_key: str | None = None

def ensure_aria2c():
    if shutil.which("aria2c"):
        print("aria2c:", shutil.which("aria2c")); return
    subprocess.run(["bash","-lc","export DEBIAN_FRONTEND=noninteractive; sudo apt-get update -y -qq && sudo apt-get install -y -qq aria2"], check=True)
    print("aria2c:", shutil.which("aria2c"))

def _latest_asset(repo="ai-dock/llama.cpp-cuda", cuda="12.8", arch="amd64"):
    r = requests.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=30); r.raise_for_status()
    m = sorted((a["name"], a["browser_download_url"]) for a in r.json()["assets"] if a["name"].endswith(".tar.gz") and f"cuda-{cuda}" in a["name"] and arch in a["name"])
    if not m: raise RuntimeError("Asset llama.cpp CUDA tidak ditemukan.")
    return m[0][1]

def _env(home, devices="0,1"):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = devices
    e["LD_LIBRARY_PATH"] = ":".join([str(home),"/usr/local/cuda-12.8/lib64","/usr/local/cuda/lib64","/usr/local/nvidia/lib64",e.get("LD_LIBRARY_PATH","")])
    return e

def ensure_llamacpp_cuda(paths=None, force=False):
    paths = paths or RuntimePaths(); paths.make_dirs(); ensure_aria2c()
    llama = find_first([paths.llama_dir], "llama-server")
    if not llama or force:
        url = _latest_asset()
        arc = paths.downloads_dir / Path(url).name
        download_file(url, arc, backend="aria2", force=force)
        if paths.llama_dir.exists():
            for c in paths.llama_dir.iterdir():
                shutil.rmtree(c) if c.is_dir() else c.unlink()
        extract_archive(arc, paths.llama_dir)
        llama = find_first([paths.llama_dir], "llama-server")
    if not llama: raise FileNotFoundError("llama-server tidak ditemukan.")
    cli = find_first([paths.llama_dir], "llama-cli")
    for p in [llama, cli]:
        if p: chmod_exec(p)
    env = _env(llama.parent)
    subprocess.run([str(llama),"--version"], env=env, check=False)
    subprocess.run([str(llama),"--list-devices"], env=env, check=False)
    return write_state(llama_home=str(llama.parent), llama_bin=str(llama), llama_cli=str(cli or ""), model_dir=str(paths.model_dir), mmproj_dir=str(paths.mmproj_dir), port=8080)

def download_model(model_url, output_name=None, hf_token=None, paths=None, force=False, backend="aria2"):
    paths = paths or RuntimePaths(); paths.make_dirs(); ensure_aria2c()
    dest = paths.model_dir / (output_name or filename_from_url(model_url))
    token = hf_token or os.environ.get("HF_TOKEN","").strip() or None
    p = download_file(model_url, dest, token=token, force=force, backend=backend)
    write_state(model_path=str(p), model_url=model_url)
    return p

def download_mmproj(mmproj_url, output_name=None, hf_token=None, paths=None, force=False, backend="aria2"):
    paths = paths or RuntimePaths(); paths.make_dirs(); ensure_aria2c()
    dest = paths.mmproj_dir / (output_name or filename_from_url(mmproj_url))
    token = hf_token or os.environ.get("HF_TOKEN","").strip() or None
    p = download_file(mmproj_url, dest, token=token, force=force, backend=backend)
    write_state(mmproj_path=str(p), mmproj_url=mmproj_url, vision_enabled=True)
    return p

def download_assets(model_url, mmproj_url="", hf_token=None, backend="aria2", force=False):
    m = download_model(model_url, hf_token=hf_token, backend=backend, force=force)
    mm = download_mmproj(mmproj_url, hf_token=hf_token, backend=backend, force=force) if mmproj_url.strip() else None
    if not mm: write_state(mmproj_path="", mmproj_url="", vision_enabled=False)
    out = {"model_path": str(m), "mmproj_path": str(mm) if mm else None, "vision_enabled": bool(mm)}
    print(json.dumps(out, indent=2)); return out

def start_llama_server(config=None, paths=None):
    config = config or ServerConfig(); paths = paths or RuntimePaths(); s = read_state()
    llama, model = Path(s.get("llama_bin","")), Path(s.get("model_path",""))
    mmproj = Path(s.get("mmproj_path","")) if s.get("mmproj_path") else None
    if not llama.exists(): raise FileNotFoundError("Jalankan ensure_llamacpp_cuda dulu.")
    if not model.exists(): raise FileNotFoundError("Jalankan download_assets dulu.")
    use_vision = bool(mmproj and mmproj.exists()) if config.enable_vision == "auto" else bool(config.enable_vision)
    if use_vision and (not mmproj or not mmproj.exists()): raise FileNotFoundError("Vision aktif tapi mmproj belum ada.")
    kill_pidfile(paths.pid_path); subprocess.run(["bash","-lc","pkill -f llama-server 2>/dev/null || true"], check=False); paths.log_path.unlink(missing_ok=True)
    cmd = [str(llama),"--model",str(model),"--alias",config.model_alias,"--host",config.host,"--port",str(config.port),"--ctx-size",str(config.ctx_size),"--batch-size",str(config.batch_size),"--ubatch-size",str(config.ubatch_size),"--n-gpu-layers",str(config.gpu_layers),"--split-mode",config.split_mode,"--tensor-split",config.tensor_split,"--threads",str(config.threads),"--parallel",str(config.parallel),"--reasoning-format",config.reasoning_format,"--chat-template-kwargs",json.dumps({"enable_thinking": config.enable_thinking})]
    if config.cont_batching: cmd.append("--cont-batching")
    if config.metrics: cmd.append("--metrics")
    if config.verbose: cmd.append("--verbose")
    if config.no_warmup: cmd.append("--no-warmup")
    if config.api_key: cmd += ["--api-key", config.api_key]
    if use_vision:
        cmd += ["--mmproj", str(mmproj)]
        if config.mmproj_offload: cmd.append("--mmproj-offload")
        if config.image_min_tokens is not None: cmd += ["--image-min-tokens", str(config.image_min_tokens)]
        if config.image_max_tokens is not None: cmd += ["--image-max-tokens", str(config.image_max_tokens)]
    proc = subprocess.Popen(cmd, stdout=open(paths.log_path,"w"), stderr=subprocess.STDOUT, env=_env(llama.parent, config.cuda_visible_devices))
    paths.pid_path.write_text(str(proc.pid))
    write_state(port=config.port, model_alias=config.model_alias, vision_active=use_vision, llama_pid=proc.pid)
    print(f"Started llama-server PID={proc.pid}\nModel alias: {config.model_alias}\nVision active: {use_vision}\nLog: {paths.log_path}")
    time.sleep(5); print_status(paths=paths, lines=80); return proc.pid

def wait_until_ready(port=None, timeout_s=900, paths=None):
    paths = paths or RuntimePaths(); port = port or int(read_state().get("port",8080)); t0=time.time()
    while time.time()-t0 < timeout_s:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            print("health:", r.text)
            if r.status_code == 200 and '"status":"ok"' in r.text: return True
        except Exception as e: print("health error:", e)
        print(tail(paths.log_path, 24)); time.sleep(10)
    return False

def _headers(api_key=None):
    h = {"Content-Type":"application/json"}
    if api_key: h["Authorization"] = f"Bearer {api_key}"
    return h

def test_models_endpoint(port=None, api_key=None):
    port = port or int(read_state().get("port",8080))
    r = requests.get(f"http://127.0.0.1:{port}/v1/models", headers=_headers(api_key), timeout=30)
    print("GET /v1/models:", r.status_code); print(r.text[:3000]); r.raise_for_status(); return r.json()

def test_chat_completion(prompt="Say hello in one short sentence.", port=None, model=None, api_key=None, max_tokens=64, temperature=.7, clean=True):
    s=read_state(); port = port or int(s.get("port",8080)); model = model or s.get("model_alias","local")
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", headers=_headers(api_key), json={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":temperature}, timeout=180)
    print("POST /v1/chat/completions:", r.status_code); print(r.text[:3000]); r.raise_for_status()
    d = r.json(); msg = d.get("choices",[{}])[0].get("message",{})
    raw = d.get("__verbose",{}).get("content") or msg.get("content") or msg.get("reasoning_content") or ""
    out = clean_gemma4_text(raw) if clean else raw
    print("\n===== CLEAN TEXT ====="); print(out); return out

def _tiny_png():
    return "data:image/png;base64," + base64.b64encode(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p94AAAAASUVORK5CYII=")).decode()

def test_vision_completion(port=None, model=None, api_key=None, prompt="Describe this image briefly.", skip_if_no_mmproj=True, max_tokens=128):
    s=read_state()
    if skip_if_no_mmproj and not s.get("vision_active"):
        print("Vision test skipped: mmproj tidak aktif."); return None
    port = port or int(s.get("port",8080)); model = model or s.get("model_alias","local")
    payload={"model":model,"messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":_tiny_png()}}]}],"max_tokens":max_tokens,"temperature":.2}
    r=requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", headers=_headers(api_key), json=payload, timeout=240)
    print("POST vision:", r.status_code); print(r.text[:3000]); r.raise_for_status()
    d=r.json(); msg=d.get("choices",[{}])[0].get("message",{})
    out=clean_gemma4_text(d.get("__verbose",{}).get("content") or msg.get("content") or msg.get("reasoning_content") or "")
    print("\n===== CLEAN VISION TEXT ====="); print(out); return out

def print_status(paths=None, lines=80):
    paths = paths or RuntimePaths()
    print("===== PROCESS ====="); subprocess.run(["bash","-lc","ps -eo pid,ppid,pcpu,pmem,etime,cmd | grep -E 'llama-server|PID' | grep -v grep || true"], check=False)
    print("\n===== GPU ====="); subprocess.run(["bash","-lc","nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true"], check=False)
    print("\n===== LOG TAIL ====="); print(tail(paths.log_path, lines))

def stop_llama_server(paths=None):
    paths = paths or RuntimePaths(); kill_pidfile(paths.pid_path); subprocess.run(["bash","-lc","pkill -f llama-server 2>/dev/null || true"], check=False); print("llama-server stopped.")
