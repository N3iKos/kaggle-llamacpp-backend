from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests

from .download import download_with_tqdm, extract_archive
from .utils import chmod_exec, get_secret_or_env, kill_pidfile, safe_token_preview, tail, write_state

NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip"


def ensure_ngrok(ngrok_dir: str | Path = "/kaggle/working/ngrok") -> Path:
    ngrok_dir = Path(ngrok_dir)
    ngrok_dir.mkdir(parents=True, exist_ok=True)
    ngrok_bin = ngrok_dir / "ngrok"
    if not ngrok_bin.exists():
        archive = ngrok_dir / "ngrok.zip"
        download_with_tqdm(NGROK_URL, archive)
        extract_archive(archive, ngrok_dir)
        chmod_exec(ngrok_bin)
    out = subprocess.run([str(ngrok_bin), "version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(out.stdout.strip())
    return ngrok_bin


def configure_ngrok_token(*, secret_name: str = "NGROK_AUTHTOKEN", ngrok_bin: str | Path = "/kaggle/working/ngrok/ngrok") -> str:
    token = get_secret_or_env(secret_name).strip()
    if not token or token in {"PASTE_TOKEN_HERE", "ISI_TOKEN_NGROK_KAMU_DI_SINI"}:
        raise ValueError("Ngrok token masih placeholder/kosong.")
    if " " in token or "\n" in token or len(token) < 20:
        raise ValueError(f"Format token mencurigakan: {safe_token_preview(token)}. Pastikan memakai authtoken, bukan API key/command lengkap.")
    print("Loaded ngrok token:", safe_token_preview(token))
    subprocess.run([str(ngrok_bin), "config", "add-authtoken", token], stdout=subprocess.DEVNULL, check=True)
    print("ngrok authtoken configured.")
    return token


def start_ngrok_tunnel(*, port: int = 8080, secret_name: str = "NGROK_AUTHTOKEN", ngrok_dir: str | Path = "/kaggle/working/ngrok", log_path: str | Path = "/kaggle/working/ngrok.log", pid_path: str | Path = "/kaggle/working/ngrok.pid", timeout_s: int = 60) -> str:
    ngrok_bin = ensure_ngrok(ngrok_dir)
    configure_ngrok_token(secret_name=secret_name, ngrok_bin=ngrok_bin)
    log_path = Path(log_path)
    pid_path = Path(pid_path)
    kill_pidfile(pid_path)
    subprocess.run(["bash", "-lc", "pkill -f 'ngrok http' 2>/dev/null || true"], check=False)
    log_path.unlink(missing_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([str(ngrok_bin), "http", str(port), "--log=stdout"], stdout=log_f, stderr=subprocess.STDOUT)
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"Started ngrok PID={proc.pid}")
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
            data = r.json()
            for t in data.get("tunnels", []):
                url = t.get("public_url", "")
                if url.startswith("https://"):
                    Path("/kaggle/working/ngrok_public_url.txt").write_text(url + "\n", encoding="utf-8")
                    write_state(ngrok_public_url=url, ngrok_pid=proc.pid)
                    print("PUBLIC_URL=", url)
                    return url
        except Exception as exc:
            last_err = str(exc)
        if proc.poll() is not None:
            print("ngrok process exited.")
            print(tail(log_path, 120))
            raise RuntimeError("ngrok gagal start. Lihat log di atas.")
        time.sleep(2)
    print(tail(log_path, 120))
    raise TimeoutError(f"Tidak menemukan HTTPS tunnel dalam {timeout_s}s. Last error: {last_err}")


def stop_ngrok_tunnel(pid_path: str | Path = "/kaggle/working/ngrok.pid") -> None:
    kill_pidfile(pid_path)
    subprocess.run(["bash", "-lc", "pkill -f 'ngrok http' 2>/dev/null || true"], check=False)
    print("ngrok stopped.")
