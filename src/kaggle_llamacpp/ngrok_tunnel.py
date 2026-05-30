
import subprocess, time
from pathlib import Path
import requests
from .download import download_file, extract_archive
from .utils import chmod_exec, get_secret_or_env, kill_pidfile, tail, token_preview, write_state

NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip"

def ensure_ngrok(ngrok_dir="/kaggle/working/ngrok"):
    d=Path(ngrok_dir); d.mkdir(parents=True, exist_ok=True); b=d/"ngrok"
    if not b.exists():
        z=d/"ngrok.zip"; download_file(NGROK_URL,z,backend="aria2"); extract_archive(z,d); chmod_exec(b)
    print(subprocess.check_output([str(b),"version"], text=True).strip()); return b

def configure_ngrok_token(secret_name="NGROK_AUTHTOKEN", ngrok_bin="/kaggle/working/ngrok/ngrok"):
    token=get_secret_or_env(secret_name).strip()
    if not token or token in {"PASTE_TOKEN_HERE","ISI_TOKEN_NGROK_KAMU_DI_SINI"}: raise ValueError("Ngrok token kosong/placeholder.")
    if " " in token or "\n" in token or len(token)<20: raise ValueError(f"Format token mencurigakan: {token_preview(token)}")
    print("Loaded ngrok token:", token_preview(token))
    subprocess.run([str(ngrok_bin),"config","add-authtoken",token], stdout=subprocess.DEVNULL, check=True)
    return token

def start_ngrok_tunnel(port=8080, secret_name="NGROK_AUTHTOKEN", ngrok_dir="/kaggle/working/ngrok", log_path="/kaggle/working/ngrok.log", pid_path="/kaggle/working/ngrok.pid", timeout_s=60):
    b=ensure_ngrok(ngrok_dir); configure_ngrok_token(secret_name=secret_name, ngrok_bin=b)
    log_path, pid_path=Path(log_path), Path(pid_path)
    kill_pidfile(pid_path); subprocess.run(["bash","-lc","pkill -f 'ngrok http' 2>/dev/null || true"], check=False); log_path.unlink(missing_ok=True)
    proc=subprocess.Popen([str(b),"http",str(port),"--log=stdout"], stdout=open(log_path,"w"), stderr=subprocess.STDOUT)
    pid_path.write_text(str(proc.pid)); print("Started ngrok PID=", proc.pid)
    end=time.time()+timeout_s
    while time.time()<end:
        try:
            data=requests.get("http://127.0.0.1:4040/api/tunnels", timeout=3).json()
            for t in data.get("tunnels",[]):
                u=t.get("public_url","")
                if u.startswith("https://"):
                    Path("/kaggle/working/ngrok_public_url.txt").write_text(u+"\n")
                    write_state(ngrok_public_url=u, ngrok_pid=proc.pid)
                    print("PUBLIC_URL=", u); return u
        except Exception: pass
        if proc.poll() is not None:
            print(tail(log_path, 120)); raise RuntimeError("ngrok gagal start.")
        time.sleep(2)
    print(tail(log_path, 120)); raise TimeoutError("Tidak menemukan HTTPS tunnel.")

def stop_ngrok_tunnel(pid_path="/kaggle/working/ngrok.pid"):
    kill_pidfile(pid_path); subprocess.run(["bash","-lc","pkill -f 'ngrok http' 2>/dev/null || true"], check=False); print("ngrok stopped.")
