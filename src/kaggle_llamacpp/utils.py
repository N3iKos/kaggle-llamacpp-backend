
import json, os, re, subprocess, time
from pathlib import Path

STATE_PATH = Path("/kaggle/working/llamacpp_state.json")

def read_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}

def write_state(**kw):
    s = read_state()
    for k, v in kw.items():
        s[k] = str(v) if isinstance(v, Path) else v
    STATE_PATH.write_text(json.dumps(s, indent=2, sort_keys=True))
    return s

def chmod_exec(p):
    p = Path(p); p.chmod(p.stat().st_mode | 0o111)

def tail(p, n=80):
    p = Path(p)
    if not p.exists(): return ""
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])

def kill_pidfile(pidfile):
    p = Path(pidfile)
    if not p.exists(): return
    try:
        pid = int(p.read_text().strip())
        subprocess.run(["bash","-lc",f"kill {pid} 2>/dev/null || true"], check=False)
        time.sleep(2)
    except Exception:
        pass
    p.unlink(missing_ok=True)

def find_first(bases, name):
    for b in bases:
        b = Path(b)
        if b.exists():
            for p in b.rglob(name):
                if p.is_file(): return p
    return None

def clean_gemma4_text(t):
    if not t: return ""
    for x in ["<|channel>thought","<|channel>final","<channel|>","<|turn>"]:
        t = t.replace(x, "")
    return re.sub(r"<\|.*?\|>", "", t).strip()

def get_secret_or_env(name):
    v = os.environ.get(name, "").strip()
    if v: return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name).strip()
    except Exception as e:
        raise RuntimeError(f"Secret/env {name!r} tidak ditemukan. Pastikan secret dicentang/attached di Kaggle Secrets.") from e

def token_preview(t):
    t = (t or "").strip()
    return f"{t[:6]}...{t[-4:]} len={len(t)}" if len(t) > 12 else f"too-short len={len(t)}"
