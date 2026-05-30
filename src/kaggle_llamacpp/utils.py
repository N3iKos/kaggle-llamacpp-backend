from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Optional

STATE_PATH = Path("/kaggle/working/llamacpp_state.json")


def read_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def write_state(**updates: Any) -> dict[str, Any]:
    state = read_state()
    state.update({k: str(v) if isinstance(v, Path) else v for k, v in updates.items()})
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state


def chmod_exec(path: str | Path) -> None:
    p = Path(path)
    p.chmod(p.stat().st_mode | 0o111)


def tail(path: str | Path, n: int = 80) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace", encoding="utf-8").splitlines()
    return "\n".join(lines[-n:])


def kill_pidfile(pidfile: str | Path) -> None:
    p = Path(pidfile)
    if not p.exists():
        return
    try:
        pid = int(p.read_text().strip())
    except Exception:
        p.unlink(missing_ok=True)
        return
    subprocess.run(["bash", "-lc", f"kill {pid} 2>/dev/null || true"], check=False)
    time.sleep(2)
    p.unlink(missing_ok=True)


def find_first(paths: Iterable[Path], name: str) -> Optional[Path]:
    for base in paths:
        for p in base.rglob(name):
            if p.is_file():
                return p
    return None


def clean_gemma4_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("<|channel>thought", "")
    text = text.replace("<|channel>final", "")
    text = text.replace("<channel|>", "")
    text = text.replace("<|turn>", "")
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()


def safe_token_preview(token: str) -> str:
    token = (token or "").strip()
    if len(token) <= 12:
        return f"<too-short len={len(token)}>"
    return f"{token[:6]}...{token[-4:]} (len={len(token)})"


def get_secret_or_env(name: str) -> str:
    token = os.environ.get(name, "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        token = UserSecretsClient().get_secret(name).strip()
        return token
    except Exception as exc:
        raise RuntimeError(
            f"Secret/env {name!r} tidak ditemukan. Di Kaggle, buat Add-ons > Secrets dengan nama {name!r} dan aktifkan untuk notebook ini."
        ) from exc
