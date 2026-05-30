from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm.auto import tqdm


def filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError("Tidak bisa menentukan nama file dari URL. Isi output_name manual.")
    return name


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _rpc_call(port: int, secret: str, method: str, params: list | None = None, timeout: int = 10):
    payload = {
        "jsonrpc": "2.0",
        "id": str(random.random()),
        "method": method,
        "params": [f"token:{secret}"] + (params or []),
    }
    r = requests.post(f"http://127.0.0.1:{port}/jsonrpc", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"aria2 RPC error: {data['error']}")
    return data.get("result")


def download_with_requests_tqdm(
    url: str,
    dest: str | Path,
    *,
    token: Optional[str] = None,
    force: bool = False,
    resume: bool = True,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Fallback downloader. Single-stream tapi progress realtime."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"File sudah ada, skip: {dest} ({dest.stat().st_size / 1024**3:.2f} GiB)")
        return dest

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    existing = 0
    mode = "wb"
    if resume and dest.exists() and not force:
        existing = dest.stat().st_size
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60) as r:
        if existing and r.status_code == 200:
            existing = 0
            mode = "wb"
        r.raise_for_status()
        total_header = r.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        if total is not None and existing and r.status_code == 206:
            total += existing

        with open(dest, mode) as f, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            mininterval=0.5,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def download_with_aria2_tqdm(
    url: str,
    dest: str | Path,
    *,
    token: Optional[str] = None,
    force: bool = False,
    connections: int = 16,
    split: int = 16,
    min_split_size: str = "64M",
    poll_interval: float = 0.5,
) -> Path:
    """aria2c downloader with tqdm progress.

    aria2 tetap melakukan parallel segmented download. Python hanya polling JSON-RPC
    untuk menggambar progress bar yang lebih rapi/realtime di Kaggle.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"File sudah ada, skip: {dest} ({dest.stat().st_size / 1024**3:.2f} GiB)")
        return dest

    aria2c = shutil.which("aria2c")
    if not aria2c:
        raise FileNotFoundError("aria2c tidak ditemukan. Jalankan ensure_aria2c() atau install aria2.")

    if force:
        dest.unlink(missing_ok=True)
        Path(str(dest) + ".aria2").unlink(missing_ok=True)

    port = _free_port()
    secret = f"kaggle-{random.randrange(10**12, 10**13)}"
    log_path = dest.parent / f".aria2-rpc-{dest.name}.log"

    cmd = [
        aria2c,
        "--enable-rpc=true",
        "--rpc-listen-all=false",
        f"--rpc-listen-port={port}",
        f"--rpc-secret={secret}",
        "--console-log-level=warn",
        "--summary-interval=0",
        "--quiet=true",
    ]

    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    try:
        # Wait RPC ready.
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                _rpc_call(port, secret, "aria2.getVersion", [])
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"aria2c RPC daemon exited. Log: {log_path.read_text(errors='replace')}")
                time.sleep(0.2)
        else:
            raise TimeoutError("Timeout menunggu aria2 RPC ready.")

        options: dict[str, object] = {
            "dir": str(dest.parent),
            "out": dest.name,
            "continue": "true",
            "max-connection-per-server": str(connections),
            "split": str(split),
            "min-split-size": min_split_size,
            "file-allocation": "none",
            "allow-overwrite": "true",
            "auto-file-renaming": "false",
        }
        if token:
            options["header"] = [f"Authorization: Bearer {token}"]

        gid = _rpc_call(port, secret, "aria2.addUri", [[url], options])

        last_completed = 0
        bar: tqdm | None = None

        while True:
            status = _rpc_call(
                port,
                secret,
                "aria2.tellStatus",
                [gid, ["status", "totalLength", "completedLength", "downloadSpeed", "errorCode", "errorMessage"]],
            )

            state = status.get("status", "")
            total = int(status.get("totalLength") or 0)
            completed = int(status.get("completedLength") or 0)
            speed = int(status.get("downloadSpeed") or 0)

            if bar is None:
                bar = tqdm(
                    total=total if total > 0 else None,
                    initial=completed,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    mininterval=0.5,
                )
                last_completed = completed
            else:
                if total > 0 and bar.total != total:
                    bar.total = total
                delta = completed - last_completed
                if delta > 0:
                    bar.update(delta)
                    last_completed = completed
                if speed:
                    bar.set_postfix_str(f"{speed / 1024**2:.1f} MiB/s")

            if state == "complete":
                if bar:
                    if total > completed:
                        bar.update(total - completed)
                    bar.close()
                break

            if state == "error":
                if bar:
                    bar.close()
                err = status.get("errorMessage") or "unknown error"
                code = status.get("errorCode") or "?"
                raise RuntimeError(f"aria2 download error code={code}: {err}")

            if state in {"removed"}:
                if bar:
                    bar.close()
                raise RuntimeError(f"aria2 download state={state}")

            time.sleep(poll_interval)

        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"Download selesai tapi file tidak ditemukan/empty: {dest}")

        return dest

    finally:
        try:
            _rpc_call(port, secret, "aria2.shutdown", [], timeout=2)
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        log_f.close()


def download_file(
    url: str,
    dest: str | Path,
    *,
    token: Optional[str] = None,
    force: bool = False,
    backend: str = "aria2",
) -> Path:
    if backend == "aria2":
        try:
            return download_with_aria2_tqdm(url, dest, token=token, force=force)
        except Exception as exc:
            print(f"aria2+tqdm gagal: {exc}")
            print("Fallback ke requests+tqdm.")
            return download_with_requests_tqdm(url, dest, token=token, force=force)
    if backend == "requests":
        return download_with_requests_tqdm(url, dest, token=token, force=force)
    raise ValueError(f"backend tidak dikenal: {backend}")


def extract_archive(archive: str | Path, out_dir: str | Path) -> Path:
    archive = Path(archive)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    else:
        raise ValueError(f"Format archive belum didukung: {archive}")

    return out_dir
