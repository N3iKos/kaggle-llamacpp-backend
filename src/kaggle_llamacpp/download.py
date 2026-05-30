from __future__ import annotations

import tarfile
import urllib.parse
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm.auto import tqdm

from .utils import chmod_exec


def filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError("Tidak bisa menentukan nama file dari URL. Isi output_name manual.")
    return name


def download_with_tqdm(
    url: str,
    dest: str | Path,
    *,
    token: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    force: bool = False,
    resume: bool = True,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Single-stream downloader dengan tqdm. Progress lebih jelas di Kaggle daripada aria2 output."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"File sudah ada, skip: {dest} ({dest.stat().st_size / 1024**3:.2f} GiB)")
        return dest

    req_headers = dict(headers or {})
    if token:
        req_headers["Authorization"] = f"Bearer {token}"

    mode = "wb"
    existing = 0
    if resume and dest.exists() and not force:
        existing = dest.stat().st_size
        if existing > 0:
            req_headers["Range"] = f"bytes={existing}-"
            mode = "ab"

    with requests.get(url, headers=req_headers, stream=True, allow_redirects=True, timeout=60) as r:
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
