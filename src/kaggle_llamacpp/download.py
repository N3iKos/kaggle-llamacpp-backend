
import random, shutil, socket, subprocess, tarfile, time, urllib.parse, zipfile
from pathlib import Path
import requests
from tqdm.auto import tqdm

def filename_from_url(url):
    name = Path(urllib.parse.urlparse(url).path).name
    if not name: raise ValueError("Tidak bisa ambil nama file dari URL.")
    return name

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _rpc(port, secret, method, params=None):
    r = requests.post(f"http://127.0.0.1:{port}/jsonrpc", json={
        "jsonrpc":"2.0","id":str(random.random()),"method":method,
        "params":[f"token:{secret}"] + (params or [])
    }, timeout=10)
    r.raise_for_status()
    j = r.json()
    if "error" in j: raise RuntimeError(j["error"])
    return j.get("result")

def download_requests(url, dest, token=None, force=False):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size and not force:
        print(f"File sudah ada, skip: {dest}")
        return dest
    h = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=h, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0) or None
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name) as bar:
            for c in r.iter_content(8*1024*1024):
                if c:
                    f.write(c); bar.update(len(c))
    return dest

def download_aria2(url, dest, token=None, force=False):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size and not force:
        print(f"File sudah ada, skip: {dest}")
        return dest
    if force:
        dest.unlink(missing_ok=True); Path(str(dest)+".aria2").unlink(missing_ok=True)
    if not shutil.which("aria2c"):
        raise FileNotFoundError("aria2c not found")
    port, secret = _free_port(), f"s{random.randrange(10**12)}"
    log = dest.parent / f".aria2-{dest.name}.log"
    proc = subprocess.Popen(["aria2c","--enable-rpc=true","--rpc-listen-all=false",f"--rpc-listen-port={port}",f"--rpc-secret={secret}","--console-log-level=warn","--summary-interval=0","--quiet=true"], stdout=open(log,"w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try: _rpc(port, secret, "aria2.getVersion"); break
            except Exception: time.sleep(.2)
        opt = {"dir":str(dest.parent),"out":dest.name,"continue":"true","max-connection-per-server":"16","split":"16","min-split-size":"64M","file-allocation":"none","allow-overwrite":"true","auto-file-renaming":"false"}
        if token: opt["header"]=[f"Authorization: Bearer {token}"]
        gid = _rpc(port, secret, "aria2.addUri", [[url], opt])
        bar = None; last = 0
        while True:
            st = _rpc(port, secret, "aria2.tellStatus", [gid, ["status","totalLength","completedLength","downloadSpeed","errorCode","errorMessage"]])
            total, done, speed = int(st.get("totalLength") or 0), int(st.get("completedLength") or 0), int(st.get("downloadSpeed") or 0)
            if bar is None:
                bar = tqdm(total=total or None, initial=done, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name); last = done
            else:
                if total and bar.total != total: bar.total = total
                if done > last: bar.update(done-last); last = done
                if speed: bar.set_postfix_str(f"{speed/1024**2:.1f} MiB/s")
            if st["status"] == "complete":
                bar.close(); break
            if st["status"] == "error":
                bar.close(); raise RuntimeError(st.get("errorMessage") or st.get("errorCode"))
            time.sleep(.5)
        return dest
    finally:
        try: _rpc(port, secret, "aria2.shutdown")
        except Exception: pass
        if proc.poll() is None:
            proc.terminate()

def download_file(url, dest, token=None, force=False, backend="aria2"):
    if backend == "aria2":
        try: return download_aria2(url, dest, token=token, force=force)
        except Exception as e:
            print("aria2+tqdm gagal:", e, "\nFallback requests+tqdm.")
            return download_requests(url, dest, token=token, force=force)
    return download_requests(url, dest, token=token, force=force)

def extract_archive(archive, out_dir):
    archive, out_dir = Path(archive), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith((".tar.gz",".tgz")):
        with tarfile.open(archive, "r:gz") as t: t.extractall(out_dir)
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z: z.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive: {archive}")
    return out_dir
