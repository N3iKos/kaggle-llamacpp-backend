# Kaggle llama.cpp CUDA backend

Repo kecil untuk menjalankan `llama.cpp` CUDA + GGUF + ngrok di Kaggle tanpa notebook panjang.

## Perubahan penting

Downloader sekarang memakai **aria2c + tqdm**:

- `aria2c` tetap melakukan parallel segmented download.
- Python membaca progress dari JSON-RPC aria2.
- `tqdm` menampilkan progress bar realtime yang rapi di Kaggle.
- Jika aria2 gagal, fallback otomatis ke `requests + tqdm`.

Aria2 mendukung parallel segmented HTTP download melalui opsi seperti `--split` dan `--max-connection-per-server`, serta bisa dikontrol lewat JSON-RPC.

## Notebook

`notebooks/kaggle_runner.ipynb` hanya punya 4 cell utama:

1. Clone repo, install dependency, install/download `aria2c`, download prebuilt `llama.cpp-cuda`.
2. Download model GGUF dari URL.
3. Konfigurasi runtime, start server background, health check, test response.
4. Start ngrok tunnel dari Kaggle Secret.

## Push ke GitHub

```bash
cd kaggle-llamacpp-backend-aria-tqdm
git init
git add .
git commit -m "initial kaggle llama.cpp backend aria tqdm"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/kaggle-llamacpp-backend.git
git push -u origin main
```

Lalu edit `REPO_URL` di notebook:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/kaggle-llamacpp-backend.git"
```

## Kaggle Secret untuk ngrok

Buat secret bernama:

```text
NGROK_AUTHTOKEN
```

Isinya authtoken dari dashboard ngrok, bukan API key.

## Model default

Default notebook memakai Gemma 4 26B Q4 GGUF. Untuk tes cepat, ganti ke model 7B/8B Q4.
