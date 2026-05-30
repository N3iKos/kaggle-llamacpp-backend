# Kaggle llama.cpp CUDA backend

Repo kecil untuk menjalankan `llama.cpp` CUDA + GGUF + ngrok di Kaggle tanpa notebook panjang.

Target awal:
- Kaggle GPU 2x Tesla T4.
- Prebuilt `llama.cpp` CUDA 12.8 dari `ai-dock/llama.cpp-cuda`.
- Model GGUF dari URL Hugging Face.
- `llama-server` berjalan di background.
- Health check dan test OpenAI-compatible endpoint.
- Ngrok tunnel memakai Kaggle Secret `NGROK_AUTHTOKEN`.

## Push ke GitHub

```bash
cd kaggle-llamacpp-backend
git init
git add .
git commit -m "initial kaggle llama.cpp backend"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/kaggle-llamacpp-backend.git
git push -u origin main
```

Setelah repo ada di GitHub, buka `notebooks/kaggle_runner.ipynb`, lalu ubah:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/kaggle-llamacpp-backend.git"
```

## Secret ngrok

Di Kaggle Notebook:
1. Buka **Add-ons > Secrets**.
2. Tambah secret bernama persis `NGROK_AUTHTOKEN`.
3. Isi dengan authtoken dari dashboard ngrok, bukan API key.
4. Aktifkan akses secret untuk notebook.

## Cell notebook

Notebook hanya 4 cell utama:

1. Clone repo + install dependency + download prebuilt `llama.cpp-cuda`.
2. Download model GGUF dengan progress `tqdm`.
3. Config server + start background + health check + test response.
4. Start ngrok tunnel dari Kaggle Secret.

## Model URL

Default notebook memakai Gemma 4 26B Q4 GGUF. Untuk tes cepat, ganti ke model 7B/8B Q4.

## Catatan Gemma 4

Gemma 4 dapat mengeluarkan tag channel seperti `<|channel>thought`. Modul ini mencoba mengambil `__verbose.content` lebih dulu dan membersihkan tag lewat `clean_gemma4_text()`.
