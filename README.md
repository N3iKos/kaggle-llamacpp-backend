# Kaggle llama.cpp API Config Backend

Config-driven backend untuk menjalankan `llama.cpp` CUDA di Kaggle.

Fokus:
- OpenAI-compatible API.
- `/v1/models` rapi lewat `--alias`.
- API key lewat `--api-key`.
- Preset context: `safe`, `balanced`, `long_context_20k`, `vision_safe`.
- Text mode: `model.gguf`.
- Vision mode: `model.gguf + mmproj.gguf`.
- Downloader `aria2c + tqdm`.
- Ngrok tunnel dari Kaggle Secret.

## Feasibility

Implementasi ini mengikuti fitur resmi `llama-server`:
- OpenAI-compatible `/v1/chat/completions`.
- Multimodal melalui `image_url` content part bila model mendukung.
- `/v1/models` dan `--alias`.
- `--api-key`.
- `--mmproj` untuk multimodal projector.

Vision mode membutuhkan pasangan model vision GGUF dan mmproj GGUF yang cocok. Model teks biasa tidak otomatis bisa melihat gambar hanya karena diberi mmproj lain.

## Notebook

`notebooks/kaggle_runner.ipynb` tetap 4 cell utama:

1. Clone repo + install dependency + download prebuilt `llama.cpp-cuda`.
2. Load config + download assets.
3. Start server + health check + `/v1/models` check + chat test.
4. Start ngrok tunnel.

## Push ke GitHub

```bash
cd kaggle-llamacpp-api-config
git init
git add .
git commit -m "config driven kaggle llama.cpp backend"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/kaggle-llamacpp-api-config.git
git push -u origin main
```

## Kaggle Secrets

Untuk ngrok:
- buat secret `ngrok` atau `NGROK_AUTHTOKEN`;
- pastikan checkbox secret dicentang di notebook;
- gunakan label yang sama di `start_ngrok_tunnel(secret_name=...)`.

## Text config

Lihat:

```text
configs/config.text.example.yaml
```

## Vision config

Lihat:

```text
configs/config.vision.example.yaml
```

Ganti `model_url` dan `mmproj_url` dengan pasangan model vision dan projector yang benar.
