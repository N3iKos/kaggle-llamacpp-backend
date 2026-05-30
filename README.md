# Kaggle llama.cpp backend final

Notebook pendek untuk llama.cpp CUDA GGUF di Kaggle.

Cell utama:
1. Setup repo + prebuilt llama.cpp CUDA.
2. Download model GGUF + optional mmproj dengan aria2+tqdm.
3. Config server, alias model, /health, /v1/models, chat test, optional vision test.
4. Start ngrok.

Aplikasi lain:
- Base URL: `https://NGROK_URL/v1`
- Model ID: `local`
- API key: bebas, kecuali `api_key` diaktifkan di config.

Vision:
Isi `MMPROJ_URL` di Cell 2. Path mmproj otomatis tersimpan dan dipakai di Cell 3.
