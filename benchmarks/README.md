# LACF benchmark (v2)

## Persiapan
```powershell
pip install -r backend/requirements.txt pytest
ollama pull llama3.2:3b
ollama pull llama-guard3:1b      # baseline Llama Guard (opsional; tanpa ini C5 = NOT_RUN). Tag terpasang dideteksi otomatis.
pytest -q tests/test_benchmarks.py   # tes instrumen (tanpa Ollama)
```

## Menjalankan
Windows PowerShell:
```powershell
.\run_all.ps1 -RunId smoke -Limit 5     # smoke test 5 item/dataset (cek dulu sebelum run penuh)
.\run_all.ps1 -RunId main               # run penuh; mengulang dengan RunId sama = resume
```
Linux/macOS/WSL: `LIMIT=5 ./run_all.sh smoke` lalu `./run_all.sh main`.

Hasil: `data/results/<run_id>/{results_d1..d5.csv, summary.md, summary.json, run_meta.json}`.

## Yang BELUM ada (ditandai NOT_RUN, bukan diisi angka)
NeMo Guardrails (C6/C7); judge LLM/rater manusia untuk D4 G4-G6 dan EPITOME D2; D1 dengan proposer LLM (UPDR end-to-end);
Tier A (Llama 3.1 8B: set `LACF_MODEL`); akurasi tutor D5 (dataset tidak punya answer key).
