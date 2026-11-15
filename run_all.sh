#!/bin/bash
# Jalankan dari root repo:  ./run_all.sh [run_id]        (smoke test: LIMIT=5 ./run_all.sh smoke)
PYTHON_BIN="python"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python3"
RUN_ID="${1:-main}"
LIMIT="${LIMIT:-0}"
OUT="data/results"

echo "🚀 LACF BENCHMARK SUITE  (run: $RUN_ID, limit: $LIMIT)"
echo "=============================================================================="

echo "🔹 [0/3] Preflight (Ollama, model, generate terstruktur)..."
$PYTHON_BIN -m benchmarks.llm || { echo "❌ Preflight gagal. Dibatalkan."; exit 1; }

echo "🔹 [1/3] Validasi dataset (prompt tidak boleh kosong)..."
$PYTHON_BIN -m benchmarks.datasets || { echo "❌ Dataset tidak valid. Dibatalkan."; exit 1; }

echo "🔹 [2/3] Menjalankan D1-D5..."
$PYTHON_BIN -m benchmarks.run_suite --datasets d1,d2,d3,d4,d5 --run-id "$RUN_ID" --out "$OUT" --limit "$LIMIT"
RC=$?
if [ $RC -ne 0 ]; then
  echo "❌ Benchmark berhenti (exit $RC). Hasil parsial tersimpan; jalankan lagi dengan run_id yang sama untuk melanjutkan."
  exit $RC
fi

echo "🔹 [3/3] Ringkasan metrik..."
$PYTHON_BIN -m benchmarks.metrics --run "$OUT/$RUN_ID"

echo "🧹 Membebaskan VRAM..."
curl -s -X POST "${OLLAMA_HOST:-http://localhost:11434}/api/generate" -d '{"model":"llama3.2:3b","keep_alive":0}' >/dev/null 2>&1
echo "🎉 Selesai. Hasil: $OUT/$RUN_ID/ (results_d*.csv, summary.md, summary.json, run_meta.json)"
