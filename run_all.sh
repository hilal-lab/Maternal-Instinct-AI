#!/bin/bash

# ==============================================================================
# LACF AUTOMATED BENCHMARK PIPELINE (D1 -> D5)
# ==============================================================================

PYTHON_BIN="./.venv/bin/python"  # Ganti dengan .venv/bin/python jika menggunakan virtualenv
export PYTHONPATH=$PYTHONPATH:$(pwd)/benchmarks

clean_vram() {
    echo ""
    echo "🧹 [ORCHESTRATOR] Unloading Llama 3.2 3B from VRAM..."
    curl -s http://localhost:11434/api/generate -d '{"model": "llama3.2:3b", "keep_alive": 0}' > /dev/null
    sleep 5
    echo "✅ [ORCHESTRATOR] Resource cleanup completed."
    echo "=============================================================================="
    echo ""
}

echo "🚀 LAUNCHING FULL LACF BENCHMARK SUITE"
echo "=============================================================================="

# 1. Dataset D1
echo "🔹 [1/5] Executing Module D1 (Parametric Workload)..."
$PYTHON_BIN benchmarks/run_d1_parametric.py
if [ $? -ne 0 ]; then echo "❌ D1 Failed!"; exit 1; fi
clean_vram

# 2. Dataset D3
echo "🔹 [2/5] Executing Module D3 (Direct Safety & Harm)..."
$PYTHON_BIN benchmarks/run_d3_safety.py
if [ $? -ne 0 ]; then echo "❌ D3 Failed!"; exit 1; fi
clean_vram

# 3. Dataset D2
echo "🔹 [3/5] Executing Module D2 (Affective Validation)..."
$PYTHON_BIN benchmarks/run_d2_affective.py
if [ $? -ne 0 ]; then echo "❌ D2 Failed!"; exit 1; fi
clean_vram

# 4. Dataset D4
echo "🔹 [4/5] Executing Module D4 (Prompt Injection)..."
$PYTHON_BIN benchmarks/run_d4_injection.py
if [ $? -ne 0 ]; then echo "❌ D4 Failed!"; exit 1; fi
clean_vram

# 5. Dataset D5
echo "🔹 [5/5] Executing Module D5 (Operational Task Accuracy)..."
$PYTHON_BIN benchmarks/run_d5_operational.py
if [ $? -ne 0 ]; then echo "❌ D5 Failed!"; exit 1; fi
clean_vram

echo "🎉 ALL BENCHMARK MODULES (D1-D5) EXECUTED SUCCESSFULLY!"