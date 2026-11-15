# Jalankan dari root repo (PowerShell):
#   .\run_all.ps1                         # run penuh, run_id = main (resume otomatis jika diulang)
#   .\run_all.ps1 -RunId smoke -Limit 5   # smoke test: 5 item per dataset
#   .\run_all.ps1 -Datasets "d1,d4"       # hanya sebagian dataset
param(
    [string]$RunId = "main",
    [int]$Limit = 0,
    [string]$Datasets = "d1,d2,d3,d4,d5"
)

$env:PYTHONUTF8 = "1"
$py = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $py = "py" }
if (-not $py) { Write-Host "[ERROR] Python tidak ditemukan. Aktifkan venv dulu."; exit 1 }

$Out = "data/results"
$OllamaBase = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://localhost:11434" }
if ($OllamaBase -notmatch "^http") { $OllamaBase = "http://$OllamaBase" }

Write-Host "LACF BENCHMARK SUITE  (run: $RunId, limit: $Limit, datasets: $Datasets)"
Write-Host "=============================================================================="

Write-Host "[0/3] Preflight (Ollama, model, generate terstruktur, Llama Guard)..."
& $py -m benchmarks.llm
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Preflight gagal. Dibatalkan."; exit 1 }

Write-Host "[1/3] Validasi dataset (prompt tidak boleh kosong)..."
& $py -m benchmarks.datasets
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Dataset tidak valid. Dibatalkan."; exit 1 }

Write-Host "[2/3] Menjalankan benchmark..."
& $py -m benchmarks.run_suite --datasets "$Datasets" --run-id $RunId --out $Out --limit $Limit
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Host "[ERROR] Benchmark berhenti (exit $rc). Hasil parsial tersimpan; jalankan lagi dengan -RunId $RunId untuk melanjutkan."
    exit $rc
}

Write-Host "[3/3] Ringkasan metrik..."
& $py -m benchmarks.metrics --run "$Out/$RunId"

Write-Host "Membebaskan VRAM..."
try {
    Invoke-RestMethod -Method Post -Uri "$OllamaBase/api/generate" -ContentType "application/json" `
        -Body '{"model":"llama3.2:3b","keep_alive":0}' | Out-Null
} catch { }

Write-Host "SELESAI. Hasil: $Out/$RunId/  (results_d*.csv, summary.md, summary.json, run_meta.json)"
