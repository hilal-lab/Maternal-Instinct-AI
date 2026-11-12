import os
import re
import json
import time
import sys
import requests
import pandas as pd

OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"
MODEL_NAME = "llama3.2:3b"

# Berhenti total jika Ollama gagal berturut-turut (mencegah "sukses palsu" dengan semua SYS_TIMEOUT)
MAX_CONSECUTIVE_FAILURES = 5
_consecutive_failures = 0


def unload_ollama_vram():
    """Mengosongkan VRAM Ollama secara paksa."""
    try:
        requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "keep_alive": 0}, timeout=10)
        time.sleep(5)
        print("  🧹 [CORE] VRAM berhasil dikosongkan.")
    except Exception as e:
        print(f"  ⚠️ [CORE] Gagal mengosongkan VRAM: {e}")


def query_ollama_safe(prompt, system_prompt="", retries=3, timeout=180):
    """Kueri Ollama dengan retry. Error HTTP kini DICATAT (bukan disembunyikan)."""
    global _consecutive_failures
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "keep_alive": -1,  # ANGKA, bukan string "-1" (string tanpa satuan ditolak Ollama: HTTP 400)
        "options": {"temperature": 0.0, "num_predict": 512},
    }

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            start_time = time.time()
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                _consecutive_failures = 0
                return response.json().get("response", ""), latency_ms

            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"  ⚠️ Attempt {attempt}/{retries} gagal -> {last_error}")
            if response.status_code in (400, 404):  # permanen: retry percuma
                break
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"  ⚠️ Attempt {attempt}/{retries} timeout/error ({e}). Retrying in 3s...")
            time.sleep(3)

    _consecutive_failures += 1
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        print(f"\n❌ [FATAL] Ollama gagal {MAX_CONSECUTIVE_FAILURES}x berturut-turut. Error terakhir: {last_error}")
        print("   Jalankan: python benchmarks/preflight.py  untuk diagnosis. Menghentikan benchmark.")
        sys.exit(2)
    return "SYS_TIMEOUT", 0.0


def preflight():
    """Cek koneksi, ketersediaan model, dan satu generate nyata. Return (ok, pesan)."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    except requests.exceptions.RequestException as e:
        return False, (f"Tidak bisa terhubung ke Ollama di {OLLAMA_BASE} ({e}). "
                       "Di WSL2 dengan Ollama di Windows, set OLLAMA_HOST ke IP host Windows "
                       "atau aktifkan mirrored networking.")
    if r.status_code != 200:
        return False, f"/api/tags -> HTTP {r.status_code}"
    names = [m.get("name", "") for m in r.json().get("models", [])]
    if not any(n == MODEL_NAME or n.startswith(MODEL_NAME) for n in names):
        return False, f"Model '{MODEL_NAME}' belum ada. Terpasang: {names}. Jalankan: ollama pull {MODEL_NAME}"
    try:
        g = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": "Balas hanya: OK", "stream": False,
            "keep_alive": -1, "options": {"temperature": 0.0, "num_predict": 8}}, timeout=300)
    except requests.exceptions.RequestException as e:
        return False, f"Generate percobaan gagal: {e}"
    if g.status_code != 200:
        return False, f"Generate percobaan -> HTTP {g.status_code}: {g.text[:200]}"
    return True, "Ollama siap."


def extract_json(text):
    """Ekstraksi objek JSON dari teks luaran SLM menggunakan Regex."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return None


def save_checkpoint(file_path, row_data):
    """Simpan hasil per item secara real-time (Append CSV)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([row_data])
    df.to_csv(file_path, mode='a', index=False, header=not file_exists)


def get_completed_ids(file_path, id_column="item_id"):
    """ID yang sudah SUKSES diproses. Baris SYS_TIMEOUT dibuang (dibackup ke .bak) agar diulang."""
    if not os.path.isfile(file_path):
        return set()
    try:
        df = pd.read_csv(file_path)
        if "status_l3" in df.columns:
            bad = df["status_l3"].astype(str) == "SYS_TIMEOUT"
            if bad.any():
                os.replace(file_path, file_path + ".bak")
                df[~bad].to_csv(file_path, index=False)
                print(f"  ♻️ {int(bad.sum())} baris SYS_TIMEOUT lama dibuang dari checkpoint (backup: {file_path}.bak)")
                df = df[~bad]
        if id_column in df.columns:
            return set(df[id_column].astype(str).tolist())
    except Exception:
        pass
    return set()
