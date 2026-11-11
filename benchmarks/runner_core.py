import os
import re
import json
import time
import requests
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

def unload_ollama_vram():
    """Mengosongkan VRAM Ollama secara paksa."""
    try:
        payload = {"model": MODEL_NAME, "keep_alive": 0}
        requests.post(OLLAMA_URL, json=payload, timeout=10)
        time.sleep(5)
        print("  🧹 [CORE] VRAM berhasil dikosongkan.")
    except Exception as e:
        print(f"  ⚠️ [CORE] Gagal mengosongkan VRAM: {e}")

def query_ollama_safe(prompt, system_prompt="", retries=3, timeout=180):
    """Kueri Ollama dengan proteksi timeout, retry, dan keep_alive."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "keep_alive": "-1",  # Tahan di VRAM selama iterasi item berjalan
        "options": {
            "temperature": 0.0,
            "num_predict": 512
        }
    }
    
    for attempt in range(1, retries + 1):
        try:
            start_time = time.time()
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                text_out = response.json().get("response", "")
                return text_out, latency_ms
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Attempt {attempt}/{retries} timeout/error ({e}). Retrying in 3s...")
            time.sleep(3)
            
    return "SYS_TIMEOUT", 0.0

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
    """Ambil daftar ID yang sudah diproses untuk auto-resume."""
    if not os.path.isfile(file_path):
        return set()
    try:
        df = pd.read_csv(file_path)
        if id_column in df.columns:
            return set(df[id_column].astype(str).tolist())
    except Exception:
        pass
    return set()