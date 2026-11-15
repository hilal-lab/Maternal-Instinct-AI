"""
Klien Ollama untuk benchmark.

Perbaikan dibanding runner_core lama:
  * keep_alive berupa ANGKA (-1), bukan string "-1" (ditolak Ollama => HTTP 400)
  * error HTTP dicatat dan dikembalikan (tidak ditelan jadi "SYS_TIMEOUT")
  * output terstruktur via parameter `format` (JSON schema) -> kegagalan format tidak lagi
    dihitung sebagai "intersepsi"
  * `done_reason == "length"` terdeteksi sebagai output terpotong
  * berhenti (OllamaDown) setelah N kegagalan beruntun, supaya run tidak "sukses palsu"
"""
import os
import sys
import time
from dataclasses import dataclass

import requests

BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not BASE.startswith("http"):
    BASE = "http://" + BASE
MODEL = os.environ.get("LACF_MODEL", "llama3.2:3b")
GUARD_MODEL = os.environ.get("LACF_GUARD_MODEL", "llama-guard3")  # nama dasar; tag terpasang di-resolve otomatis
_resolved_guard = None
_schema_supported = True  # turun ke format="json" bila Ollama lama menolak JSON-schema
MAX_CONSECUTIVE_FAILURES = 5

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "study_work_h": {"type": "number"},
        "sleep_h": {"type": "number"},
        "rest_h": {"type": "number"},
        "weekly_load_h": {"type": "number"},
    },
    "required": ["study_work_h", "sleep_h", "rest_h", "weekly_load_h"],
}


class OllamaDown(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str = ""
    latency_ms: float = 0.0
    ok: bool = False
    error: str = ""
    truncated: bool = False


_fail_streak = 0


def _register(ok, err=""):
    global _fail_streak
    if ok:
        _fail_streak = 0
        return
    _fail_streak += 1
    if _fail_streak >= MAX_CONSECUTIVE_FAILURES:
        raise OllamaDown(f"Ollama gagal {MAX_CONSECUTIVE_FAILURES}x berturut-turut. Error terakhir: {err}")


def note_success():
    """Dipanggil setelah panggilan lewat klien backend (Layer 1/2) berhasil."""
    _register(True)


def note_failure_if_backend(exc):
    """Hitung kegagalan klien backend (mode LACF_STRICT) ke streak; bisa melempar OllamaDown."""
    if str(exc).startswith("OLLAMA_FAILURE"):
        _register(False, str(exc))


def generate(prompt, system="", schema=None, temperature=0.0, num_predict=512, seed=0,
             timeout=240, retries=2, model=None):
    payload = {
        "model": model or MODEL, "prompt": prompt, "stream": False, "keep_alive": -1,
        "options": {"temperature": temperature, "num_predict": num_predict, "seed": seed},
    }
    if system:
        payload["system"] = system
    global _schema_supported
    if schema:
        payload["format"] = schema if _schema_supported else "json"
    last = ""
    max_attempts = retries + (1 if (schema and _schema_supported) else 0)  # +1 untuk penurunan format
    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.perf_counter()
            r = requests.post(f"{BASE}/api/generate", json=payload, timeout=timeout)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                body = r.json()
                _register(True)
                return LLMResult(body.get("response", ""), dt, True, "", body.get("done_reason") == "length")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 400 and schema and _schema_supported:
                # Ollama < 0.5 tidak mendukung JSON-schema pada `format`; turun ke mode JSON biasa.
                _schema_supported = False
                payload["format"] = "json"
                print("  ⚠️ Ollama menolak format=schema; memakai format=\"json\" (skema dijaga lewat prompt + validasi).")
                continue
            if r.status_code in (400, 404):
                break
        except requests.exceptions.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2)
    _register(False, last)
    return LLMResult(error=last)


def chat(messages, model, temperature=0.0, num_predict=64, timeout=120):
    """/api/chat — dipakai untuk Llama Guard (template chat model yang menentukan format prompt)."""
    payload = {"model": model, "messages": messages, "stream": False, "keep_alive": -1,
               "options": {"temperature": temperature, "num_predict": num_predict}}
    try:
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}/api/chat", json=payload, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            _register(True)
            return LLMResult(r.json().get("message", {}).get("content", ""), dt, True)
        err = f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.RequestException as e:
        err = f"{type(e).__name__}: {e}"
    _register(False, err)
    return LLMResult(error=err)


def model_available(name):
    try:
        r = requests.get(f"{BASE}/api/tags", timeout=10)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(n == name or n.startswith(name + ":") or n.startswith(name) for n in names), names
    except Exception:
        return False, []


def resolve_guard_model():
    """Cari tag Llama Guard yang BENAR-BENAR terpasang (mis. llama-guard3:1b). 'llama-guard3' polos = ':latest' (8b)."""
    global _resolved_guard
    if _resolved_guard:
        return _resolved_guard
    _, names = model_available(GUARD_MODEL)
    exact = [n for n in names if n == GUARD_MODEL]
    cands = exact or [n for n in names if n.startswith(GUARD_MODEL)] or [n for n in names if n.lower().startswith("llama-guard")]
    _resolved_guard = cands[0] if cands else None
    return _resolved_guard


def guard_check(user_prompt, assistant_reply=None):
    """
    Llama Guard 3 via Ollama. Return (verdict, categories, latency_ms, error).
    verdict in {"safe","unsafe",None}. None = tidak bisa dinilai (jangan dihitung aman).
    """
    msgs = [{"role": "user", "content": user_prompt}]
    if assistant_reply is not None:
        msgs.append({"role": "assistant", "content": assistant_reply})
    gm = resolve_guard_model()
    if not gm:
        return None, "", 0.0, "model Llama Guard tidak terpasang"
    res = chat(msgs, gm)
    if not res.ok:
        return None, "", res.latency_ms, res.error
    out = res.text.strip().lower()
    first = out.splitlines()[0].strip() if out else ""
    if first.startswith("unsafe"):
        cats = out.splitlines()[1].strip() if len(out.splitlines()) > 1 else ""
        return "unsafe", cats, res.latency_ms, ""
    if first.startswith("safe"):
        return "safe", "", res.latency_ms, ""
    return None, "", res.latency_ms, f"keluaran guard tak dikenali: {res.text[:60]!r}"


def preflight(need_guard=False):
    """(ok, pesan). Cek koneksi, model, dan satu generate terstruktur nyata."""
    try:
        r = requests.get(f"{BASE}/api/tags", timeout=10)
    except requests.exceptions.RequestException as e:
        return False, (f"Tidak bisa terhubung ke Ollama di {BASE} ({e}). Di WSL2 dengan Ollama di Windows, "
                       "set OLLAMA_HOST ke IP host Windows.")
    if r.status_code != 200:
        return False, f"/api/tags -> HTTP {r.status_code}"
    ok_main, names = model_available(MODEL)
    if not ok_main:
        return False, f"Model '{MODEL}' belum ada (terpasang: {names}). Jalankan: ollama pull {MODEL}"
    g = generate('Balas JSON {"study_work_h":1,"sleep_h":8,"rest_h":1,"weekly_load_h":7}', schema=PLAN_SCHEMA,
                 num_predict=64, retries=1)
    if not g.ok:
        return False, f"Generate terstruktur gagal: {g.error}"
    msg = "Ollama siap (generate terstruktur OK)."
    gm = resolve_guard_model()
    if gm:
        v, _, _, err = guard_check("Hello, how are you?")  # uji panggilan guard sungguhan, bukan sekadar cek nama
        ok_guard = v is not None
        msg += f" Llama Guard: {gm} {'OK' if ok_guard else 'TERPASANG TAPI GAGAL (' + err + ') -> C5 NOT_RUN'}."
    else:
        ok_guard = False
        msg += f" Llama Guard: TIDAK terpasang (C5 NOT_RUN). Pasang dengan: ollama pull {GUARD_MODEL}:1b"
    if need_guard and not ok_guard:
        return False, "Llama Guard dibutuhkan tetapi tidak berfungsi. " + msg
    return True, msg


if __name__ == "__main__":
    ok, m = preflight()
    print(("✅ " if ok else "❌ ") + m)
    sys.exit(0 if ok else 1)
