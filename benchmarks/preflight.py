import sys
from runner_core import preflight, MODEL_NAME, OLLAMA_BASE

print(f"🔎 Preflight Ollama @ {OLLAMA_BASE} | model: {MODEL_NAME}")
ok, msg = preflight()
print(("✅ " if ok else "❌ ") + msg)
sys.exit(0 if ok else 1)
