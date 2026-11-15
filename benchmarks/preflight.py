import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.llm import MODEL, BASE, preflight  # noqa: E402

print(f"🔎 Preflight Ollama @ {BASE} | model: {MODEL}")
ok, msg = preflight()
print(("✅ " if ok else "❌ ") + msg)
sys.exit(0 if ok else 1)
