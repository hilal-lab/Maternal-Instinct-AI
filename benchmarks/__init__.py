"""LACF benchmark suite."""
import sys

# Konsol Windows (cp1252) gagal mencetak emoji -> paksa UTF-8 agar run tidak crash saat print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
