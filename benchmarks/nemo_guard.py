"""NeMo Guardrails adapter for the LACF benchmark."""

import os
import time
from pathlib import Path


class NemoGuardError(RuntimeError):
    pass


class NemoGuard:
    def __init__(self, config_dir=None):
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as exc:
            raise NemoGuardError(
                "NeMo Guardrails belum terpasang. "
                "Jalankan pip install -r benchmarks/requirements-nemo.txt"
            ) from exc

        default_dir = (
            Path(__file__).resolve().parent / "nemo_config"
        )

        self.config_dir = Path(
            config_dir
            or os.environ.get("LACF_NEMO_CONFIG", default_dir)
        )

        if not self.config_dir.is_dir():
            raise NemoGuardError(
                f"Folder konfigurasi NeMo tidak ditemukan: {self.config_dir}"
            )

        if not (self.config_dir / "config.yml").is_file():
            raise NemoGuardError("config.yml tidak ditemukan")

        if not (self.config_dir / "prompts.yml").is_file():
            raise NemoGuardError("prompts.yml tidak ditemukan")

        try:
            config = RailsConfig.from_path(str(self.config_dir))
            self.rails = LLMRails(config)
        except Exception as exc:
            raise NemoGuardError(
                f"Gagal memuat konfigurasi NeMo: {type(exc).__name__}: {exc}"
            ) from exc

    def generate(self, prompt):
        if not prompt or not str(prompt).strip():
            raise ValueError("Prompt tidak boleh kosong")

        started = time.perf_counter()

        try:
            response = self.rails.generate(
                messages=[
                    {
                        "role": "user",
                        "content": str(prompt),
                    }
                ]
            )

            if not isinstance(response, dict):
                raise NemoGuardError(
                    f"Format respons NeMo tidak valid: {type(response).__name__}"
                )

            content = response.get("content", "")

            if not isinstance(content, str) or not content.strip():
                raise NemoGuardError(
                    "NeMo menghasilkan respons kosong"
                )

            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            return content, latency_ms

        except NemoGuardError:
            raise
        except Exception as exc:
            raise NemoGuardError(
                f"NeMo inference gagal: {type(exc).__name__}: {exc}"
            ) from exc

    def preflight(self):
        """Uji bahwa NeMo dan model lokal benar-benar dapat menghasilkan respons."""
        content, latency_ms = self.generate(
            "Say hello briefly."
        )

        return {
            "status": "READY",
            "latency_ms": round(latency_ms, 2),
            "response_preview": content[:120],
        }