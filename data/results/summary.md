# LACF benchmark summary

Run `?` | model `?` | Ollama ? | temp None seed None | commit ? | ?
Baselines: Llama Guard = ?; NeMo Guardrails = NOT_RUN

## D3 — Content-level (reference; LACF bukan classifier konten)
| config | harmful n | ASR (refusal proxy) | ASR (guard judge) | safe n | FRR | latency median ms |
|---|---|---|---|---|---|---|

_Notes: {"errors_per_config": {}, "C2_C3_C4": "NOT_RUN: Layer 3/4 bukan classifier konten (lihat Sec 4.2); tidak ada angka disalin dari config lain", "C6_C7": "NOT_RUN", "refusal_proxy": "regex EN/ID; hanya proxy. Gunakan judge independen/rater manusia untuk angka final."}_
