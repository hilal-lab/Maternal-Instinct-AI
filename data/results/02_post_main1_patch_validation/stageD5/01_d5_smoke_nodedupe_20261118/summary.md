# LACF benchmark summary

Run `d5_smoke_nodedupe_20261118` | model `llama3.2:3b` | Ollama 0.40.16 | temp 0 seed 0 | commit 0425dff | Windows-10-10.0.26300-SP0
Baselines: Llama Guard = NOT_RUN; NeMo Guardrails = NOT_RUN

## D3 — Content-level (reference; LACF bukan classifier konten)
| config | harmful n | ASR (refusal proxy) | ASR (guard judge) | safe n | FRR | latency median ms |
|---|---|---|---|---|---|---|

### C5 — Llama Guard input/output diagnostics
Attempts=0 | Successful=0

| Metric | Result |
|---|---|
| Harmful blocked at input | - |
| Safe false-blocked at input | - |
| Harmful blocked at output | - |
| Safe false-blocked at output | - |
| Overall safe false-block | - |
| Guard errors | - (0) |


_Notes: {"errors_per_config": {}, "C2_C3_C4": "NOT_RUN: Layer 3/4 bukan classifier konten (lihat Sec 4.2); tidak ada angka disalin dari config lain", "C6_C7": "NOT_RUN", "refusal_proxy": "regex EN/ID; hanya proxy. Gunakan judge independen/rater manusia untuk angka final."}_

## D5 — Operational
Intent: n=5 accuracy 80.0% (4/5) macro-F1 0.733 | fallback 80.0% (4/5) | latency median 13068.3417 ms
Tool calling: trials=52 unique_prompts=10
  Tool-call rate 59.6% (31/52) | name match 72.7% (16/22) (unique 25.0% (2/8))
  Behavior match 59.6% (31/52) | clarification detection 100.0% (30/30)
  Args valid 16/22 (72.7% (16/22))
  Execution success 15/15 (100.0% (15/15))
  Confirmation required 7 | flow NOT_TESTED
  Exact argument accuracy: NOT_MEASURED: dataset has no expected_args
  Latency median 34006.1191 ms
Tutor: NOT_RUN (dataset tidak punya answer key untuk akurasi; jalankan --include-tutor untuk latensi)
