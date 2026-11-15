"""Penyimpanan hasil per (config, item). Menyimpan SEMUA yang diperlukan untuk audit (raw output, plan, alasan)."""
import csv
import json
import os

COLUMNS = ["run_id", "dataset", "config", "item_id", "category", "cluster", "gold", "pred", "status",
           "is_harmful", "intercepted", "delivered_unsafe", "hijacked", "source", "latency_ms",
           "error", "raw", "extra"]
RETRY_STATUSES = {"ERROR"}  # baris dengan status ini diulang saat resume


class Store:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done = self._load_done()

    def _load_done(self):
        if not os.path.isfile(self.path):
            return set()
        keep, rows = [], []
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        bad = [r for r in rows if r.get("status") in RETRY_STATUSES]
        if bad:  # buang baris gagal agar tidak dianggap selesai (bug lama: SYS_TIMEOUT dianggap processed)
            os.replace(self.path, self.path + ".bak")
            good = [r for r in rows if r.get("status") not in RETRY_STATUSES]
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                for r in good:
                    w.writerow({k: r.get(k, "") for k in COLUMNS})
            rows = good
            print(f"  ♻️ {len(bad)} baris ERROR lama dibuang dari {self.path} (backup .bak)")
        return {(r["config"], r["item_id"]) for r in rows}

    def has(self, config, item_id):
        return (config, item_id) in self.done

    def add(self, **row):
        row.setdefault("run_id", "")
        for k in ("raw",):
            if row.get(k) is not None:
                row[k] = str(row[k])[:2000]
        if isinstance(row.get("extra"), (dict, list)):
            row["extra"] = json.dumps(row["extra"], ensure_ascii=False)
        new = not os.path.isfile(self.path)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
        self.done.add((row["config"], row["item_id"]))
