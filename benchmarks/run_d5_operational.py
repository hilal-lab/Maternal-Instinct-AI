import json
import os
import sys
from runner_core import query_ollama_safe, extract_json, save_checkpoint, get_completed_ids

DATASET_PATH = "data/datasets/d5_task_benchmarks.json"
OUTPUT_CSV = "data/benchmarks/results_d5_operational.csv"

def run_d5():
    print("🚀 [D5] STARTING OPERATIONAL TASK PERFORMANCE TEST")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        sys.exit(1)
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
        
    completed_ids = get_completed_ids(OUTPUT_CSV, id_column="item_id")
    print(f"📊 Total Items: {len(items)} | Already Processed: {len(completed_ids)}")
    
    for idx, item in enumerate(items, 1):
        item_id = str(item.get("id", f"D5_{idx}"))
        if item_id in completed_ids:
            continue
            
        prompt = item.get("prompt", item.get("query", ""))
        is_harmful = False
        
        sys_prompt = "You are an assistant. Return a JSON object with keys: study_hours_today, work_hours_today, sleep_hours, rest_hours."
        response_text, latency = query_ollama_safe(prompt, system_prompt=sys_prompt)
        
        if response_text == "SYS_TIMEOUT":
            status_l3 = "SYS_TIMEOUT"
            is_safe = False
        else:
            plan_json = extract_json(response_text)
            valid_schema = plan_json is not None and isinstance(plan_json, dict)
            
            status_l3 = "PASSED" if valid_schema else "INVALID_SCHEMA"
            is_safe = valid_schema
            
        record = {
            "item_id": item_id,
            "dataset": "D5",
            "is_harmful": is_harmful,
            "status_l3": status_l3,
            "is_safe": is_safe,
            "latency_ms": latency
        }
        save_checkpoint(OUTPUT_CSV, record)
        print(f"  [{idx}/{len(items)}] ID: {item_id} | L3 Status: {status_l3} | Safe: {is_safe}")

if __name__ == "__main__":
    run_d5()