import json
import os
import random
from typing import Dict, List, Tuple, Any

class LACFDatasetLoader:
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = data_dir

    def _load_json(self, filename: str) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dataset tidak ditemukan: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dataset_d1(self, dev_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Memuat Dataset D1 (Parametric Plan Scenarios) dengan 20/80 split"""
        data = self._load_json("D1_plan_scenarios.json")
        random.seed(seed)
        random.shuffle(data)
        
        split_idx = int(len(data) * dev_ratio)
        dev_split = data[:split_idx]
        test_split = data[split_idx:]
        return dev_split, test_split

    def load_dataset_d2(self, dev_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Memuat Dataset D2 (User Affect & Distress Messages) dengan 20/80 split"""
        data = self._load_json("D2_user_affect_messages.json")
        random.seed(seed)
        random.shuffle(data)
        
        split_idx = int(len(data) * dev_ratio)
        dev_split = data[:split_idx]
        test_split = data[split_idx:]
        return dev_split, test_split

    def load_dataset_d3(self) -> List[Dict]:
        """Memuat Dataset D3 (XSTest + HarmBench)"""
        return self._load_json("D3_harmfulness_prompts.json")

    def load_dataset_d4(self) -> List[Dict]:
        """Memuat Dataset D4 (Indirect Prompt Injection G1-G6)"""
        return self._load_json("D4_prompt_injections.json")

    def load_dataset_d5(self, task_type: str = None) -> List[Dict]:
        """Memuat Dataset D5 (Task Performance Benchmark)"""
        data = self._load_json("D5_task_benchmarks.json")
        if task_type:
            return [item for item in data if item.get("task_type") == task_type]
        return data


# Contoh Penggunaan:
if __name__ == "__main__":
    loader = LACFDatasetLoader()
    
    # Load D1
    d1_dev, d1_test = loader.load_dataset_d1()
    print(f"[D1] Dev set: {len(d1_dev)} items | Test set: {len(d1_test)} items")
    
    # Load D2
    d2_dev, d2_test = loader.load_dataset_d2()
    print(f"[D2] Dev set: {len(d2_dev)} items | Test set: {len(d2_test)} items")
    
    # Load D4
    d4_all = loader.load_dataset_d4()
    print(f"[D4] Total Injection cases: {len(d4_all)} items")