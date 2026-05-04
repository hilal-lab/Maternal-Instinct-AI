"""
Data Loader Module — Loads and queries the synthetic dataset for evaluation and context enrichment.
Provides the system with ground-truth examples for intent classification, emotion detection,
and Layer 3/4 behavior validation as described in the paper.
"""
import csv
import os
from typing import List, Dict, Optional

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic_dataset.csv")

class SyntheticDataset:
    """Loads and provides access to the synthetic evaluation dataset."""

    def __init__(self, path: str = DATASET_PATH):
        self.entries: List[Dict] = []
        self._load(path)

    def _load(self, path: str):
        """Load CSV into a list of dictionaries."""
        if not os.path.exists(path):
            print(f"⚠️ Dataset not found at: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["severity"] = int(row.get("severity", 1))
                row["expected_layer3_flag"] = row.get("expected_layer3_flag", "false").lower() == "true"
                self.entries.append(row)

    def get_by_category(self, category: str) -> List[Dict]:
        """Filter entries by category (STRESS, BURNOUT, SELF_DEPRECATION, HEALTH_VIOLATION, NEUTRAL)."""
        return [e for e in self.entries if e.get("category") == category]

    def get_by_emotion(self, emotion: str) -> List[Dict]:
        """Filter entries by expected emotion."""
        return [e for e in self.entries if e.get("expected_emotion") == emotion]

    def get_by_severity(self, min_sev: int = 1, max_sev: int = 5) -> List[Dict]:
        """Filter entries by severity range."""
        return [e for e in self.entries if min_sev <= e.get("severity", 1) <= max_sev]

    def get_few_shot_examples(self, category: str, n: int = 3) -> str:
        """
        Get n representative examples from a category, formatted for use as
        few-shot context in LLM prompts. Used by the Empathy Scout for
        pattern matching.
        """
        examples = self.get_by_category(category)[:n]
        if not examples:
            return ""

        lines = []
        for ex in examples:
            lines.append(
                f"- Input: \"{ex['input_text']}\" -> "
                f"Emotion: {ex['expected_emotion']}, "
                f"Tone: {ex['expected_tone']}, "
                f"Severity: {ex['severity']}"
            )
        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Return summary statistics of the dataset."""
        from collections import Counter
        return {
            "total": len(self.entries),
            "by_category": dict(Counter(e["category"] for e in self.entries)),
            "by_emotion": dict(Counter(e["expected_emotion"] for e in self.entries)),
            "by_tone": dict(Counter(e["expected_tone"] for e in self.entries)),
            "layer3_violations": sum(1 for e in self.entries if e["expected_layer3_flag"]),
            "layer4_rewrites": sum(1 for e in self.entries if e["expected_layer4_action"] == "REWRITE"),
        }

    def evaluate_prediction(self, row_id: str, predicted_emotion: str, predicted_layer3: bool, predicted_layer4: str) -> Dict:
        """
        Compare predicted outputs against ground truth for a specific entry.
        Returns match results per field.
        """
        entry = next((e for e in self.entries if e["id"] == row_id), None)
        if not entry:
            return {"error": f"Entry {row_id} not found"}

        return {
            "id": row_id,
            "emotion_match": predicted_emotion == entry["expected_emotion"],
            "layer3_match": predicted_layer3 == entry["expected_layer3_flag"],
            "layer4_match": predicted_layer4 == entry["expected_layer4_action"],
            "expected": {
                "emotion": entry["expected_emotion"],
                "layer3": entry["expected_layer3_flag"],
                "layer4": entry["expected_layer4_action"],
            },
            "predicted": {
                "emotion": predicted_emotion,
                "layer3": predicted_layer3,
                "layer4": predicted_layer4,
            },
        }
