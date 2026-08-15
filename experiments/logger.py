import os
import sys
import json
import csv
import time
from typing import Dict, Any, List, Optional


class ExperimentLogger:
    """
    Structured experiment record logger for VisionGuide AI Phase 11.
    Persists experimental runs into category-specific JSON and CSV artifacts.
    """

    CATEGORIES = [
        "camera",
        "detection",
        "tracking",
        "phmu",
        "distance",
        "danger",
        "free_space",
        "decision",
        "audio",
        "end_to_end",
        "safety",
        "resource",
    ]

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.ensure_directories()

    def ensure_directories(self):
        """Create category directory taxonomy under experiments/."""
        os.makedirs(self.base_dir, exist_ok=True)
        for cat in self.CATEGORIES:
            os.makedirs(os.path.join(self.base_dir, cat), exist_ok=True)

    def log_experiment(
        self,
        category: str,
        experiment_id: str,
        scenario: str,
        configuration: Dict[str, Any],
        measured_outputs: Dict[str, Any],
        result: str,
        limitations: Optional[List[str]] = None,
    ) -> str:
        """Record an experimental run into JSON and CSV format."""
        if category not in self.CATEGORIES:
            category = "end_to_end"

        cat_dir = os.path.join(self.base_dir, category)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        record_id = f"{experiment_id}_{timestamp_str}"

        record = {
            "record_id": record_id,
            "experiment_id": experiment_id,
            "category": category,
            "timestamp": time.time(),
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "os_environment": "Windows 11 / Python 3.14.6",
            "scenario": scenario,
            "configuration": configuration,
            "measured_outputs": measured_outputs,
            "result": result,
            "limitations": limitations or [],
        }

        # Save JSON record
        json_path = os.path.join(cat_dir, f"{record_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        # Append to CSV summary
        csv_path = os.path.join(cat_dir, f"{category}_summary.csv")
        file_exists = os.path.exists(csv_path)

        flattened = {
            "record_id": record_id,
            "experiment_id": experiment_id,
            "date_time": record["date_time"],
            "scenario": scenario,
            "result": result,
        }
        for k, v in measured_outputs.items():
            if isinstance(v, (int, float, str, bool)):
                flattened[f"out_{k}"] = v

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flattened.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flattened)

        return json_path
