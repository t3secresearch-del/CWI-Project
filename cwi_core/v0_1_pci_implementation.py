"""
CWI-Project - EXP-047B
File: cwi_core/v0_1_pci_implementation.py
Author: Nasri - T3Sec Research, Lubumbashi
Philosophy: We don't reinvent concrete, we build a house that never existed.

FOUNDATIONS (REUSED):
- Causal Inference: DoWhy
- Causal Theory: Judea Pearl
- Deep Learning: PyTorch
- Active Learning

HOUSE (INVENTED):
- PCI: Predictive Causal Intelligence
- Persistent feedback learning
- Adaptive calibration

LINEAGE:
- EXP-047: Lost due to Colab /content reset
- EXP-047B: GitHub-first reconstruction

IMPORTANT: PCI does not discover causal truth automatically.
Causal conclusions depend on data and assumptions.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

class Foundations:
    def __init__(self):
        self.causal_lib = "DoWhy"
        self.theory = "Judea Pearl"
        self.dl_lib = "PyTorch"
    
    def check_dependencies(self):
        return {
            "dowhy": DOWHY_AVAILABLE,
            "status": "OK" if DOWHY_AVAILABLE else "Need: pip install dowhy"
        }
    
    def file_sha256(self, file_path: str):
        p = Path(file_path)
        if not p.exists():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()
    
    def get_lineage(self):
        return {
            "EXP-047": "LOST - /content",
            "EXP-047B": "GitHub First",
            "now": datetime.now(timezone.utc).isoformat()
        }

class PCI:
    def __init__(self, data=None, common_causes=None, storage_dir=".cwi"):
        self.version = "v0.1-EXP-047B"
        self.data = data
        self.common_causes = common_causes or []
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.storage_dir / "feedback.jsonl"
        self.history_file = self.storage_dir / "predictions.jsonl"
        self.calibration = self._load_calibration()
        self.seen_predictions = self._load_seen_ids()
    
    def _load_calibration(self):
        if not self.feedback_file.exists():
            return {"bias": 0.0, "count": 0, "avg_error": 0.0}
        errors = []
        for line in self.feedback_file.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    errors.append(rec.get("error", 0))
                except:
                    pass
        if not errors:
            return {"bias": 0.0, "count": 0, "avg_error": 0.0}
        bias = sum(errors) / len(errors)
        return {"bias": bias, "count": len(errors), "avg_error": sum(abs(e) for e in errors)/len(errors)}
    
    def _load_seen_ids(self):
        seen = set()
        if self.feedback_file.exists():
            for line in self.feedback_file.read_text().splitlines():
                if line.strip():
                    try:
                        seen.add(json.loads(line).get("prediction_id"))
                    except:
                        pass
        return seen
    
    def predict_causal_effect(self, cause, effect, run_refutation=False):
        if not DOWHY_AVAILABLE:
            raise RuntimeError("DoWhy not installed. Run: pip install dowhy pandas")
        if self.data is None:
            raise ValueError("PCI needs data: PCI(data=df, common_causes=[...])")
        
        model = CausalModel(
            data=self.data,
            treatment=cause,
            outcome=effect,
            common_causes=self.common_causes
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
        raw_value = float(estimate.value)
        calibrated_value = raw_value - self.calibration["bias"]
        pred_id = str(uuid.uuid4())[:8]
        
        result = {
            "prediction_id": pred_id,
            "cause": cause,
            "effect": effect,
            "raw_estimate": round(raw_value, 4),
            "calibrated_estimate": round(calibrated_value, 4),
            "calibration_bias_used": round(self.calibration["bias"], 4),
            "foundation": "DoWhy backdoor.linear_regression",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if run_refutation:
            try:
                refute = model.refute_estimate(identified, estimate, method_name="random_common_cause")
                result["refutation"] = str(refute)[:300]
            except Exception as e:
                result["refutation_error"] = str(e)[:200]
        
        with open(self.history_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        return result
    
    def self_improve(self, feedback: dict):
        pred_id = feedback.get("prediction_id")
        observed = feedback.get("observed_effect")
        if not pred_id or observed is None:
            return {"status": "REJECTED", "reason": "Need prediction_id and observed_effect"}
        if pred_id in self.seen_predictions:
            return {"status": "REJECTED", "reason": f"Duplicate feedback for {pred_id}"}
        
        pred = None
        if self.history_file.exists():
            for line in self.history_file.read_text().splitlines():
                if pred_id in line:
                    try:
                        rec = json.loads(line)
                        if rec["prediction_id"] == pred_id:
                            pred = rec
                            break
                    except:
                        pass
        if not pred:
            return {"status": "REJECTED", "reason": f"Prediction {pred_id} not found"}
        
        error = pred["calibrated_estimate"] - float(observed)
        record = {
            "prediction_id": pred_id,
            "predicted": pred["calibrated_estimate"],
            "observed": float(observed),
            "error": round(error, 4),
            "notes": feedback.get("notes", ""),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.seen_predictions.add(pred_id)
        self.calibration = self._load_calibration()
        return {
            "status": "LEARNED",
            "error": record["error"],
            "new_bias": round(self.calibration["bias"], 4),
            "avg_error": round(self.calibration["avg_error"], 4),
            "total_feedbacks": self.calibration["count"]
        }
    
    def get_sha256_lineage(self):
        p = Path(__file__)
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "local"
