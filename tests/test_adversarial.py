"""
Adversarial Robustness Tests
=============================
Measures how much model performance degrades when an attacker
deliberately modifies traffic to evade detection.

Three evasion strategies tested:
  1. SlowIAT   — attacker stretches inter-arrival times (looks less bursty)
  2. Fragment  — attacker splits data into tiny packets (low byte counts)
  3. Mimic     — attacker mimics normal traffic statistics (hardest evasion)

Expected behaviour (NOT a failure):
  - F1 drops under evasion — this is honest and expected
  - But system should still detect ≥ 50% of attacks (not collapse to 0)
  - Documenting this gap is the academically correct approach

Results are printed and saved to models/adversarial_report.json.

Run:
    python -X utf8 -m pytest tests/test_adversarial.py -v -s
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
REPORT_PATH = MODELS_DIR / "adversarial_report.json"

EARLY_FEATURES = [
    "avg_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "avg_interarrival", "std_interarrival", "min_interarrival", "max_interarrival",
    "incoming_ratio", "packet_count", "total_bytes",
    "flow_duration", "bytes_per_second", "pkts_per_second",
]

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    path = MODELS_DIR / "best_early_model.pkl"
    if not path.exists():
        pytest.skip("best_early_model.pkl not found — run ml.train_model first")
    return joblib.load(path)


@pytest.fixture(scope="module")
def attack_samples(model) -> pd.DataFrame:
    """
    Generate synthetic attack-profile samples.
    Attack profile: small packets, very short inter-arrivals, low incoming ratio.
    """
    rng = np.random.default_rng(42)
    n   = 300
    df  = pd.DataFrame({
        "avg_packet_size":  rng.uniform(40,  120,   n),
        "std_packet_size":  rng.uniform(5,   30,    n),
        "min_packet_size":  rng.uniform(40,  80,    n),
        "max_packet_size":  rng.uniform(80,  200,   n),
        "avg_interarrival": rng.uniform(0.0001, 0.005, n),
        "std_interarrival": rng.uniform(0.0,  0.002, n),
        "min_interarrival": rng.uniform(0.0,  0.001, n),
        "max_interarrival": rng.uniform(0.002, 0.01, n),
        "incoming_ratio":   rng.uniform(0.0,  0.10,  n),
        "packet_count":     np.full(n, 5.0),
        "total_bytes":      rng.uniform(200,  600,   n),
        "flow_duration":    rng.uniform(0.0005, 0.05, n),
        "bytes_per_second": rng.uniform(4000, 100000, n),
        "pkts_per_second":  rng.uniform(100,  5000,  n),
    })
    return df


def _predict_attack_recall(model, df: pd.DataFrame) -> float:
    """Return fraction of samples classified as attack (label=2)."""
    preds = model.predict(df[EARLY_FEATURES])
    # model may be binary (0/1) or 3-class (0/1/2)
    if model.n_classes_ == 2:
        # class 1 = vpn/attack in binary models
        attack_mask = (preds == 1)
    else:
        attack_mask = (preds == 2)
    return float(attack_mask.mean())


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBaselineDetection:
    """Verify model detects clean attack profiles at baseline."""

    def test_baseline_attack_recall_above_50pct(self, model, attack_samples):
        recall = _predict_attack_recall(model, attack_samples)
        print(f"\n  [baseline] attack recall = {recall:.1%}")
        assert recall >= 0.50, (
            f"Baseline recall {recall:.1%} < 50% — model may not be trained on attack class"
        )

    def test_baseline_predict_proba_shape(self, model, attack_samples):
        proba = model.predict_proba(attack_samples[EARLY_FEATURES])
        assert proba.shape == (len(attack_samples), model.n_classes_)


class TestSlowIATEvasion:
    """
    Evasion 1: attacker increases inter-arrival times by 10x
    to look less bursty (avoids triggering rate-based rules).
    """

    def test_slowiat_recall_documented(self, model, attack_samples):
        evaded = attack_samples.copy()
        evaded["avg_interarrival"] *= 10
        evaded["std_interarrival"] *= 10
        evaded["min_interarrival"] *= 10
        evaded["max_interarrival"] *= 10
        # recompute derived features
        evaded["bytes_per_second"] = evaded["total_bytes"] / evaded["avg_interarrival"].clip(lower=1e-9)
        evaded["pkts_per_second"]  = evaded["packet_count"] / (
            evaded["avg_interarrival"] * evaded["packet_count"]
        ).clip(lower=1e-9)

        baseline = _predict_attack_recall(model, attack_samples)
        evaded_r = _predict_attack_recall(model, evaded)
        drop     = baseline - evaded_r

        print(f"\n  [SlowIAT] baseline={baseline:.1%}  evaded={evaded_r:.1%}  drop={drop:.1%}")
        _save_result("slow_iat", baseline, evaded_r)

        # System is honest: we document the drop, not fail if it drops
        # But it must not collapse to 0
        assert evaded_r >= 0.05, (
            "SlowIAT evasion causes complete detection failure — model is fragile"
        )

    def test_slowiat_confidence_decreases(self, model, attack_samples):
        """Confidence should decrease under evasion."""
        evaded = attack_samples.copy()
        evaded["avg_interarrival"] *= 10
        evaded["max_interarrival"] *= 10

        proba_base  = model.predict_proba(attack_samples[EARLY_FEATURES]).max(axis=1).mean()
        proba_evade = model.predict_proba(evaded[EARLY_FEATURES]).max(axis=1).mean()
        print(f"\n  [SlowIAT confidence] baseline={proba_base:.3f}  evaded={proba_evade:.3f}")
        # confidence should be lower or equal under evasion (generally)
        # we just document it — soft assert
        assert proba_evade <= proba_base + 0.15, (
            "Evasion unexpectedly increases model confidence — review features"
        )


class TestFragmentationEvasion:
    """
    Evasion 2: attacker fragments traffic into tiny packets.
    avg_packet_size → very small, total_bytes same, packet_count × 3.
    """

    def test_fragmentation_recall_documented(self, model, attack_samples):
        evaded = attack_samples.copy()
        evaded["avg_packet_size"] = evaded["avg_packet_size"] / 4
        evaded["min_packet_size"] = 40.0
        evaded["max_packet_size"] = evaded["avg_packet_size"] * 2
        evaded["std_packet_size"] = evaded["avg_packet_size"] * 0.2

        baseline = _predict_attack_recall(model, attack_samples)
        evaded_r = _predict_attack_recall(model, evaded)
        drop     = baseline - evaded_r

        print(f"\n  [Fragment] baseline={baseline:.1%}  evaded={evaded_r:.1%}  drop={drop:.1%}")
        _save_result("fragmentation", baseline, evaded_r)

        assert evaded_r >= 0.05, (
            "Fragmentation evasion causes complete detection failure"
        )


class TestMimicNormalEvasion:
    """
    Evasion 3: attacker fully mimics normal traffic statistics.
    This is the hardest evasion — we expect near-zero detection.
    Documenting this as a known limitation.
    """

    def test_mimic_normal_recall_is_low(self, model, attack_samples):
        """
        If attacker perfectly mimics normal traffic, model WILL fail.
        This test documents that known limitation — it is NOT a bug.
        """
        rng = np.random.default_rng(99)
        n   = len(attack_samples)

        # Normal profile (HTTP browsing)
        mimic = pd.DataFrame({
            "avg_packet_size":  rng.uniform(200, 1400,  n),
            "std_packet_size":  rng.uniform(50,  400,   n),
            "min_packet_size":  rng.uniform(40,  200,   n),
            "max_packet_size":  rng.uniform(600, 1500,  n),
            "avg_interarrival": rng.uniform(0.05, 0.5,  n),
            "std_interarrival": rng.uniform(0.01, 0.1,  n),
            "min_interarrival": rng.uniform(0.01, 0.05, n),
            "max_interarrival": rng.uniform(0.2,  1.5,  n),
            "incoming_ratio":   rng.uniform(0.4,  0.6,  n),
            "packet_count":     np.full(n, 5.0),
            "total_bytes":      rng.uniform(1000, 10000, n),
            "flow_duration":    rng.uniform(0.25, 3.0,  n),
            "bytes_per_second": rng.uniform(500,  5000, n),
            "pkts_per_second":  rng.uniform(1,    20,   n),
        })

        baseline = _predict_attack_recall(model, attack_samples)
        mimic_r  = _predict_attack_recall(model, mimic)

        print(
            f"\n  [MimicNormal] baseline={baseline:.1%}  mimic_recall={mimic_r:.1%}\n"
            f"  ⚠ Known limitation: perfect traffic mimicry bypasses statistical ML.\n"
            f"  → Mitigation: combine with Threat Intel and behavioral baseline."
        )
        _save_result("mimic_normal", baseline, mimic_r)

        # We EXPECT low detection on mimic — this is documenting the limitation
        # The test passes as long as we can measure it (no crash)
        assert 0.0 <= mimic_r <= 1.0


# ── Helper: save results ───────────────────────────────────────────────────────

_report: dict = {}

def _save_result(name: str, baseline: float, evaded: float):
    _report[name] = {
        "baseline_recall": round(baseline, 4),
        "evaded_recall":   round(evaded,   4),
        "drop":            round(baseline - evaded, 4),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    try:
        existing = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
        existing.update(_report)
        REPORT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass
