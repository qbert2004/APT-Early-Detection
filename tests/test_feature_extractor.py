"""Unit tests for FeatureExtractor."""
from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from features.feature_extractor import (
    EARLY_FEATURES,
    FULL_EXTRA_FEATURES,
    LABEL_MAP,
    FeatureExtractor,
)

# ── fixtures ───────────────────────────────────────────────────────────────────

SYNTHETIC_CSV = textwrap.dedent("""\
    flow_id,avg_packet_size,std_packet_size,min_packet_size,max_packet_size,\
avg_interarrival,std_interarrival,min_interarrival,max_interarrival,\
incoming_ratio,packet_count,total_bytes,flow_duration,bytes_per_second,pkts_per_second,label
    0,200.0,50.0,100.0,300.0,0.01,0.002,0.005,0.02,0.3,5,1000.0,0.05,20000.0,100.0,normal
    1,400.0,100.0,200.0,600.0,0.02,0.005,0.01,0.04,0.6,5,2000.0,0.10,20000.0,50.0,vpn
    2,150.0,30.0,80.0,250.0,0.005,0.001,0.002,0.01,0.9,5,750.0,0.025,30000.0,200.0,attack
""")


@pytest.fixture()
def synthetic_csv_path(tmp_path):
    p = tmp_path / "synthetic.csv"
    p.write_text(SYNTHETIC_CSV)
    return p


# ── FeatureExtractor.from_synthetic_csv ───────────────────────────────────────

class TestFromSyntheticCsv:
    def test_returns_correct_shape(self, synthetic_csv_path):
        fe = FeatureExtractor(n_packets=5)
        X, y = fe.from_synthetic_csv(synthetic_csv_path, mode="early")
        assert X.shape == (3, len(EARLY_FEATURES))
        assert len(y) == 3

    def test_labels_are_numeric(self, synthetic_csv_path):
        fe = FeatureExtractor(n_packets=5)
        _, y = fe.from_synthetic_csv(synthetic_csv_path, mode="early")
        assert set(y.unique()).issubset({0, 1, 2})

    def test_correct_label_mapping(self, synthetic_csv_path):
        fe = FeatureExtractor(n_packets=5)
        _, y = fe.from_synthetic_csv(synthetic_csv_path, mode="early")
        assert y.iloc[0] == LABEL_MAP["normal"]
        assert y.iloc[1] == LABEL_MAP["vpn"]
        assert y.iloc[2] == LABEL_MAP["attack"]

    def test_no_nan_in_features(self, synthetic_csv_path):
        fe = FeatureExtractor(n_packets=5)
        X, _ = fe.from_synthetic_csv(synthetic_csv_path, mode="early")
        assert not X.isnull().any().any()

    def test_no_inf_in_features(self, synthetic_csv_path):
        fe = FeatureExtractor(n_packets=5)
        X, _ = fe.from_synthetic_csv(synthetic_csv_path, mode="early")
        assert not np.isinf(X.values).any()

    def test_early_features_only(self, synthetic_csv_path):
        fe = FeatureExtractor(n_packets=5)
        X, _ = fe.from_synthetic_csv(synthetic_csv_path, mode="early")
        assert list(X.columns) == EARLY_FEATURES

    def test_unknown_label_raises(self, tmp_path):
        bad_csv = SYNTHETIC_CSV.replace("normal", "unknown_class", 1)
        p = tmp_path / "bad.csv"
        p.write_text(bad_csv)
        fe = FeatureExtractor(n_packets=5)
        with pytest.raises(ValueError, match="Unknown labels"):
            fe.from_synthetic_csv(p)

    def test_handles_inf_values(self, tmp_path):
        """Rows with inf should be filled with median, not crash."""
        csv = SYNTHETIC_CSV.replace("20000.0,100.0,normal", "inf,100.0,normal", 1)
        p = tmp_path / "inf.csv"
        p.write_text(csv)
        fe = FeatureExtractor(n_packets=5)
        X, _ = fe.from_synthetic_csv(p)
        assert not np.isinf(X.values).any()
        assert not X.isnull().any().any()


# ── LABEL_MAP ─────────────────────────────────────────────────────────────────

class TestLabelMap:
    def test_all_three_classes(self):
        assert set(LABEL_MAP.keys()) == {"normal", "vpn", "attack"}

    def test_values_are_ints(self):
        assert all(isinstance(v, int) for v in LABEL_MAP.values())

    def test_unique_values(self):
        vals = list(LABEL_MAP.values())
        assert len(vals) == len(set(vals))
