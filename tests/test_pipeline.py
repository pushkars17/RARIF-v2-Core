import pandas as pd
import pytest

from config import PredictionInput
from data_generator import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features


def test_no_post_accident_leakage_in_features():
    leakage_cols = {"accident_severity", "casualties", "vehicles_involved", "risk_score"}
    all_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert leakage_cols.isdisjoint(all_features)


def test_engineer_features_fills_festival_and_creates_tier():
    sample = pd.DataFrame(
        [
            {
                "city": "Pune",
                "state": "Maharashtra",
                "latitude": 18.52,
                "longitude": 73.85,
                "date": "2024-10-22",
                "hour": 23,
                "day_of_week": "Tuesday",
                "is_weekend": 0,
                "is_peak_hour": 0,
                "lanes": 4,
                "traffic_signal": 1,
                "temperature": 27,
                "road_type": "highway",
                "weather": "fog",
                "visibility": "low",
                "traffic_density": "high",
                "cause": "weather",
                "festival": None,
                "risk_score": 0.85,
            }
        ]
    )
    out = engineer_features(sample)
    assert out.loc[0, "festival"] == "None"
    assert out.loc[0, "season"] == "Post-Monsoon"
    assert out.loc[0, "high_risk_hour"] == 1
    assert out.loc[0, "risk_tier"] == 2


def test_prediction_input_rejects_invalid_hour():
    with pytest.raises(Exception):
        PredictionInput(
            city="Delhi",
            state="Delhi",
            latitude=28.61,
            longitude=77.20,
            hour=28,  # invalid (> 23)
            day_of_week="Monday",
            is_weekend=0,
            is_peak_hour=1,
            lanes=4,
            traffic_signal=1,
            temperature=30,
            road_type="urban",
            weather="clear",
            visibility="high",
            traffic_density="medium",
            cause="overspeeding",
        )