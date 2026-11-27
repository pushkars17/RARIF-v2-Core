from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PipelineConfig

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "lanes",
    "traffic_signal",
    "temperature",
    "is_weekend",
    "is_peak_hour",
    "month",
    "hour_sin",
    "hour_cos",
    "high_risk_hour",
]

CATEGORICAL_FEATURES = [
    "city",
    "state",
    "day_of_week",
    "season",
    "road_type",
    "weather",
    "visibility",
    "traffic_density",
    "cause",
    "festival",
]

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def risk_score_to_tier(score: float) -> int:
    if score < 0.35:
        return 0
    if score <= 0.65:
        return 1
    return 2


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Fill missing festival values for regular non-festival days
    data["festival"] = data["festival"].fillna("None").astype(str).str.strip()
    data.loc[data["festival"] == "", "festival"] = "None"

    # Extract month and season from date
    dt = pd.to_datetime(data["date"], errors="coerce")
    data["month"] = dt.dt.month.fillna(1).astype(int)
    data["season"] = data["month"].apply(month_to_season)

    # Cyclical hour features + late-night/early-morning indicator
    hour = pd.to_numeric(data["hour"], errors="coerce").fillna(0).astype(int)
    data["hour"] = hour
    data["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    data["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    data["high_risk_hour"] = hour.isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    # Target tier from composite risk_score (0.10 to 1.00)
    data["risk_tier"] = data["risk_score"].apply(risk_score_to_tier).astype(int)

    return data


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def load_or_build_processed_df(cfg: PipelineConfig, use_cache: bool = True) -> pd.DataFrame:
    cache_path = Path(cfg.parquet_cache_path)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw_df = pd.read_csv(cfg.data_csv_path)
    processed_df = engineer_features(raw_df)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(cache_path, index=False)
    return processed_df


def prepare_datasets(cfg: PipelineConfig, use_cache: bool = True):
    df = load_or_build_processed_df(cfg, use_cache=use_cache)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["risk_tier"].to_numpy()
    meta_df = df[
        ["city", "state", "road_type", "weather", "hour", "latitude", "longitude", "risk_score"]
    ].copy()

    X_train_raw, X_test_raw, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        meta_df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "df": df,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train.reset_index(drop=True),
        "meta_test": meta_test.reset_index(drop=True),
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    config = PipelineConfig()
    bundle = prepare_datasets(config, use_cache=False)
    print("Processed rows:", len(bundle["df"]))
    print("Train shape:", bundle["X_train"].shape, "| Test shape:", bundle["X_test"].shape)
    print("Transformed feature count:", len(bundle["feature_names"]))
