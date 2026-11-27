from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class PipelineConfig(BaseModel):
    data_csv_path: Path = Path("data/indian_roads_dataset.csv")
    parquet_cache_path: Path = Path("artifacts/processed_features.parquet")
    model_artifact_path: Path = Path("artifacts/best_model.joblib")
    duckdb_path: Path = Path("artifacts/rarif.duckdb")
    outputs_dir: Path = Path("outputs")

    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42, ge=0)
    optuna_trials: int = Field(default=15, ge=5, le=100)
    primary_model: Literal["xgboost", "lightgbm", "random_forest"] = "xgboost"

    @field_validator("data_csv_path")
    @classmethod
    def check_dataset_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{v}'. Place indian_roads_dataset.csv inside 'data/'."
            )
        return v


class PredictionInput(BaseModel):
    city: Literal[
        "Pune",
        "Mumbai",
        "Chandigarh",
        "Chennai",
        "Delhi",
        "Bangalore",
        "Hyderabad",
        "Kolkata",
    ]
    state: str = Field(min_length=2)
    latitude: float = Field(ge=6.0, le=38.0)
    longitude: float = Field(ge=68.0, le=98.0)
    month: int = Field(default=7, ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    day_of_week: Literal[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    is_weekend: int = Field(ge=0, le=1)
    is_peak_hour: int = Field(ge=0, le=1)
    lanes: int = Field(ge=1, le=12)
    traffic_signal: int = Field(ge=0, le=1)
    temperature: int = Field(ge=-5, le=55)
    road_type: Literal["highway", "urban", "rural"]
    weather: Literal["clear", "rain", "fog"]
    visibility: Literal["low", "medium", "high"]
    traffic_density: Literal["low", "medium", "high"]
    cause: Literal[
        "weather",
        "distraction",
        "overspeeding",
        "drunk driving",
        "poor road",
    ]
    festival: str = Field(default="None")