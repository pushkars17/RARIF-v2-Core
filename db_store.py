from datetime import datetime
from pathlib import Path
import duckdb
import pandas as pd


class RarifDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _connect(self):
        return duckdb.connect(str(self.db_path))

    def _init_tables(self):
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id VARCHAR PRIMARY KEY,
                    created_at TIMESTAMP,
                    model_name VARCHAR,
                    accuracy DOUBLE,
                    macro_f1 DOUBLE,
                    weighted_f1 DOUBLE,
                    is_best BOOLEAN
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS city_hotspots (
                    city VARCHAR,
                    state VARCHAR,
                    total_accidents INTEGER,
                    avg_risk_score DOUBLE,
                    high_risk_share DOUBLE
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS hourly_hotspots (
                    hour INTEGER,
                    total_accidents INTEGER,
                    avg_risk_score DOUBLE,
                    high_risk_share DOUBLE
                )
                """
            )

    def log_model_run(
        self,
        run_id: str,
        model_name: str,
        accuracy: float,
        macro_f1: float,
        weighted_f1: float,
        is_best: bool = False,
    ):
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO model_runs VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    datetime.now(),
                    model_name,
                    float(accuracy),
                    float(macro_f1),
                    float(weighted_f1),
                    bool(is_best),
                ],
            )

    def sync_hotspots(self, df: pd.DataFrame):
        city_summary = (
            df.groupby(["city", "state"], as_index=False)
            .agg(
                total_accidents=("risk_score", "count"),
                avg_risk_score=("risk_score", "mean"),
                high_risk_share=("risk_tier", lambda s: float((s == 2).mean())),
            )
            .sort_values("avg_risk_score", ascending=False)
        )

        hourly_summary = (
            df.groupby("hour", as_index=False)
            .agg(
                total_accidents=("risk_score", "count"),
                avg_risk_score=("risk_score", "mean"),
                high_risk_share=("risk_tier", lambda s: float((s == 2).mean())),
            )
            .sort_values("hour")
        )

        with self._connect() as con:
            con.execute("DELETE FROM city_hotspots")
            con.register("city_df", city_summary)
            con.execute("INSERT INTO city_hotspots SELECT * FROM city_df")

            con.execute("DELETE FROM hourly_hotspots")
            con.register("hourly_df", hourly_summary)
            con.execute("INSERT INTO hourly_hotspots SELECT * FROM hourly_df")

    def get_model_runs(self) -> pd.DataFrame:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM model_runs ORDER BY created_at DESC"
            ).df()

    def get_city_hotspots(self) -> pd.DataFrame:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM city_hotspots ORDER BY avg_risk_score DESC"
            ).df()

    def get_hourly_hotspots(self) -> pd.DataFrame:
        with self._connect() as con:
            return con.execute(
                "SELECT * FROM hourly_hotspots ORDER BY hour ASC"
            ).df()