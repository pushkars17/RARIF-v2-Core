import argparse
import uuid

from config import PipelineConfig
from data_generator import prepare_datasets
from db_store import RarifDB
from evaluate import evaluate_model
from explain import generate_shap_explanations
from model import train_and_select_best


def parse_args():
    parser = argparse.ArgumentParser(description="RARIF Framework v2 Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Primary model family to tune with Optuna",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute features from CSV instead of reading parquet cache",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plot generation for faster runs",
    )
    return parser.parse_args()


def run_pipeline():
    args = parse_args()
    cfg = PipelineConfig(
        primary_model=args.model,
        optuna_trials=args.trials,
    )

    run_id = str(uuid.uuid4())[:8]
    print(f"Starting RARIF v2 pipeline | run_id={run_id} | primary_model={cfg.primary_model}")

    # 1. Load & engineer features
    bundle = prepare_datasets(cfg, use_cache=not args.no_cache)
    db = RarifDB(cfg.duckdb_path)
    db.sync_hotspots(bundle["df"])

    # 2. Train candidate models & tune best with Optuna
    results, best_name, best_estimator = train_and_select_best(
        cfg=cfg,
        X_train=bundle["X_train"],
        y_train=bundle["y_train"],
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],
        preprocessor=bundle["preprocessor"],
        feature_names=bundle["feature_names"],
    )

    # Log all candidate runs into DuckDB
    for m_name, metrics in results.items():
        db.log_model_run(
            run_id=f"{run_id}_{m_name}",
            model_name=m_name,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
            is_best=(m_name == best_name),
        )

    print("\nCandidate Model Comparison (Test Set):")
    for m_name, metrics in results.items():
        star = " <-- BEST" if m_name == best_name else ""
        print(
            f"  - {m_name:18s} | Acc: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}{star}"
        )

    # 3. Evaluate best model & save confusion matrix + city metrics
    evaluate_model(
        cfg=cfg,
        estimator=best_estimator,
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],
        meta_test=bundle["meta_test"],
        model_name=best_name,
    )

    # 4. Explain feature impact using SHAP
    if not args.skip_shap:
        generate_shap_explanations(
            cfg=cfg,
            estimator=best_estimator,
            X_sample=bundle["X_test"],
            feature_names=bundle["feature_names"],
            max_rows=1000,
        )

    print(f"\nPipeline complete! Model saved at '{cfg.model_artifact_path}' and DB at '{cfg.duckdb_path}'.")


if __name__ == "__main__":
    run_pipeline()
