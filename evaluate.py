from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import PipelineConfig
from data_generator import RISK_LABELS

TARGET_NAMES = [RISK_LABELS[0], RISK_LABELS[1], RISK_LABELS[2]]


def evaluate_model(
    cfg: PipelineConfig,
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    model_name: str = "best_model",
) -> dict:
    outputs_dir = Path(cfg.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    y_pred = estimator.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    report_text = classification_report(
        y_test, y_pred, target_names=TARGET_NAMES, digits=4
    )
    print(f"\n=== Evaluation Summary ({model_name}) ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(report_text)

    # Save classification report text
    report_path = outputs_dir / "classification_report.txt"
    report_path.write_text(
        f"Model: {model_name}\nAccuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n{report_text}",
        encoding="utf-8",
    )

    # Plot and save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        ax=ax,
    )
    ax.set_title(f"RARIF Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Risk Tier")
    ax.set_ylabel("Actual Risk Tier")
    plt.tight_layout()

    cm_path = outputs_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # City-wise accuracy & macro F1 breakdown
    city_rows = []
    eval_df = meta_test.copy()
    eval_df["y_true"] = y_test
    eval_df["y_pred"] = y_pred

    for city, grp in eval_df.groupby("city"):
        city_rows.append(
            {
                "city": city,
                "samples": int(len(grp)),
                "accuracy": float(accuracy_score(grp["y_true"], grp["y_pred"])),
                "macro_f1": float(
                    f1_score(
                        grp["y_true"],
                        grp["y_pred"],
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    city_metrics_df = pd.DataFrame(city_rows).sort_values("accuracy", ascending=False)
    city_csv_path = outputs_dir / "city_metrics.csv"
    city_metrics_df.to_csv(city_csv_path, index=False)

    print("City-wise Test Performance:")
    print(city_metrics_df.to_string(index=False))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix_path": str(cm_path),
        "city_metrics": city_metrics_df,
    }
