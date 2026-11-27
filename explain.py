from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import PipelineConfig


def _extract_2d_shap_matrix(shap_values) -> np.ndarray:
    """Handles SHAP outputs across binary/multi-class/Explanation versions."""
    if hasattr(shap_values, "values"):
        vals = np.asarray(shap_values.values)
    elif isinstance(shap_values, list):
        # List of arrays per class -> average absolute impact across classes
        arrs = [np.abs(np.asarray(a)) for a in shap_values]
        return np.mean(arrs, axis=0)
    else:
        vals = np.asarray(shap_values)

    if vals.ndim == 3:
        # Shape (samples, features, classes) -> mean abs across classes
        return np.mean(np.abs(vals), axis=2)
    if vals.ndim == 2:
        return np.abs(vals)
    raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")


def generate_shap_explanations(
    cfg: PipelineConfig,
    estimator,
    X_sample: np.ndarray,
    feature_names: list[str],
    max_rows: int = 1000,
) -> pd.DataFrame:
    outputs_dir = Path(cfg.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    n_rows = min(len(X_sample), max_rows)
    X_sub = X_sample[:n_rows]

    # Use TreeExplainer for tree ensembles; fallback to KernelExplainer if needed
    try:
        explainer = shap.TreeExplainer(estimator)
        raw_shap = explainer.shap_values(X_sub)
    except Exception:
        bg = shap.sample(X_sample, min(100, len(X_sample)), random_state=cfg.random_state)
        explainer = shap.KernelExplainer(estimator.predict_proba, bg)
        raw_shap = explainer.shap_values(X_sub, nsamples=100)

    mean_abs_imp = _extract_2d_shap_matrix(raw_shap).mean(axis=0)

    # Clean transformer prefix like 'num__' or 'cat__'
    cleaned_names = [
        name.split("__", 1)[1] if "__" in name else name for name in feature_names
    ]

    imp_df = (
        pd.DataFrame(
            {
                "feature": cleaned_names,
                "mean_abs_shap": mean_abs_imp,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top_csv_path = outputs_dir / "top_shap_features.csv"
    imp_df.to_csv(top_csv_path, index=False)

    # Plot top 15 features bar chart
    top_n = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(top_n["feature"], top_n["mean_abs_shap"], color="#2563eb")
    ax.set_title("Top 15 Risk Drivers (Mean |SHAP| Impact)")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    shap_img_path = outputs_dir / "shap_summary.png"
    plt.savefig(shap_img_path, dpi=200)
    plt.close(fig)

    print("\nTop 10 Most Influential Risk Features:")
    print(imp_df.head(10).to_string(index=False))

    return imp_df
