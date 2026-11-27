from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from config import PipelineConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _create_base_models(random_state: int) -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def tune_model_with_optuna(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 15,
    random_state: int = 42,
):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "lightgbm":
            estimator = lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 120, 400),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 50),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif model_name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 120, 350),
                max_depth=trial.suggest_int("max_depth", 8, 24),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            estimator = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 120, 400),
                max_depth=trial.suggest_int("max_depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )

        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    if model_name == "lightgbm":
        best_estimator = lgb.LGBMClassifier(
            **best_params, random_state=random_state, n_jobs=-1, verbose=-1
        )
    elif model_name == "random_forest":
        best_estimator = RandomForestClassifier(
            **best_params, random_state=random_state, n_jobs=-1
        )
    else:
        best_estimator = XGBClassifier(
            **best_params,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )

    best_estimator.fit(X_train, y_train)
    return best_estimator, study.best_value, best_params


def train_and_select_best(
    cfg: PipelineConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor,
    feature_names: list[str],
):
    candidates = _create_base_models(cfg.random_state)
    results = {}

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = {
            "estimator": clf,
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro")),
            "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
            "tuned": False,
        }

    # Tune primary model with Optuna and compare
    tuned_estimator, cv_f1, best_params = tune_model_with_optuna(
        model_name=cfg.primary_model,
        X_train=X_train,
        y_train=y_train,
        n_trials=cfg.optuna_trials,
        random_state=cfg.random_state,
    )
    tuned_name = f"{cfg.primary_model}_optuna"
    tuned_preds = tuned_estimator.predict(X_test)
    results[tuned_name] = {
        "estimator": tuned_estimator,
        "accuracy": float(accuracy_score(y_test, tuned_preds)),
        "macro_f1": float(f1_score(y_test, tuned_preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, tuned_preds, average="weighted")),
        "tuned": True,
        "cv_macro_f1": float(cv_f1),
        "best_params": best_params,
    }

    best_model_name = max(results.keys(), key=lambda k: results[k]["macro_f1"])
    best_estimator = results[best_model_name]["estimator"]

    artifact_path = Path(cfg.model_artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_to_save = {
        "model_name": best_model_name,
        "estimator": best_estimator,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metrics": results[best_model_name],
    }
    joblib.dump(bundle_to_save, artifact_path)

    return results, best_model_name, best_estimator


def load_saved_bundle(model_artifact_path: Path):
    path = Path(model_artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model artifact not found at '{path}'. Run main.py first."
        )
    return joblib.load(path)
