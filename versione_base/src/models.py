import logging
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .dataset import build_user_dataframe, build_user_manga_dataframe
from .features import engineer_features, engineer_manga_features

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_DATASET_SIZE = 30


class BaselineModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["hist_user_mean"].values

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def evaluate_model(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train_pruned_tree(X_train, y_train, X_val, y_val):
    dt = DecisionTreeRegressor(random_state=42)
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]

    best_score = float("inf")
    best_dt = None

    for alpha in ccp_alphas[:: max(1, len(ccp_alphas) // 50)]:
        model = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        if mae < best_score:
            best_score = mae
            best_dt = model

    if best_dt is None:
        best_dt = DecisionTreeRegressor(random_state=42, max_depth=5).fit(X_train, y_train)
    return best_dt


def _get_candidate_models(dataset_size, X_train_inner=None, y_train_inner=None, X_val_inner=None, y_val_inner=None):
    models = {
        "Baseline (Hist Mean)": BaselineModel(),
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "Bayesian Ridge": Pipeline([("scaler", StandardScaler()), ("bayesian", BayesianRidge())]),
        "Decision Tree (Raw)": DecisionTreeRegressor(random_state=42, min_samples_leaf=3),
    }
    if X_train_inner is not None and len(X_train_inner) > 10:
        models["Decision Tree (Pruned)"] = train_pruned_tree(X_train_inner, y_train_inner, X_val_inner, y_val_inner)
    if dataset_size > 75:
        models["Random Forest"] = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42)
    return models


def _extract_feature_importance(model, columns):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, Pipeline):
        last_step = model.steps[-1][1]
        if hasattr(last_step, "feature_importances_"):
            importances = last_step.feature_importances_
    if importances is not None:
        return (
            pd.DataFrame({"Feature": columns, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(20)
            .to_dict("records")
        )
    return None


def _get_shap_explanations(model, X_sample, feature_names):
    try:
        bg_sample = X_sample.sample(min(20, len(X_sample)), random_state=42)
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, bg_sample)
        shap_values = explainer(bg_sample)
        mean_abs_shap = np.abs(shap_values.values).mean(0)
        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP_Importance": mean_abs_shap}).sort_values(
            "SHAP_Importance", ascending=False
        )
        return shap_df.head(20).to_dict("records")
    except Exception as exc:
        logger.warning("SHAP calculation failed: %s", exc)
        return None


def _run_tscv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
        X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]
        try:
            candidate = clone(model)
            candidate.fit(X_t, y_t)
            preds = candidate.predict(X_v)
            fold_scores.append(mean_absolute_error(y_v, preds))
        except Exception:
            pass
    return fold_scores


def _evaluate_candidates(models, X, y, n_splits=3):
    results = []
    if len(X) < 40:
        inner_split = int(len(X) * 0.8)
        X_train_in, y_train_in = X.iloc[:inner_split], y.iloc[:inner_split]
        X_val_in, y_val_in = X.iloc[inner_split:], y.iloc[inner_split:]
        for name, model in models.items():
            try:
                if name != "Baseline (Hist Mean)" and "Pruned" not in name:
                    model.fit(X_train_in, y_train_in)
                preds = model.predict(X_val_in)
                results.append(evaluate_model(y_val_in, preds, name))
            except Exception as exc:
                logger.warning("Model %s failed: %s", name, exc)
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for name, model in models.items():
            if name == "Decision Tree (Pruned)":
                continue
            maes, rmses, r2s = [], [], []
            for train_idx, val_idx in tscv.split(X):
                X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
                X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]
                try:
                    candidate = clone(model)
                    candidate.fit(X_t, y_t)
                    preds = candidate.predict(X_v)
                    maes.append(mean_absolute_error(y_v, preds))
                    rmses.append(np.sqrt(mean_squared_error(y_v, preds)))
                    r2s.append(r2_score(y_v, preds))
                except Exception:
                    pass
            if maes:
                results.append({"model": name, "MAE": np.mean(maes), "RMSE": np.mean(rmses), "R2": np.mean(r2s)})
        if "Decision Tree (Pruned)" in models:
            inner_split = int(len(X) * 0.8)
            X_val_in, y_val_in = X.iloc[inner_split:], y.iloc[inner_split:]
            preds = models["Decision Tree (Pruned)"].predict(X_val_in)
            results.append(evaluate_model(y_val_in, preds, "Decision Tree (Pruned)"))
    if not results:
        results = [evaluate_model(y, BaselineModel().predict(X), "Baseline (Hist Mean)")]
    return pd.DataFrame(results).sort_values("MAE")


def _core_train_pipeline(X, y):
    X = X.fillna(0)
    n_samples = len(X)
    test_end = int(n_samples * 0.85)
    if test_end <= 0 or (n_samples - test_end) < 2:
        test_end = int(n_samples * 0.8)

    X_train_val = X.iloc[:test_end]
    y_train_val = y.iloc[:test_end]
    X_test = X.iloc[test_end:]
    y_test = y.iloc[test_end:]

    inner_split = int(len(X_train_val) * 0.8)
    X_train_inner = X_train_val.iloc[:inner_split]
    y_train_inner = y_train_val.iloc[:inner_split]
    X_val_inner = X_train_val.iloc[inner_split:]
    y_val_inner = y_train_val.iloc[inner_split:]

    models = _get_candidate_models(len(X_train_val), X_train_inner, y_train_inner, X_val_inner, y_val_inner)
    n_splits_cv = 3 if len(X_train_val) > 75 else 2
    eval_df = _evaluate_candidates(models, X_train_val, y_train_val, n_splits=n_splits_cv)

    best_model_name = eval_df.iloc[0]["model"]
    best_model = models[best_model_name]
    metrics_list = eval_df.to_dict("records")

    if best_model_name != "Baseline (Hist Mean)" and "Pruned" not in best_model_name:
        best_model.fit(X_train_val, y_train_val)

    final_preds = best_model.predict(X_test)
    final_test_metrics = evaluate_model(y_test, final_preds, best_model_name)
    metrics_list[0]["MAE"] = final_test_metrics["MAE"]
    metrics_list[0]["RMSE"] = final_test_metrics["RMSE"]
    metrics_list[0]["R2"] = final_test_metrics["R2"]
    metrics_list[0]["model"] = f"{best_model_name} (Test Eval)"

    return {
        "model": best_model,
        "model_name": best_model_name,
        "train_columns": list(X_train_val.columns),
        "metrics": metrics_list,
        "feature_importance": _extract_feature_importance(best_model, X_train_val.columns),
        "shap_explanations": _get_shap_explanations(best_model, X_val_inner, X_train_val.columns),
        "tscv_scores": _run_tscv(best_model, X_train_val, y_train_val, n_splits=n_splits_cv),
        "fallback_recommended": final_test_metrics.get("R2", 0) < 0,
    }


def train_and_evaluate_all_models(username: str) -> Dict[str, Any]:
    df_raw = build_user_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty:
        return {"status": "error", "message": f"No data found for {username}."}
    if len(df_raw) < MIN_DATASET_SIZE:
        return {"status": "fallback", "message": f"Only {len(df_raw)} rated anime found. Too few for ML - use Cold Start."}

    X, y = engineer_features(df_raw)
    result = _core_train_pipeline(X, y)
    model_artifact = {
        "model_name": result["model_name"],
        "model": result["model"],
        "train_columns": result["train_columns"],
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "shap_explanations": result["shap_explanations"],
        "tscv_scores": result["tscv_scores"],
        "fallback_recommended": result["fallback_recommended"],
        "trained_at": pd.Timestamp.now().isoformat(),
        "dataset_size": len(df_raw),
    }
    artifact_path = MODELS_DIR / f"{username.lower()}_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    return model_artifact


def train_and_evaluate_all_manga_models(username: str) -> Dict[str, Any]:
    df_raw = build_user_manga_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty:
        return {"status": "error", "message": f"No manga data found for {username}."}
    if len(df_raw) < MIN_DATASET_SIZE:
        return {"status": "fallback", "message": f"Only {len(df_raw)} rated manga found. Too few for ML - use Cold Start."}

    X, y = engineer_manga_features(df_raw)
    result = _core_train_pipeline(X, y)
    model_artifact = {
        "model_name": result["model_name"],
        "model": result["model"],
        "train_columns": result["train_columns"],
        "metrics": result["metrics"],
        "feature_importance": result["feature_importance"],
        "shap_explanations": result["shap_explanations"],
        "tscv_scores": result["tscv_scores"],
        "fallback_recommended": result["fallback_recommended"],
        "trained_at": pd.Timestamp.now().isoformat(),
        "dataset_size": len(df_raw),
    }
    artifact_path = MODELS_DIR / f"{username.lower()}_manga_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    return model_artifact


if __name__ == "__main__":
    test_username = os.environ.get("TEST_USERNAME", "example_user")
    result = train_and_evaluate_all_models(test_username)
    if "metrics" in result:
        print(pd.DataFrame(result["metrics"]).to_string(index=False))
