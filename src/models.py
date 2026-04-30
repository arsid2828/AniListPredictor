import pandas as pd
import numpy as np
import logging
import joblib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import shap

from .dataset import build_user_dataframe, build_user_manga_dataframe
from .features import engineer_features, engineer_manga_features

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_DATASET_SIZE = 30


def _report_progress(progress_callback: Optional[Callable[[float, str], None]], value: float, message: str):
    if progress_callback is not None:
        progress_callback(float(value), str(message))

class BaselineModel:
    """Predicts using the user's historical mean up to that point."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return X['hist_user_mean'].values
    def get_params(self, deep=True):
        return {}
    def set_params(self, **params):
        return self

class ClassifierToRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps a classification model to use it for regression by rounding the targets.
    Useful for 'True' Naive Bayes (GaussianNB) on score data.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.classes_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def fit(self, X, y):
        # Round y to nearest integer to create classes
        y_class = np.round(y).astype(int)
        self.classifier.fit(X, y_class)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X):
        # Could use predict_proba for a weighted mean, but simple class prediction is 'truer' to NB
        return self.classifier.predict(X).astype(float)

    def get_params(self, deep=True):
        return {"classifier": self.classifier}

    def set_params(self, **params):
        if "classifier" in params:
            self.classifier = params["classifier"]
        return self

def evaluate_model(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'model': name, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

def train_pruned_tree(X_train, y_train, X_val, y_val):
    """Find best alpha for cost-complexity pruning using Validation set."""
    dt = DecisionTreeRegressor(random_state=42)
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]
    
    best_alpha = 0
    best_score = float('inf')
    best_dt = None
    
    for alpha in ccp_alphas[::max(1, len(ccp_alphas)//50)]:
        model = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        if mae < best_score:
            best_score = mae
            best_alpha = alpha
            best_dt = model
            
    logger.info(f"Best CCP Alpha chosen: {best_alpha:.5f} with Val MAE: {best_score:.3f}")
    if best_dt is None:
        best_dt = DecisionTreeRegressor(random_state=42, max_depth=5).fit(X_train, y_train)
    return best_dt

def _get_candidate_models(dataset_size, X_train_inner=None, y_train_inner=None, X_val_inner=None, y_val_inner=None):
    """Restituisce i modelli candidati in base alla dimensione del dataset esplorata (Gating)."""
    models = {
        'Baseline (Hist Mean)': BaselineModel(),
        'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        'Bayesian Ridge (Naive Regression)': Pipeline([('scaler', StandardScaler()), ('bayesian', BayesianRidge())]),
        'Decision Tree (Raw)': DecisionTreeRegressor(random_state=42, min_samples_leaf=3)
    }
    
    # Pruned tree richiede validation split esplicito
    if X_train_inner is not None and len(X_train_inner) > 10:
        models['Decision Tree (Pruned)'] = train_pruned_tree(X_train_inner, y_train_inner, X_val_inner, y_val_inner)
    
    if dataset_size > 75:
        models['Random Forest'] = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42)
        models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        
    if dataset_size > 150:
        models['XGBoost'] = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
        models['LightGBM'] = LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        models['Neural Network (MLP)'] = Pipeline([
            ('scaler', StandardScaler()), 
            ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ])
        
    return models

def _optuna_tune(model_name, X_train, y_train, X_val, y_val, n_trials=30):
    """Ottimizza esclusivamente il modello selezionato sulla fold cronologica di Validation."""
    best_model_obj = [None]
    best_mae_val = [float('inf')]
    
    def objective(trial):
        if 'Random Forest' in model_name:
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('rf_n_est', 50, 300),
                max_depth=trial.suggest_int('rf_depth', 3, 15),
                min_samples_leaf=trial.suggest_int('rf_msl', 1, 10),
                random_state=42
            )
        elif 'XGB' in model_name:
            model = XGBRegressor(
                n_estimators=trial.suggest_int('xgb_n_est', 50, 300),
                max_depth=trial.suggest_int('xgb_depth', 3, 10),
                learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.2),
                reg_alpha=trial.suggest_float('xgb_alpha', 0.0, 5.0),
                subsample=trial.suggest_float('xgb_sub', 0.6, 1.0),
                random_state=42, verbosity=0
            )
        elif 'LightGBM' in model_name:
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('lgbm_n_est', 50, 300),
                max_depth=trial.suggest_int('lgbm_depth', 3, 10),
                learning_rate=trial.suggest_float('lgbm_lr', 0.01, 0.2),
                reg_alpha=trial.suggest_float('lgbm_alpha', 0.0, 5.0),
                subsample=trial.suggest_float('lgbm_sub', 0.6, 1.0),
                random_state=42, verbose=-1
            )
        elif 'Gradient Boosting' in model_name:
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int('gb_n_est', 50, 300),
                max_depth=trial.suggest_int('gb_depth', 3, 10),
                learning_rate=trial.suggest_float('gb_lr', 0.01, 0.2),
                min_samples_leaf=trial.suggest_int('gb_msl', 1, 10),
                random_state=42
            )
        else:
            raise optuna.exceptions.TrialPruned() # Modello non supportato per il tuning
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        
        if mae < best_mae_val[0]:
            best_mae_val[0] = mae
            best_model_obj[0] = model
        
        return mae
        
    study = optuna.create_study(direction='minimize')
    if any(m in model_name for m in ['Random Forest', 'XGB', 'LightGBM', 'Gradient Boosting']):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return best_model_obj[0], study.best_value, study.best_params
    else:
        return None, None, None

def _extract_feature_importance(model, columns):
    """Extract feature importance from any model type."""
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif isinstance(model, Pipeline):
        last_step = model.steps[-1][1]
        if hasattr(last_step, 'feature_importances_'):
            importances = last_step.feature_importances_
    
    if importances is not None:
        return pd.DataFrame({
            'Feature': columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20).to_dict('records')
    return None

def _get_shap_explanations(model, X_sample, feature_names):
    """Calculate SHAP values for the best model using a small sample."""
    try:
        bg_sample = X_sample.sample(min(20, len(X_sample)), random_state=42)
        if hasattr(model, 'feature_importances_') or isinstance(model, (XGBRegressor, LGBMRegressor, RandomForestRegressor)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, bg_sample)
            
        shap_values = explainer(bg_sample)
        mean_abs_shap = np.abs(shap_values.values).mean(0)
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': mean_abs_shap
        }).sort_values('SHAP_Importance', ascending=False)
        
        return shap_df.head(20).to_dict('records')
    except Exception as e:
        logger.warning(f"SHAP calculation failed: {e}")
        return None

def _run_tscv(model, X, y, n_splits=5):
    """Run TimeSeriesSplit cross-validation on the Train+Val set and return fold MAEs."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
        X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]
        try:
            m = clone(model)
            m.fit(X_t, y_t)
            p = m.predict(X_v)
            fold_scores.append(mean_absolute_error(y_v, p))
        except Exception as e:
            pass
    return fold_scores

def _evaluate_candidates(models, X, y, n_splits=3):
    """Valuta i modelli candidati rigorosamente su set di Validation (cronologico)."""
    results = []
    
    if len(X) < 40:
        # Pochi dati: singolo split hold-out cronologico all'interno del Train+Val
        inner_split = int(len(X) * 0.8)
        X_train_in, y_train_in = X.iloc[:inner_split], y.iloc[:inner_split]
        X_val_in, y_val_in = X.iloc[inner_split:], y.iloc[inner_split:]
        
        for name, model in models.items():
            try:
                if name != 'Baseline (Hist Mean)' and "Pruned" not in name:
                    model.fit(X_train_in, y_train_in)
                preds = model.predict(X_val_in)
                res = evaluate_model(y_val_in, preds, name)
                results.append(res)
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
    else:
        # Dimensioni Ok: TimeSeriesSplit per la pura Model Selection
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for name, model in models.items():
            if name == 'Decision Tree (Pruned)':
                continue
            maes, rmses, r2s = [], [], []
            for train_idx, val_idx in tscv.split(X):
                X_t, y_t = X.iloc[train_idx], y.iloc[train_idx]
                X_v, y_v = X.iloc[val_idx], y.iloc[val_idx]
                try:
                    m = clone(model)
                    m.fit(X_t, y_t)
                    p = m.predict(X_v)
                    maes.append(mean_absolute_error(y_v, p))
                    rmses.append(np.sqrt(mean_squared_error(y_v, p)))
                    r2s.append(r2_score(y_v, p))
                except Exception as e:
                    pass
            if maes:
                results.append({'model': name, 'MAE': np.mean(maes), 'RMSE': np.mean(rmses), 'R2': np.mean(r2s)})
            
        if 'Decision Tree (Pruned)' in models:
             inner_split = int(len(X) * 0.8)
             X_val_in, y_val_in = X.iloc[inner_split:], y.iloc[inner_split:]
             p = models['Decision Tree (Pruned)'].predict(X_val_in)
             res = evaluate_model(y_val_in, p, 'Decision Tree (Pruned)')
             results.append(res)
             
    if not results:
        results = [evaluate_model(y, BaselineModel().predict(X), 'Baseline (Hist Mean)')]
        
    return pd.DataFrame(results).sort_values('MAE')

def _core_train_pipeline(X, y, use_optuna=True, progress_callback=None):
    """Core training logic (Model Selection rigorosamente su Train/Val, Valutazione onesta su Test finale)."""
    X = X.fillna(0)
    n_samples = len(X)
    _report_progress(progress_callback, 0.35, "Preparing train/validation/test splits")
    
    # 1. SPLIT TEMPORALE CRONOLOGICO (Isolamento del Test Set)
    test_end = int(n_samples * 0.85)

    if test_end <= 0 or (n_samples - test_end) < 2:
        test_end = int(n_samples * 0.8)

    X_train_val = X.iloc[:test_end]
    y_train_val = y.iloc[:test_end]
    X_test = X.iloc[test_end:]
    y_test = y.iloc[test_end:]
    
    # Inner split cronologico per tuning e modelli che richiedono custom validation set
    inner_split = int(len(X_train_val) * 0.8)
    X_train_inner = X_train_val.iloc[:inner_split]
    y_train_inner = y_train_val.iloc[:inner_split]
    X_val_inner = X_train_val.iloc[inner_split:]
    y_val_inner = y_train_val.iloc[inner_split:]
    
    # 2. GATING DEI MODELLI (Scartiamo quelli troppo costosi per dataset minimi)
    models = _get_candidate_models(len(X_train_val), X_train_inner, y_train_inner, X_val_inner, y_val_inner)
    
    # 3. VALUTAZIONE CANDIDATI (Rigida Validation per il Model Selection)
    _report_progress(progress_callback, 0.5, "Evaluating candidate models")
    n_splits_cv = 3 if len(X_train_val) > 75 else 2
    eval_df = _evaluate_candidates(models, X_train_val, y_train_val, n_splits=n_splits_cv)
    
    best_model_name = eval_df.iloc[0]['model']
    best_model = models[best_model_name]
    metrics_list = eval_df.to_dict('records')
    
    # 4. TUNING OPTUNA (Solo sul miglior modello emerso, ottimizzato su Train_Val inner)
    optuna_result = None
    if use_optuna and len(X_train_val) > 40:
        _report_progress(progress_callback, 0.65, "Running Optuna tuning on the best candidate")
        optuna_model, optuna_val_mae, optuna_params = _optuna_tune(
            best_model_name, X_train_inner, y_train_inner, X_val_inner, y_val_inner, n_trials=30
        )
        
        if optuna_model is not None and optuna_val_mae < eval_df.iloc[0]['MAE']:
            best_model = optuna_model
            best_model_name = f"Optuna Tuned ({best_model_name})"
            # Inseriamo i ratio migliori al primo posto
            metrics_list.insert(0, {'model': best_model_name, 'MAE': round(optuna_val_mae, 4), 'RMSE': 0.0, 'R2': 0.0})
            optuna_result = {'best_params': optuna_params, 'val_mae': optuna_val_mae}
            
    # 5. RETRAIN FINALE PER DEPLOYMENT
    if best_model_name != 'Baseline (Hist Mean)' and "Pruned" not in best_model_name:
        _report_progress(progress_callback, 0.8, "Refitting the final model")
        best_model.fit(X_train_val, y_train_val) # Usa TUTTO il sapere tranne l'Holdout finale
        
    # 6. VALUTAZIONE FINALE ONESTA SUL TEST SET INCONTAMINATO
    _report_progress(progress_callback, 0.9, "Computing final evaluation and explanations")
    final_preds = best_model.predict(X_test)
    final_test_metrics = evaluate_model(y_test, final_preds, best_model_name)
    
    # Riscriviamo i campi 'metriche' del primo risultato cosicché la UI (app/main.py) mostri il punteggio onesto (Test) sul TOP layer
    metrics_list[0]['MAE'] = final_test_metrics['MAE']
    metrics_list[0]['RMSE'] = final_test_metrics['RMSE']
    metrics_list[0]['R2'] = final_test_metrics['R2']
    metrics_list[0]['model'] = f"{best_model_name} (Test Eval)"
    
    tscv_scores = _run_tscv(best_model, X_train_val, y_train_val, n_splits=n_splits_cv)
    feature_importance = _extract_feature_importance(best_model, X_train_val.columns)
    shap_explanations = _get_shap_explanations(best_model, X_val_inner, X_train_val.columns)
    
    fallback_recommended = final_test_metrics.get('R2', 0) < 0
    
    return {
        'model': best_model,
        'model_name': best_model_name,
        'train_columns': list(X_train_val.columns),
        'metrics': metrics_list,
        'feature_importance': feature_importance,
        'shap_explanations': shap_explanations,
        'optuna_result': optuna_result,
        'tscv_scores': tscv_scores,
        'fallback_recommended': fallback_recommended,
    }

def train_and_evaluate_all_models(username: str, use_optuna: bool = True, progress_callback=None) -> Dict[str, Any]:
    """Runs the entire ANIME pipeline for a username."""
    _report_progress(progress_callback, 0.05, "Fetching anime list and building the dataset")
    df_raw = build_user_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty:
        return {"status": "error", "message": f"No data found for {username}."}
    if len(df_raw) < MIN_DATASET_SIZE:
        return {"status": "fallback", "message": f"Only {len(df_raw)} rated anime found. Too few for ML - use Cold Start."}
    
    _report_progress(progress_callback, 0.2, "Engineering anime features")
    X, y = engineer_features(df_raw)
    result = _core_train_pipeline(X, y, use_optuna=use_optuna, progress_callback=progress_callback)
    
    _report_progress(progress_callback, 0.97, "Saving the trained anime model")
    model_artifact = {
        'model_name': result['model_name'],
        'model': result['model'],
        'train_columns': result['train_columns'],
        'metrics': result['metrics'],
        'feature_importance': result['feature_importance'],
        'shap_explanations': result['shap_explanations'],
        'optuna_result': result['optuna_result'],
        'tscv_scores': result['tscv_scores'],
        'fallback_recommended': result['fallback_recommended'],
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'dataset_size': len(df_raw),
    }
    
    artifact_path = MODELS_DIR / f"{username.lower()}_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    logger.info(f"Saved best anime model ({result['model_name']}) to {artifact_path}")
    _report_progress(progress_callback, 1.0, "Anime training completed")
    
    return model_artifact

def train_and_evaluate_all_manga_models(username: str, use_optuna: bool = True, progress_callback=None) -> Dict[str, Any]:
    """Runs the entire MANGA pipeline for a username."""
    _report_progress(progress_callback, 0.05, "Fetching manga list and building the dataset")
    df_raw = build_user_manga_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty:
        return {"status": "error", "message": f"No manga data found for {username}."}
    if len(df_raw) < MIN_DATASET_SIZE:
        return {"status": "fallback", "message": f"Only {len(df_raw)} rated manga found. Too few for ML - use Cold Start."}
    
    _report_progress(progress_callback, 0.2, "Engineering manga features")
    X, y = engineer_manga_features(df_raw)
    result = _core_train_pipeline(X, y, use_optuna=use_optuna, progress_callback=progress_callback)
    
    _report_progress(progress_callback, 0.97, "Saving the trained manga model")
    model_artifact = {
        'model_name': result['model_name'],
        'model': result['model'],
        'train_columns': result['train_columns'],
        'metrics': result['metrics'],
        'feature_importance': result['feature_importance'],
        'shap_explanations': result['shap_explanations'],
        'optuna_result': result['optuna_result'],
        'tscv_scores': result['tscv_scores'],
        'fallback_recommended': result['fallback_recommended'],
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'dataset_size': len(df_raw),
    }
    
    artifact_path = MODELS_DIR / f"{username.lower()}_manga_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    logger.info(f"Saved best manga model ({result['model_name']}) to {artifact_path}")
    _report_progress(progress_callback, 1.0, "Manga training completed")
    
    return model_artifact

if __name__ == "__main__":
    test_username = os.environ.get("TEST_USERNAME", "example_user")
    result = train_and_evaluate_all_models(test_username)
    if 'metrics' in result:
        print(pd.DataFrame(result['metrics']).to_string(index=False))
        print(f"\nBest: {result['model_name']}")
        if result.get('tscv_scores'):
            print(f"TSCV MAE: {np.mean(result['tscv_scores']):.4f} ± {np.std(result['tscv_scores']):.4f}")
