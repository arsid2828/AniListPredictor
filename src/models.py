import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Any

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

def _optuna_tune(X_train, y_train, X_val, y_val, n_trials=30):
    """Bayesian hyperparameter search across tree-based and neural models."""
    best_model_obj = [None]
    best_mae_val = [float('inf')]
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgbm', 'gb', 'mlp'])
        
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('rf_n_est', 50, 300),
                max_depth=trial.suggest_int('rf_depth', 3, 15),
                min_samples_leaf=trial.suggest_int('rf_msl', 1, 10),
                random_state=42
            )
        elif model_type == 'xgb':
            model = XGBRegressor(
                n_estimators=trial.suggest_int('xgb_n_est', 50, 300),
                max_depth=trial.suggest_int('xgb_depth', 3, 10),
                learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.2),
                reg_alpha=trial.suggest_float('xgb_alpha', 0.0, 5.0),
                subsample=trial.suggest_float('xgb_sub', 0.6, 1.0),
                random_state=42, verbosity=0
            )
        elif model_type == 'lgbm':
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('lgbm_n_est', 50, 300),
                max_depth=trial.suggest_int('lgbm_depth', 3, 10),
                learning_rate=trial.suggest_float('lgbm_lr', 0.01, 0.2),
                reg_alpha=trial.suggest_float('lgbm_alpha', 0.0, 5.0),
                subsample=trial.suggest_float('lgbm_sub', 0.6, 1.0),
                random_state=42, verbose=-1
            )
        elif model_type == 'mlp':
            h1 = trial.suggest_int('mlp_h1', 16, 128)
            h2 = trial.suggest_int('mlp_h2', 8, 64)
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=(h1, h2),
                    activation=trial.suggest_categorical('mlp_act', ['relu', 'tanh']),
                    alpha=trial.suggest_float('mlp_reg', 1e-5, 1e-2, log=True),
                    learning_rate_init=trial.suggest_float('mlp_lr', 1e-4, 1e-2, log=True),
                    max_iter=500,
                    random_state=42
                ))
            ])
        else:
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int('gb_n_est', 50, 300),
                max_depth=trial.suggest_int('gb_depth', 3, 10),
                learning_rate=trial.suggest_float('gb_lr', 0.01, 0.2),
                min_samples_leaf=trial.suggest_int('gb_msl', 1, 10),
                random_state=42
            )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        
        if mae < best_mae_val[0]:
            best_mae_val[0] = mae
            best_model_obj[0] = model
        
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return best_model_obj[0], study.best_value, study.best_params

def _extract_feature_importance(model, columns):
    """Extract feature importance from any model type."""
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif isinstance(model, VotingRegressor) and hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'feature_importances_'):
                importances = est.feature_importances_
                break
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
        # Use a small background sample for faster computation
        bg_sample = X_sample.sample(min(20, len(X_sample)), random_state=42)
        
        # Determine specific explainer type
        if hasattr(model, 'feature_importances_') or isinstance(model, (XGBRegressor, LGBMRegressor, RandomForestRegressor)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, bg_sample)
            
        shap_values = explainer(bg_sample)
        
        # Get mean absolute SHAP values for global Importance
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
    """Run TimeSeriesSplit cross-validation and return fold MAEs."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
        X_fold_test, y_fold_test = X.iloc[test_idx], y.iloc[test_idx]
        
        try:
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            fold_preds = fold_model.predict(X_fold_test)
            fold_mae = mean_absolute_error(y_fold_test, fold_preds)
            fold_scores.append(fold_mae)
        except Exception as e:
            logger.warning(f"TSCV fold failed: {e}")
    
    return fold_scores

def _build_models(X_train, y_train, X_val, y_val):
    """Build the model dictionary with all candidates, including new Neural and Bayesian models."""
    return {
        'Baseline (Hist Mean)': BaselineModel(),
        'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        'Bayesian Ridge (Naive Regression)': Pipeline([('scaler', StandardScaler()), ('bayesian', BayesianRidge())]),
        'Gaussian NB (Naive Classification)': Pipeline([
            ('scaler', StandardScaler()), 
            ('nb', ClassifierToRegressorWrapper(GaussianNB()))
        ]),
        'KNN Regressor': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),
        'Decision Tree (Raw)': DecisionTreeRegressor(random_state=42, min_samples_leaf=3),
        'Decision Tree (Pruned)': train_pruned_tree(X_train, y_train, X_val, y_val),
        'Random Forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, reg_alpha=1.0, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, reg_alpha=1.0, random_state=42, verbose=-1),
        'Neural Network (MLP)': Pipeline([
            ('scaler', StandardScaler()), 
            ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
        ]),
        'Voting Ensemble (RF+GB)': VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42))
        ], weights=[0.55, 0.45]),
        'Stacking Ensemble (Advanced)': StackingRegressor(
            estimators=[
                ('xgb', XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)),
                ('rf', RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=42)),
                ('nb', ClassifierToRegressorWrapper(GaussianNB()))
            ],
            final_estimator=Ridge(alpha=1.0)
        )
    }

def _core_train_pipeline(X, y, use_optuna=True):
    """Core training logic shared by anime and manga."""
    X = X.fillna(0)
    
    # Chronological split: 70% train, 15% val, 15% test
    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    if len(X_val) == 0 or len(X_test) == 0:
        train_end = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:], y.iloc[train_end:]
        X_test, y_test = X_val, y_val
    
    # Build and train all models
    models = _build_models(X_train, y_train, X_val, y_val)
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        if name not in ['Decision Tree (Pruned)', 'Baseline (Hist Mean)']:
            model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds, name)
        results.append(metrics)
        trained_models[name] = model
    
    results_df = pd.DataFrame(results).sort_values('MAE')
    
    # === DYNAMIC MODEL SELECTION ===
    best_model_name = results_df.iloc[0]['model']
    best_model = trained_models[best_model_name]
    
    # === OPTUNA FINE TUNING ===
    optuna_result = None
    if use_optuna and len(X_train) >= 30:
        try:
            optuna_model, optuna_val_mae, optuna_params = _optuna_tune(
                X_train, y_train, X_val, y_val, n_trials=30
            )
            if optuna_model is not None:
                optuna_test_preds = optuna_model.predict(X_test)
                optuna_test_metrics = evaluate_model(y_test, optuna_test_preds, "Optuna Tuned")
                
                if optuna_test_metrics['MAE'] < results_df.iloc[0]['MAE']:
                    best_model = optuna_model
                    best_model_name = f"Optuna Tuned ({type(optuna_model).__name__})"
                    optuna_test_metrics['model'] = best_model_name
                    results.append(optuna_test_metrics)
                    trained_models[best_model_name] = best_model
                    results_df = pd.DataFrame(results).sort_values('MAE')
                
                optuna_result = {'best_params': optuna_params, 'val_mae': optuna_val_mae}
        except Exception as e:
            logger.warning(f"Optuna tuning failed: {e}")
    
    # === TIMESERIES CV ===
    tscv_scores = None
    try:
        tscv_scores = _run_tscv(best_model, X, y, n_splits=min(5, len(X) // 20))
    except Exception as e:
        logger.warning(f"TimeSeriesSplit CV failed: {e}")
    
    # Feature importance and SHAP
    feature_importance = _extract_feature_importance(best_model, X_train.columns)
    shap_explanations = _get_shap_explanations(best_model, X_val, X_train.columns)
    
    # Fallback check
    best_r2 = results_df.iloc[0]['R2']
    fallback_recommended = best_r2 < 0
    
    return {
        'model': best_model,
        'model_name': best_model_name,
        'train_columns': list(X_train.columns),
        'metrics': results_df.to_dict('records'),
        'feature_importance': feature_importance,
        'shap_explanations': shap_explanations,
        'optuna_result': optuna_result,
        'tscv_scores': tscv_scores,
        'fallback_recommended': fallback_recommended,
        'X_train_sample': X_train.sample(min(50, len(X_train)), random_state=42),
    }

def train_and_evaluate_all_models(username: str, use_optuna: bool = True) -> Dict[str, Any]:
    """Runs the entire ANIME pipeline for a username."""
    df_raw = build_user_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty:
        return {"status": "error", "message": f"Nessun dato trovato per {username}."}
    if len(df_raw) < MIN_DATASET_SIZE:
        return {"status": "fallback", "message": f"Solo {len(df_raw)} anime valutati. Troppo pochi per il ML — usa il Cold Start."}
    
    X, y = engineer_features(df_raw)
    result = _core_train_pipeline(X, y, use_optuna=use_optuna)
    
    model_artifact = {
        'username': username,
        'model_name': result['model_name'],
        'model': result['model'],
        'train_columns': result['train_columns'],
        'metrics': result['metrics'],
        'feature_importance': result['feature_importance'],
        'shap_explanations': result['shap_explanations'],
        'optuna_result': result['optuna_result'],
        'tscv_scores': result['tscv_scores'],
        'fallback_recommended': result['fallback_recommended'],
        'trained_at': pd.Timestamp.now().isoformat(),
        'dataset_size': len(df_raw),
        'X_train_sample': result['X_train_sample'],
    }
    
    artifact_path = MODELS_DIR / f"{username}_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    logger.info(f"Saved best anime model ({result['model_name']}) to {artifact_path}")
    
    return model_artifact

def train_and_evaluate_all_manga_models(username: str, use_optuna: bool = True) -> Dict[str, Any]:
    """Runs the entire MANGA pipeline for a username."""
    df_raw = build_user_manga_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty:
        return {"status": "error", "message": f"Nessun dato manga trovato per {username}."}
    if len(df_raw) < MIN_DATASET_SIZE:
        return {"status": "fallback", "message": f"Solo {len(df_raw)} manga valutati. Troppo pochi per il ML — usa il Cold Start."}
    
    X, y = engineer_manga_features(df_raw)
    result = _core_train_pipeline(X, y, use_optuna=use_optuna)
    
    model_artifact = {
        'username': username,
        'model_name': result['model_name'],
        'model': result['model'],
        'train_columns': result['train_columns'],
        'metrics': result['metrics'],
        'feature_importance': result['feature_importance'],
        'shap_explanations': result['shap_explanations'],
        'optuna_result': result['optuna_result'],
        'tscv_scores': result['tscv_scores'],
        'fallback_recommended': result['fallback_recommended'],
        'trained_at': pd.Timestamp.now().isoformat(),
        'dataset_size': len(df_raw),
        'X_train_sample': result['X_train_sample'],
    }
    
    artifact_path = MODELS_DIR / f"{username}_manga_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    logger.info(f"Saved best manga model ({result['model_name']}) to {artifact_path}")
    
    return model_artifact

if __name__ == "__main__":
    result = train_and_evaluate_all_models("arsid")
    if 'metrics' in result:
        print(pd.DataFrame(result['metrics']).to_string(index=False))
        print(f"\nBest: {result['model_name']}")
        if result.get('tscv_scores'):
            print(f"TSCV MAE: {np.mean(result['tscv_scores']):.4f} ± {np.std(result['tscv_scores']):.4f}")
