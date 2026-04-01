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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import build_user_dataframe, build_user_manga_dataframe
from .features import engineer_features, engineer_manga_features, temporal_train_val_test_split

logger = logging.getLogger(__name__)

MODELS_DIR = Path("c:/Users/arsid/Desktop/AnilistProject/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class BaselineModel:
    """Predicts using the user's historical mean up to that point. Valid for chronological data."""
    def fit(self, X, y):
        pass
    def predict(self, X):
        return X['hist_user_mean'].values

def evaluate_model(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def train_pruned_tree(X_train, y_train, X_val, y_val):
    """Find best alpha for cost-complexity pruning using Validation set."""
    dt = DecisionTreeRegressor(random_state=42)
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    # Prune extreme alphas to save time
    ccp_alphas = ccp_alphas[:-1] 
    
    best_alpha = 0
    best_score = float('inf')
    best_dt = None
    
    # We shouldn't test thousands of alphas if dataset is huge, but Anilist datasets are small
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

def train_and_evaluate_all_models(username: str) -> Dict[str, Any]:
    """Runs the entire pipeline for a username, evaluates models, saves best."""
    # 1. Fetch & Build Dataset
    df_raw = build_user_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty or len(df_raw) < 10:
        return {"status": "error", "message": f"Dataset for {username} is too small."}
        
    # 2. Engineer Features
    X, y = engineer_features(df_raw)
    
    # 3. Split Data chronologically
    # 70% Train, 15% Val (for pruning), 15% Test (final eval)
    # We need to fillna one last time for safety
    X = X.fillna(0)
    
    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    if len(X_val) == 0 or len(X_test) == 0:
        logger.warning("Dataset too small for 70/15/15 split. Will use 80/20 train/test without isolated val.")
        train_end = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:], y.iloc[train_end:]
        X_test, y_test = X_val, y_val

    # 4. Define Models
    models = {
        'Baseline (Hist Mean)': BaselineModel(),
        'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        'KNN Regressor': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),
        'Decision Tree (Raw)': DecisionTreeRegressor(random_state=42, min_samples_leaf=3),
        'Decision Tree (Pruned)': train_pruned_tree(X_train, y_train, X_val, y_val),
        'Random Forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'Voting Ensemble (Smooth)': VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, min_samples_leaf=1, random_state=42)),
            ('ridge', Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))]))
        ], weights=[0.6, 0.4])
    }

    # 5. Train & Evaluate
    results = []
    trained_models = {}
    
    # For baseline, we just evaluate
    for name, model in models.items():
        if name != 'Decision Tree (Pruned)' and name != 'Baseline (Hist Mean)':
             model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds, name)
        results.append(metrics)
        trained_models[name] = model
        
    results_df = pd.DataFrame(results).sort_values('MAE')
    logger.info(f"\nModel Comparison for {username}:\n" + results_df.to_string(index=False))
    
    # 6. Select Best Model
    # Forziamo l'uso del Voting Ensemble per garantire continuità nei decimali (niente gradini dell'albero!)
    best_model_name = 'Voting Ensemble (Smooth)'
    best_model = trained_models[best_model_name]
    
    # Extract Feature Importance (if tree based)
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15).to_dict('records')
    elif hasattr(best_model, 'estimators_'):
        for est in best_model.estimators_:
            if hasattr(est, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': est.feature_importances_
                }).sort_values('Importance', ascending=False).head(15).to_dict('records')
                break
    elif isinstance(best_model, Pipeline) and hasattr(best_model.steps[-1][1], 'feature_importances_'):
         feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.steps[-1][1].feature_importances_
        }).sort_values('Importance', ascending=False).head(15).to_dict('records')

    # 7. Save Artifacts for Inference
    # We save the best model, AND the training columns to align inference data
    model_artifact = {
        'username': username,
        'model_name': best_model_name,
        'model': best_model,
        'train_columns': list(X_train.columns),
        'metrics': results_df.to_dict('records'),
        'feature_importance': feature_importance
    }
    
    artifact_path = MODELS_DIR / f"{username}_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    logger.info(f"Saved best model ({best_model_name}) to {artifact_path}")
    
    return model_artifact

def train_and_evaluate_all_manga_models(username: str):
    """Runs the entire pipeline for a username's MANGA list, evaluates models, saves best."""
    df_raw = build_user_manga_dataframe(username, force_refresh=True, save_csv=True)
    if df_raw.empty or len(df_raw) < 10:
        return {"status": "error", "message": f"Manga dataset for {username} is too small ({len(df_raw) if not df_raw.empty else 0} entries)."}
        
    X, y = engineer_manga_features(df_raw)
    
    X = X.fillna(0)
    
    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    if len(X_val) == 0 or len(X_test) == 0:
        logger.warning("Manga dataset too small for 70/15/15 split. Using 80/20.")
        train_end = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:], y.iloc[train_end:]
        X_test, y_test = X_val, y_val

    models = {
        'Baseline (Hist Mean)': BaselineModel(),
        'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        'KNN Regressor': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))]),
        'Decision Tree (Raw)': DecisionTreeRegressor(random_state=42, min_samples_leaf=3),
        'Decision Tree (Pruned)': train_pruned_tree(X_train, y_train, X_val, y_val),
        'Random Forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'Voting Ensemble (Smooth)': VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=100, min_samples_leaf=1, random_state=42)),
            ('ridge', Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=5.0))]))
        ], weights=[0.6, 0.4])
    }

    results = []
    trained_models = {}
    
    for name, model in models.items():
        if name != 'Decision Tree (Pruned)' and name != 'Baseline (Hist Mean)':
             model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = evaluate_model(y_test, preds, name)
        results.append(metrics)
        trained_models[name] = model
        
    results_df = pd.DataFrame(results).sort_values('MAE')
    logger.info(f"\nManga Model Comparison for {username}:\n" + results_df.to_string(index=False))
    
    best_model_name = 'Voting Ensemble (Smooth)'
    best_model = trained_models[best_model_name]
    
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15).to_dict('records')
    elif hasattr(best_model, 'estimators_'):
        for est in best_model.estimators_:
            if hasattr(est, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': est.feature_importances_
                }).sort_values('Importance', ascending=False).head(15).to_dict('records')
                break

    model_artifact = {
        'username': username,
        'model_name': best_model_name,
        'model': best_model,
        'train_columns': list(X_train.columns),
        'metrics': results_df.to_dict('records'),
        'feature_importance': feature_importance
    }
    
    artifact_path = MODELS_DIR / f"{username}_manga_best_model.pkl"
    joblib.dump(model_artifact, artifact_path)
    logger.info(f"Saved best manga model ({best_model_name}) to {artifact_path}")
    
    return model_artifact

if __name__ == "__main__":
    # Test execution
    train_and_evaluate_all_models("arsid")
