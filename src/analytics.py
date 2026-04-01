import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor

logger = logging.getLogger(__name__)


# ========== SHAP EXPLAINABILITY ==========

def get_shap_explanation(model, X_inference, feature_names=None):
    """Generate SHAP values for a single prediction. Returns (shap_values_array, expected_value) or (None, None)."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed")
        return None, None
    
    actual_model = model
    
    # Unwrap VotingRegressor: use first tree-based estimator
    if isinstance(model, VotingRegressor) and hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'feature_importances_'):
                actual_model = est
                break
    
    # Unwrap Pipeline: use last step
    if isinstance(actual_model, Pipeline):
        actual_model = actual_model.steps[-1][1]
    
    try:
        explainer = shap.TreeExplainer(actual_model)
        sv = explainer.shap_values(X_inference)
        return sv, explainer.expected_value
    except Exception as e:
        logger.warning(f"SHAP TreeExplainer failed: {e}")
    
    return None, None


def shap_to_dataframe(shap_values, feature_names, top_n=15):
    """Convert SHAP values to a sorted DataFrame for display."""
    if shap_values is None:
        return None
    
    vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    df = pd.DataFrame({'Feature': feature_names, 'SHAP': vals})
    df['abs_SHAP'] = df['SHAP'].abs()
    df = df.sort_values('abs_SHAP', ascending=False).head(top_n)
    return df[['Feature', 'SHAP']]


# ========== PROFILE ANALYTICS ==========

def compute_genre_stats(df):
    """Average user score per genre."""
    genres = set()
    for g_str in df['genres'].dropna():
        for g in str(g_str).split(','):
            g = g.strip()
            if g:
                genres.add(g)
    
    stats = []
    for g in genres:
        mask = df['genres'].str.contains(g, na=False)
        if mask.sum() >= 1:
            stats.append({
                'genre': g,
                'mean_score': round(df.loc[mask, 'user_score'].mean(), 2),
                'count': int(mask.sum())
            })
    return pd.DataFrame(stats).sort_values('mean_score', ascending=False)


def compute_studio_stats(df, col='studios', min_count=2):
    """Average user score per studio/author."""
    buckets = {}
    for _, row in df.iterrows():
        val = row.get(col)
        if pd.isna(val) or not str(val).strip():
            continue
        for s in str(val).split(','):
            s = s.strip()
            if not s:
                continue
            if s not in buckets:
                buckets[s] = []
            buckets[s].append(row['user_score'])
    
    stats = [{'name': s, 'mean_score': round(np.mean(scores), 2), 'count': len(scores)}
             for s, scores in buckets.items() if len(scores) >= min_count]
    return pd.DataFrame(stats).sort_values('mean_score', ascending=False)


def compute_score_timeline(df):
    """Rolling mean of user scores over time."""
    timeline = df[['sort_date', 'user_score', 'title']].copy()
    timeline['sort_date'] = pd.to_datetime(timeline['sort_date'], errors='coerce')
    timeline = timeline.dropna(subset=['sort_date']).sort_values('sort_date')
    timeline['rolling_mean'] = timeline['user_score'].rolling(window=10, min_periods=1).mean()
    timeline['index'] = range(len(timeline))
    return timeline


def compute_user_bias(df):
    """Compute contrarian score and other bias metrics."""
    df_copy = df.copy()
    df_copy['averageScore_10'] = df_copy['averageScore']
    # If averageScore looks like 0-100 scale, convert
    if df_copy['averageScore_10'].median() > 15:
        df_copy['averageScore_10'] = df_copy['averageScore_10'] / 10.0
    
    df_copy['diff'] = df_copy['user_score'] - df_copy['averageScore_10']
    
    mean_diff = df_copy['diff'].mean()
    abs_mean_diff = df_copy['diff'].abs().mean()
    
    # Contrarian index: high = user diverges a lot from community
    contrarian = abs_mean_diff
    
    # Direction: positive = user rates higher than community
    direction = "generoso" if mean_diff > 0.2 else ("severo" if mean_diff < -0.2 else "allineato")
    
    return {
        'mean_diff': round(mean_diff, 2),
        'abs_mean_diff': round(abs_mean_diff, 2),
        'contrarian_index': round(contrarian, 2),
        'direction': direction,
        'total_rated': len(df),
        'mean_score': round(df['user_score'].mean(), 2),
        'std_score': round(df['user_score'].std(), 2),
    }


# ========== USER COMPATIBILITY ==========

def compute_compatibility(df1, df2, username1="User 1", username2="User 2"):
    """Compute compatibility between two users based on shared titles."""
    common = pd.merge(
        df1[['mediaId', 'title', 'user_score']],
        df2[['mediaId', 'title', 'user_score']],
        on='mediaId',
        suffixes=('_1', '_2')
    )
    
    if len(common) < 3:
        return {'status': 'insufficient', 'common_count': len(common)}
    
    # Use title from user 1
    common['title'] = common['title_1'].fillna(common['title_2'])
    
    correlation = common['user_score_1'].corr(common['user_score_2'])
    mae = mean_absolute_error(common['user_score_1'], common['user_score_2']) if len(common) > 0 else 0
    
    # Compatibility percentage (inverted MAE on 0-10 scale)
    compat_pct = max(0, min(100, (1.0 - mae / 5.0) * 100))
    
    return {
        'status': 'ok',
        'common_count': len(common),
        'correlation': round(correlation, 3) if not np.isnan(correlation) else 0,
        'mae': round(mae, 2),
        'compatibility_pct': round(compat_pct, 1),
        'mean_1': round(df1['user_score'].mean(), 2),
        'mean_2': round(df2['user_score'].mean(), 2),
        'common_df': common,
        'username1': username1,
        'username2': username2,
    }


def mean_absolute_error(y1, y2):
    return np.mean(np.abs(np.array(y1) - np.array(y2)))
