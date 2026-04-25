import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from collections import defaultdict
from sklearn.pipeline import Pipeline

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

def compute_franchise_stats(df, username, is_manga=False):
    """Compute franchise statistics using the original relation graph."""
    from .dataset import DATA_DIR
    cache_dir = DATA_DIR.parent / "cache"
    cache_file = cache_dir / (f"user_manga_list_{username.lower()}.json" if is_manga else f"user_list_{username.lower()}.json")
    
    if not cache_file.exists():
        return pd.DataFrame()
        
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return pd.DataFrame()

    # Build adjacency list
    adj = defaultdict(set)
    media_info = {}
    
    valid_types = ['ADAPTATION', 'PREQUEL', 'SEQUEL', 'PARENT', 'SIDE_STORY', 'SUMMARY', 'ALTERNATIVE', 'SPIN_OFF', 'SOURCE', 'COMPILATION', 'CONTAINS']
    
    # Extract from raw data
    for str_list in raw_data:
        entries = str_list.get('entries', [])
        for entry in entries:
            media = entry.get('media', {})
            if not media: continue
            
            m_id = str(media.get('id'))
            title_dict = media.get('title', {})
            title = title_dict.get('english') or title_dict.get('romaji') or m_id
            
            media_info[m_id] = {
                'id': m_id,
                'title': title,
                'format': media.get('format', ''),
                'popularity': media.get('popularity', 0),
                'year': media.get('seasonYear') or (media.get('startDate', {}) or {}).get('year') or 2050
            }
            
            relations = media.get('relations', {}).get('edges', [])
            for edge in relations:
                r_type = edge.get('relationType')
                node_id = str(edge.get('node', {}).get('id'))
                if r_type in valid_types:
                    adj[m_id].add(node_id)
                    adj[node_id].add(m_id)
                    
    # Find connected components (Franchises)
    visited = set()
    components = []
    
    for m_id in list(media_info.keys()) + list(adj.keys()):
        if m_id not in visited:
            comp = set()
            stack = [m_id]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    stack.extend(adj[curr] - visited)
            components.append(comp)
            
    # Map components to a franchise ID and name
    df_ids = set(df['mediaId'].astype(str))
    
    franchises = []
    # Usiamo un dizionario per ottimizzare le ricerche nel dataframe
    df_dict = df.set_index('mediaId').to_dict('index')
    
    for comp in components:
        # Intersect with what the user actually watched/read (in df)
        watched_in_comp = comp.intersection(df_ids)
        if not watched_in_comp:
            continue
            
        # Determine Franchise Name:
        comp_info = [media_info.get(mid) for mid in comp if mid in media_info]
        if not comp_info:
            continue
            
        # Prioritize main formats
        main_formats = ['TV', 'MANGA']
        main_items = [m for m in comp_info if m['format'] in main_formats]
        
        if main_items:
            # oldest main item
            main_items.sort(key=lambda x: (x['year'], -x['popularity']))
            franchise_name = main_items[0]['title']
        else:
            # oldest overall
            comp_info.sort(key=lambda x: (x['year'], -x['popularity']))
            franchise_name = comp_info[0]['title']
            
        # Compute stats for this franchise
        total_score_sum = 0.0
        total_entries = 0
        total_progress = 0
        
        for mid in watched_in_comp:
            # get user score from pre-computed dict
            row = df_dict.get(int(mid) if mid.isdigit() else mid)
            if not row: continue
            
            score = row['user_score']
            prog = row.get('progress', 0)
            if pd.isna(prog):
                prog = 0
                
            total_score_sum += score
            total_progress += prog
            total_entries += 1
            
        if total_entries > 0:
            avg_score = total_score_sum / total_entries
            
            # Bonus per entry (stagioni, film, speciali valgono uguale)
            entry_bonus = total_entries * 0.05
            
            # Bonus per lunghezza (episodi/capitoli totali)
            # Anime: ~0.1 punti ogni 50 episodi. Manga: ~0.1 punti ogni 100 capitoli.
            progress_multiplier = 0.001 if is_manga else 0.002
            progress_bonus = total_progress * progress_multiplier
            
            presence_bonus = entry_bonus + progress_bonus
            final_score = avg_score + presence_bonus
            
            franchises.append({
                'franchise': franchise_name,
                'avg_score': round(avg_score, 2),
                'final_score': round(final_score, 2),
                'total_entries': total_entries,
                'total_progress': int(total_progress)
            })
            
    res_df = pd.DataFrame(franchises)
    if not res_df.empty:
        res_df = res_df.sort_values('final_score', ascending=False)
        
    return res_df


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
    direction = "generous" if mean_diff > 0.2 else ("strict" if mean_diff < -0.2 else "balanced")
    
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

# ========== ACTIVITY ANALYTICS ==========

def _get_activity_equivalent_units(media, progress_units, media_type, anime_reference_minutes=24.0):
    """Normalize activity units to an anime-episode-equivalent scale."""
    if media_type == "ANIME":
        duration = media.get("duration") or 24
        return progress_units * (float(duration) / anime_reference_minutes)

    media_format = str(media.get("format") or "").upper()
    country = str(media.get("countryOfOrigin") or "").upper()

    # Heuristic weights tuned for readability:
    # - JP manga chapter: around 15-25 pages and anime often adapts ~2-3 chapters per TV episode
    # - KR/CN web chapters are usually longer/denser in scrolling format, so they count a bit more
    # - Novel chapters are prose and usually slower to read
    # - One-shots are often a full self-contained short work in one chapter
    if media_format == "NOVEL":
        chapters_per_episode = 1.5
    elif media_format == "ONE_SHOT":
        chapters_per_episode = 1.0
    elif country == "KR":
        chapters_per_episode = 2.0
    elif country == "CN":
        chapters_per_episode = 2.0
    else:
        chapters_per_episode = 2.5

    return progress_units / chapters_per_episode


def compute_activity_stats(activities, media_type="ANIME", anime_reference_minutes=24.0):
    """Compute analytical stats from a user's activity history."""
    if not activities:
        return None
        
    valid_statuses = ['watched episode', 'completed'] if media_type == "ANIME" else ['read chapter', 'completed']
    rows = []
    
    for act in activities:
        st = act.get('status')
        if st in valid_statuses:
            dt = datetime.fromtimestamp(act['createdAt'])
            media = act.get('media', {}) or {}
            dur = media.get('duration') or 0
            if media_type == "MANGA":
                dur = 5
                
            ep = 1
            prog = act.get('progress')
            if prog and '-' in str(prog):
                parts = str(prog).split('-')
                try:
                    p_start = int(parts[0].strip())
                    p_end = int(parts[1].strip())
                    if p_end >= p_start:
                        ep = p_end - p_start + 1
                except ValueError:
                    pass
            
            title_dict = media.get('title', {})
            t_str = title_dict.get('english') or title_dict.get('romaji') or "Unknown"
            equivalent_units = _get_activity_equivalent_units(
                media,
                ep,
                media_type=media_type,
                anime_reference_minutes=anime_reference_minutes,
            )
            
            rows.append({
                'date': dt.date(),
                'month': dt.strftime('%Y-%m'),
                'year': str(dt.year),
                'duration_hours': (dur * ep) / 60.0,
                'episodes': ep,
                'equivalent_units': equivalent_units,
                'media_type': media_type,
                'media_format': str(media.get('format') or '').upper(),
                'country_of_origin': str(media.get('countryOfOrigin') or '').upper(),
                'title': t_str
            })
            
    if not rows:
        return None
        
    df = pd.DataFrame(rows)
    
    # By month
    month_stats = df.groupby('month').agg({
        'episodes': 'sum',
        'equivalent_units': 'sum',
        'duration_hours': 'sum'
    }).reset_index().sort_values('month')
    # By year
    year_stats = df.groupby('year').agg({
        'episodes': 'sum',
        'equivalent_units': 'sum',
        'duration_hours': 'sum'
    }).reset_index().sort_values('year')
    # Max day
    day_stats = df.groupby('date').agg({
        'episodes': 'sum', 
        'equivalent_units': 'sum',
        'duration_hours': 'sum',
        'title': lambda x: list(set(x))
    }).reset_index()
    
    if day_stats.empty:
        return None
        
    max_day_row = day_stats.loc[day_stats['episodes'].idxmax()]
    
    return {
        'month_stats': month_stats,
        'year_stats': year_stats,
        'rows': df,
        'max_day': {
            'date': str(max_day_row['date']),
            'episodes': int(max_day_row['episodes']),
            'equivalent_units': round(float(max_day_row['equivalent_units']), 2),
            'hours': float(max_day_row['duration_hours']),
            'titles': max_day_row['title']
        }
    }
