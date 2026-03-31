import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def _get_list(val):
    if pd.isna(val) or not val:
        return []
    return [x.strip() for x in str(val).split(",")]

def _get_tags_with_ranks(val):
    if pd.isna(val) or not val:
        return {}
    res = {}
    for x in str(val).split(","):
        x = x.strip()
        if not x: continue
        parts = x.rsplit('=', 1)
        if len(parts) == 2:
            try:
                res[parts[0]] = float(parts[1])
            except:
                res[parts[0]] = 0.0
        else:
            res[parts[0]] = 100.0
    return res

def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates leakage-free historical features using a running state over the chronologically sorted dataset.
    """
    # Initialize running state
    state = {
        'sum_scores': 0.0,
        'count': 0,
        'genre_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'tag_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'studio_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'format_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'diff_from_global': {'sum': 0.0, 'count': 0} # Regular dict, not defaultdict
    }
    
    historical_features = []
    
    for i, row in df.iterrows():
        # --- 1. Compute features using current state (PAST knowledge only) ---
        user_mean = state['sum_scores'] / state['count'] if state['count'] > 0 else np.nan
        
        # Format affinity
        fmt = row['format']
        fmt_mean = state['format_scores'][fmt]['sum'] / state['format_scores'][fmt]['count'] if state['format_scores'][fmt]['count'] > 0 else np.nan
        
        # Genre affinity
        genres = _get_list(row['genres'])
        genre_means = []
        for g in genres:
            if state['genre_scores'][g]['count'] > 0:
                genre_means.append(state['genre_scores'][g]['sum'] / state['genre_scores'][g]['count'])
        genre_affinity = np.mean(genre_means) if len(genre_means) > 0 else np.nan
        
        # Tag affinity (weighted by current anime tag rank)
        tags_dict = _get_tags_with_ranks(row['tags'])
        tag_means = []
        weights = []
        for t, rank in tags_dict.items():
            if state['tag_scores'][t]['count'] > 0:
                score_for_tag = state['tag_scores'][t]['sum'] / state['tag_scores'][t]['count']
                tag_means.append(score_for_tag)
                weights.append(rank + 1.0) # Avoid 0 weight zeroing out valid historical means
                
        tag_affinity = np.average(tag_means, weights=weights) if len(tag_means) > 0 else np.nan
        
        # Studio affinity
        studios = _get_list(row['studios'])
        studio_means = []
        for s in studios:
            if state['studio_scores'][s]['count'] > 0:
                studio_means.append(state['studio_scores'][s]['sum'] / state['studio_scores'][s]['count'])
        studio_affinity = np.mean(studio_means) if len(studio_means) > 0 else np.nan
        
        # Global critic alignment (does user usually rate higher or lower than MAL/AniList global?)
        user_global_diff = state['diff_from_global']['sum'] / state['diff_from_global']['count'] if state['diff_from_global']['count'] > 0 else 0.0
        
        historical_features.append({
            'hist_user_mean': user_mean,
            'hist_count': state['count'],
            'hist_format_affinity': fmt_mean,
            'hist_genre_affinity': genre_affinity,
            'hist_tag_affinity': tag_affinity,
            'hist_studio_affinity': studio_affinity,
            'hist_global_diff': user_global_diff
        })
        
        # --- 2. Update state with CURRENT row ---
        score = row['user_score']
        state['sum_scores'] += score
        state['count'] += 1
        
        if pd.notna(fmt):
            state['format_scores'][fmt]['sum'] += score
            state['format_scores'][fmt]['count'] += 1
            
        for g in genres:
            state['genre_scores'][g]['sum'] += score
            state['genre_scores'][g]['count'] += 1
            
        for t, rank in tags_dict.items():
            state['tag_scores'][t]['sum'] += score
            state['tag_scores'][t]['count'] += 1
            
        for s in studios:
            state['studio_scores'][s]['sum'] += score
            state['studio_scores'][s]['count'] += 1
            
        global_score = row['averageScore']
        if pd.notna(global_score) and global_score > 0:
            # AniList global score is 0-100, scale to 0-10
            global_10 = global_score / 10.0
            diff = score - global_10
            state['diff_from_global']['sum'] += diff
            state['diff_from_global']['count'] += 1

    hist_df = pd.DataFrame(historical_features)
    return pd.concat([df, hist_df], axis=1)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes cleaned dataframe and outputs model-ready dataset"""
    if df.empty:
        return df
        
    # Ensure chronological order
    if 'sort_date' not in df.columns:
        df['sort_date'] = df['completedAt'].fillna(df['updatedAt'])
        
    df['sort_date'] = pd.to_datetime(df['sort_date'], errors='coerce')
    df = df.sort_values('sort_date').reset_index(drop=True)
    
    # Create lag / historical features
    df = create_historical_features(df)
    
    # Fill NAs in historical features with global fallbacks
    # e.g., if genre_affinity is NaN, fallback to user_mean, then to global fixed mean (e.g., 5.0)
    df['hist_user_mean'] = df['hist_user_mean'].fillna(df['user_score'].mean() if len(df) > 0 else 5.0)
    for col in ['hist_format_affinity', 'hist_genre_affinity', 'hist_tag_affinity', 'hist_studio_affinity']:
        df[col] = df[col].fillna(df['hist_user_mean'])
        
    # Standard Anime Numeric Features
    df['episodes'] = df['episodes'].fillna(0)
    df['duration'] = df['duration'].fillna(df['duration'].median() if not df['duration'].isna().all() else 24.0)
    df['seasonYear'] = df['seasonYear'].fillna(df['seasonYear'].median() if not df['seasonYear'].isna().all() else 2015.0)
    
    avg_score_median = df['averageScore'].median() if not df['averageScore'].isna().all() else 70.0
    df['averageScore'] = df['averageScore'].fillna(avg_score_median) / 10.0 # scale 0-10
    
    pop_median = df['popularity'].median() if not df['popularity'].isna().all() else 1000.0
    df['popularity'] = np.log1p(df['popularity'].fillna(pop_median)) # Log transform for heavy skew
    
    # Anime Categorical (One-Hot / Multi-Hot)
    # 1. Base Dummies
    categorical_cols = ['format', 'season', 'source']
    df_cat = pd.get_dummies(df[categorical_cols], dummy_na=True, drop_first=False)
    
    # 2. All Genres Multi-Hot
    all_genres = set()
    for row_genres in df['genres']:
        all_genres.update(_get_list(row_genres))
        
    for g in all_genres:
        if not g: continue
        df_cat[f'genre_{g}'] = df['genres'].apply(lambda x: 1.0 if g in _get_list(x) else 0.0)
        
    # 3. All Tags Multi-Hot with Ranks
    all_tags = set()
    for row_tags in df['tags']:
        all_tags.update(_get_tags_with_ranks(row_tags).keys())
        
    for t in all_tags:
        if not t: continue
        df_cat[f'tag_{t}'] = df['tags'].apply(lambda x: _get_tags_with_ranks(x).get(t, 0.0) / 100.0)

    # Combine Base, Numerical, Historical, and Categorical
    num_cols = ['episodes', 'duration', 'seasonYear', 'averageScore', 'popularity']
    hist_cols = ['hist_user_mean', 'hist_count', 'hist_format_affinity', 'hist_genre_affinity', 
                 'hist_tag_affinity', 'hist_studio_affinity', 'hist_global_diff']
                 
    X = pd.concat([df[num_cols + hist_cols], df_cat], axis=1)
    y = df['user_score']
    
    return X, y

def temporal_train_val_test_split(X: pd.DataFrame, y: pd.Series, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the DataFrames chronologically.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * (1.0 - val_ratio - test_ratio))
    val_end = int(n * (1.0 - test_ratio))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    logger.info(f"Chronological Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_inference_features(anime_data_dict, user_history_df, train_columns):
    """
    Given a new anime dict from API and user history df, construct the 1-row feature matrix
    Expected to return exactly the same columns as the model expects.
    This reconstructs the state just for this new anime using the ENTIRE user_history_df.
    """
    title_dict = anime_data_dict.get('title', {})
    
    studios_data = anime_data_dict.get('studios', {}).get('edges', [])
    main_studios = [s['node']['name'] for s in studios_data if s.get('isMain')]
    all_studios = [s['node']['name'] for s in studios_data]
    studio_str = ", ".join(main_studios) if main_studios else ", ".join(all_studios)
    
    tags_str_list = []
    for t in anime_data_dict.get('tags', []):
        t_name = str(t['name']).replace('=', '-').replace(',', '')
        t_rank = t.get('rank', 0)
        tags_str_list.append(f"{t_name}={t_rank}")
    
    row = {
        'user_score': np.nan, # Unknown target
        'format': anime_data_dict.get('format'),
        'episodes': anime_data_dict.get('episodes'),
        'duration': anime_data_dict.get('duration'),
        'season': anime_data_dict.get('season'),
        'seasonYear': anime_data_dict.get('seasonYear'),
        'averageScore': anime_data_dict.get('averageScore'),
        'popularity': anime_data_dict.get('popularity'),
        'genres': ", ".join(anime_data_dict.get('genres', [])),
        'tags': ", ".join(tags_str_list),
        'studios': studio_str,
        'sort_date': pd.Timestamp.now() # Put at the end of chronological history
    }
    
    new_df = pd.DataFrame([row])
    combined = pd.concat([user_history_df, new_df], ignore_index=True)
    
    # Engineer all features (historical states + one-hots)
    X, _ = engineer_features(combined)
    
    # Take the last row (our target anime)
    X_last = X.iloc[[-1]].copy()
    
    # Align columns to match training exactly
    for col in train_columns:
        if col not in X_last.columns:
            X_last[col] = 0
            
    X_inference = X_last[train_columns]
    X_inference = X_inference.fillna(0)
    
    return X_inference
