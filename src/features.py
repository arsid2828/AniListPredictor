import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, Any, Optional

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
        'scores_list': [],
        'genre_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'tag_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'studio_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'producer_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'creator_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'format_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'diff_from_global': {'sum': 0.0, 'abs_sum': 0.0, 'count': 0}, # Regular dict
        'scores_by_id': {}
    }
    
    historical_features = []
    
    for i, row in df.iterrows():
        # --- 1. Compute features using current state (PAST knowledge only) ---
        user_mean = state['sum_scores'] / state['count'] if state['count'] > 0 else np.nan
        hist_user_std = np.std(state['scores_list']) if len(state['scores_list']) >= 2 else 0.0
        recent_scores = state['scores_list'][-10:] if len(state['scores_list']) > 0 else []
        recent_mean_score = np.mean(recent_scores) if len(recent_scores) > 0 else np.nan
        
        # Format affinity
        fmt = row['format']
        fmt_mean = state['format_scores'][fmt]['sum'] / state['format_scores'][fmt]['count'] if state['format_scores'][fmt]['count'] > 0 else np.nan
        
        # Genre affinity and freq
        genres = _get_list(row['genres'])
        genre_means = []
        genre_counts = []
        for g in genres:
            if state['genre_scores'][g]['count'] > 0:
                genre_means.append(state['genre_scores'][g]['sum'] / state['genre_scores'][g]['count'])
            if state['count'] > 0:
                genre_counts.append(state['genre_scores'][g]['count'] / state['count'])
        genre_affinity = np.mean(genre_means) if len(genre_means) > 0 else np.nan
        hist_genre_freq = np.mean(genre_counts) if len(genre_counts) > 0 else 0.0
        
        # Tag affinity and freq
        tags_dict = _get_tags_with_ranks(row['tags'])
        tag_means = []
        weights = []
        tag_counts = []
        for t, rank in tags_dict.items():
            if state['tag_scores'][t]['count'] > 0:
                score_for_tag = state['tag_scores'][t]['sum'] / state['tag_scores'][t]['count']
                tag_means.append(score_for_tag)
                weights.append(rank + 1.0)
            if state['count'] > 0:
                tag_counts.append(state['tag_scores'][t]['count'] / state['count'])
                
        tag_affinity = np.average(tag_means, weights=weights) if len(tag_means) > 0 else np.nan
        hist_tag_freq = np.mean(tag_counts) if len(tag_counts) > 0 else 0.0
        
        # Studio affinity and freq
        studios = _get_list(row['studios'])
        studio_means = []
        studio_counts = []
        for s in studios:
            if state['studio_scores'][s]['count'] > 0:
                studio_means.append(state['studio_scores'][s]['sum'] / state['studio_scores'][s]['count'])
            if state['count'] > 0:
                studio_counts.append(state['studio_scores'][s]['count'] / state['count'])
        studio_affinity = np.mean(studio_means) if len(studio_means) > 0 else np.nan
        hist_studio_freq = np.mean(studio_counts) if len(studio_counts) > 0 else 0.0
        
        # Producer affinity and freq
        producers = _get_list(row.get('producers', ''))
        producer_means = []
        producer_counts = []
        for p in producers:
            if state['producer_scores'][p]['count'] > 0:
                producer_means.append(state['producer_scores'][p]['sum'] / state['producer_scores'][p]['count'])
            if state['count'] > 0:
                producer_counts.append(state['producer_scores'][p]['count'] / state['count'])
        producer_affinity = np.mean(producer_means) if len(producer_means) > 0 else np.nan
        hist_producer_freq = np.mean(producer_counts) if len(producer_counts) > 0 else 0.0
        
        # Creator affinity and freq
        creators = _get_list(row.get('creators', ''))
        creator_means = []
        creator_counts = []
        for c in creators:
            if state['creator_scores'][c]['count'] > 0:
                creator_means.append(state['creator_scores'][c]['sum'] / state['creator_scores'][c]['count'])
            if state['count'] > 0:
                creator_counts.append(state['creator_scores'][c]['count'] / state['count'])
        creator_affinity = np.mean(creator_means) if len(creator_means) > 0 else np.nan
        hist_creator_freq = np.mean(creator_counts) if len(creator_counts) > 0 else 0.0
        
        # Global critic alignment (does user usually rate higher or lower than MAL/AniList global?)
        user_global_diff = state['diff_from_global']['sum'] / state['diff_from_global']['count'] if state['diff_from_global']['count'] > 0 else 0.0
        user_global_mae = state['diff_from_global']['abs_sum'] / state['diff_from_global']['count'] if state['diff_from_global']['count'] > 0 else 0.0
        
        def calc_related_mean(id_str_list):
            if pd.isna(id_str_list) or not str(id_str_list).strip(): return 0, np.nan
            ids = [int(x) for x in str(id_str_list).split(',')] 
            match_scores = [state['scores_by_id'][i] for i in ids if i in state['scores_by_id']]
            if not match_scores: return 0, np.nan
            return len(match_scores), np.mean(match_scores)

        hist_franchise_count, hist_franchise_mean = calc_related_mean(row.get('related_ids'))
        hist_char_count, hist_char_mean = calc_related_mean(row.get('char_related_ids'))
        
        historical_features.append({
            'hist_user_mean': user_mean,
            'hist_user_std': hist_user_std,
            'recent_mean_score': recent_mean_score,
            'hist_count': state['count'],
            'hist_format_affinity': fmt_mean,
            'hist_genre_affinity': genre_affinity,
            'hist_genre_freq': hist_genre_freq,
            'hist_tag_affinity': tag_affinity,
            'hist_tag_freq': hist_tag_freq,
            'hist_studio_affinity': studio_affinity,
            'hist_studio_freq': hist_studio_freq,
            'hist_producer_affinity': producer_affinity,
            'hist_producer_freq': hist_producer_freq,
            'hist_creator_affinity': creator_affinity,
            'hist_creator_freq': hist_creator_freq,
            'hist_franchise_count': hist_franchise_count,
            'hist_franchise_mean': hist_franchise_mean,
            'hist_char_count': hist_char_count,
            'hist_char_mean': hist_char_mean,
            'hist_global_diff': user_global_diff,
            'hist_global_mae': user_global_mae
        })
        
        # --- 2. Update state with CURRENT row ---
        score = row['user_score']
        state['sum_scores'] += score
        state['scores_list'].append(score)
        state['count'] += 1
        if pd.notna(row.get('mediaId')):
            state['scores_by_id'][int(row['mediaId'])] = score
        
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
            
        for p in producers:
            state['producer_scores'][p]['sum'] += score
            state['producer_scores'][p]['count'] += 1
            
        for c in creators:
            state['creator_scores'][c]['sum'] += score
            state['creator_scores'][c]['count'] += 1
            
        global_score = row['averageScore']
        if pd.notna(global_score) and global_score > 0:
            # AniList global score is 0-100, scale to 0-10
            global_10 = global_score / 10.0
            diff = score - global_10
            state['diff_from_global']['sum'] += diff
            state['diff_from_global']['abs_sum'] += abs(diff)
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
    df['sort_date'] = df['sort_date'].fillna(pd.Timestamp('1970-01-01'))
    df = df.sort_values('sort_date').reset_index(drop=True)
    
    # Create lag / historical features
    df = create_historical_features(df)
    
    # Fill NAs in historical features with global fallbacks
    # e.g., if genre_affinity is NaN, fallback to user_mean, then to global fixed mean (e.g., 5.0)
    df['hist_user_mean'] = df['hist_user_mean'].fillna(df['user_score'].mean() if len(df) > 0 else 5.0)
    df['recent_mean_score'] = df['recent_mean_score'].fillna(df['hist_user_mean'])
    df['hist_franchise_count'] = df['hist_franchise_count'].fillna(0)
    df['hist_char_count'] = df['hist_char_count'].fillna(0)
    for col in ['hist_format_affinity', 'hist_genre_affinity', 'hist_tag_affinity', 'hist_studio_affinity', 'hist_producer_affinity', 'hist_creator_affinity', 'hist_franchise_mean', 'hist_char_mean']:
        df[col] = df[col].fillna(df['hist_user_mean'])
        
    # Standard Anime Numeric Features
    df['episodes'] = df['episodes'].fillna(0)
    df['duration'] = df['duration'].fillna(df['duration'].median() if not df['duration'].isna().all() else 24.0)
    df['seasonYear'] = df['seasonYear'].fillna(df['seasonYear'].median() if not df['seasonYear'].isna().all() else 2015.0)
    
    # Anime unreleased might have 0 as averageScore, which devastates the model.
    df['averageScore'] = df['averageScore'].replace(0, np.nan)
    avg_score_median = df['averageScore'].median() if not df['averageScore'].isna().all() else 70.0
    df['averageScore'] = df['averageScore'].fillna(avg_score_median) / 10.0 # scale 0-10
    
    pop_median = df['popularity'].median() if not df['popularity'].isna().all() else 1000.0
    if 'media_status' in df.columns:
        unreleased_mask = df['media_status'] == 'NOT_YET_RELEASED'
        df.loc[unreleased_mask, 'popularity'] = np.nan
    df['popularity'] = np.log1p(df['popularity'].fillna(pop_median)) # Log transform for heavy skew
    
    fav_median = df['favourites'].median() if not df['favourites'].isna().all() else 10.0
    if 'media_status' in df.columns:
        unreleased_mask = df['media_status'] == 'NOT_YET_RELEASED'
        df.loc[unreleased_mask, 'favourites'] = np.nan
    df['favourites'] = np.log1p(df['favourites'].fillna(fav_median))
    
    df['isAdult'] = df['isAdult'].fillna(0).astype(int)
    
    # New features: source material, completion ratio, rewatch count
    df['source'] = df['source'].fillna('UNKNOWN')
    df['progress'] = df['progress'].fillna(0)
    df['completion_ratio'] = (df['progress'] / df['episodes'].replace(0, np.nan)).fillna(1.0).clip(0, 1)
    df['repeat'] = df['repeat'].fillna(0).astype(int)
    
    # Anime Categorical (One-Hot / Multi-Hot)
    # 1. Base Dummies
    categorical_cols = ['format', 'countryOfOrigin', 'source']
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
    num_cols = ['episodes', 'duration', 'seasonYear', 'averageScore', 'popularity', 'favourites', 'isAdult', 'completion_ratio', 'repeat']
    hist_cols = ['hist_user_mean', 'hist_user_std', 'recent_mean_score', 'hist_count', 'hist_format_affinity', 'hist_genre_affinity', 'hist_genre_freq', 
                 'hist_tag_affinity', 'hist_tag_freq', 'hist_studio_affinity', 'hist_studio_freq', 'hist_producer_affinity', 'hist_producer_freq', 'hist_creator_affinity', 'hist_creator_freq', 'hist_franchise_count', 'hist_franchise_mean', 'hist_char_count', 'hist_char_mean', 'hist_global_diff', 'hist_global_mae']
                 
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


def _mean_or_nan(sum_count_entry):
    if sum_count_entry['count'] > 0:
        return sum_count_entry['sum'] / sum_count_entry['count']
    return np.nan


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _candidate_related_stats(id_str_list, scores_by_id):
    if not id_str_list:
        return 0, np.nan
    ids = [int(x) for x in str(id_str_list).split(',') if str(x).strip()]
    match_scores = [scores_by_id[i] for i in ids if i in scores_by_id]
    if not match_scores:
        return 0, np.nan
    return len(match_scores), float(np.mean(match_scores))


def _collect_history_state(user_history_df: pd.DataFrame):
    if user_history_df.empty:
        return {
            'user_mean': 5.0,
            'user_std': 0.0,
            'recent_mean': 5.0,
            'hist_count': 0,
            'genre_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
            'tag_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
            'studio_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
            'producer_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
            'creator_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
            'format_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
            'diff_from_global': {'sum': 0.0, 'abs_sum': 0.0, 'count': 0},
            'scores_by_id': {},
        }

    state = {
        'sum_scores': 0.0,
        'count': 0,
        'scores_list': [],
        'genre_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'tag_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'studio_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'producer_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'creator_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'format_scores': defaultdict(lambda: {'sum': 0.0, 'count': 0}),
        'diff_from_global': {'sum': 0.0, 'abs_sum': 0.0, 'count': 0},
        'scores_by_id': {},
    }

    for _, row in user_history_df.iterrows():
        score = float(row['user_score'])
        state['sum_scores'] += score
        state['scores_list'].append(score)
        state['count'] += 1

        media_id = row.get('mediaId')
        if pd.notna(media_id):
            state['scores_by_id'][int(media_id)] = score

        fmt = row.get('format')
        if pd.notna(fmt):
            state['format_scores'][fmt]['sum'] += score
            state['format_scores'][fmt]['count'] += 1

        for g in _get_list(row.get('genres', '')):
            state['genre_scores'][g]['sum'] += score
            state['genre_scores'][g]['count'] += 1

        for t in _get_tags_with_ranks(row.get('tags', '')).keys():
            state['tag_scores'][t]['sum'] += score
            state['tag_scores'][t]['count'] += 1

        for s in _get_list(row.get('studios', '')):
            state['studio_scores'][s]['sum'] += score
            state['studio_scores'][s]['count'] += 1

        for p in _get_list(row.get('producers', '')):
            state['producer_scores'][p]['sum'] += score
            state['producer_scores'][p]['count'] += 1

        for c in _get_list(row.get('creators', '')):
            state['creator_scores'][c]['sum'] += score
            state['creator_scores'][c]['count'] += 1

        global_score = row.get('averageScore')
        if pd.notna(global_score) and float(global_score) > 0:
            global_10 = float(global_score) / 10.0
            diff = score - global_10
            state['diff_from_global']['sum'] += diff
            state['diff_from_global']['abs_sum'] += abs(diff)
            state['diff_from_global']['count'] += 1

    return {
        'user_mean': state['sum_scores'] / state['count'] if state['count'] > 0 else 5.0,
        'user_std': float(np.std(state['scores_list'])) if len(state['scores_list']) >= 2 else 0.0,
        'recent_mean': float(np.mean(state['scores_list'][-10:])) if state['scores_list'] else 5.0,
        'hist_count': state['count'],
        'genre_scores': state['genre_scores'],
        'tag_scores': state['tag_scores'],
        'studio_scores': state['studio_scores'],
        'producer_scores': state['producer_scores'],
        'creator_scores': state['creator_scores'],
        'format_scores': state['format_scores'],
        'diff_from_global': state['diff_from_global'],
        'scores_by_id': state['scores_by_id'],
    }


def prepare_anime_inference_context(user_history_df: pd.DataFrame, train_columns):
    df = user_history_df.copy()
    if 'sort_date' not in df.columns:
        df['sort_date'] = df['completedAt'].fillna(df['updatedAt'])
    df['sort_date'] = pd.to_datetime(df['sort_date'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))
    df = df.sort_values('sort_date').reset_index(drop=True)
    df['source'] = df.get('source', 'UNKNOWN').fillna('UNKNOWN')

    average_score_series = pd.to_numeric(df.get('averageScore', pd.Series(dtype=float)), errors='coerce').replace(0, np.nan)
    duration_series = pd.to_numeric(df.get('duration', pd.Series(dtype=float)), errors='coerce')
    season_year_series = pd.to_numeric(df.get('seasonYear', pd.Series(dtype=float)), errors='coerce')
    popularity_series = pd.to_numeric(df.get('popularity', pd.Series(dtype=float)), errors='coerce')
    favourites_series = pd.to_numeric(df.get('favourites', pd.Series(dtype=float)), errors='coerce')

    return {
        'history_state': _collect_history_state(df),
        'train_columns': list(train_columns),
        'duration_median': float(duration_series.median()) if not duration_series.dropna().empty else 24.0,
        'season_year_median': float(season_year_series.median()) if not season_year_series.dropna().empty else 2015.0,
        'average_score_median': float(average_score_series.median()) if not average_score_series.dropna().empty else 70.0,
        'popularity_median': float(popularity_series.median()) if not popularity_series.dropna().empty else 1000.0,
        'favourites_median': float(favourites_series.median()) if not favourites_series.dropna().empty else 10.0,
    }


def prepare_manga_inference_context(user_history_df: pd.DataFrame, train_columns):
    df = user_history_df.copy()
    if 'sort_date' not in df.columns:
        df['sort_date'] = df['completedAt'].fillna(df['updatedAt'])
    df['sort_date'] = pd.to_datetime(df['sort_date'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))
    df = df.sort_values('sort_date').reset_index(drop=True)
    if 'studios' not in df.columns:
        df['studios'] = df.get('authors', '')
    if 'producers' not in df.columns:
        df['producers'] = ''
    if 'creators' not in df.columns:
        df['creators'] = df.get('authors', '')

    average_score_series = pd.to_numeric(df.get('averageScore', pd.Series(dtype=float)), errors='coerce').replace(0, np.nan)
    chapters_series = pd.to_numeric(df.get('chapters', pd.Series(dtype=float)), errors='coerce')
    volumes_series = pd.to_numeric(df.get('volumes', pd.Series(dtype=float)), errors='coerce')
    release_year_series = pd.to_numeric(df.get('releaseYear', pd.Series(dtype=float)), errors='coerce')
    popularity_series = pd.to_numeric(df.get('popularity', pd.Series(dtype=float)), errors='coerce')
    favourites_series = pd.to_numeric(df.get('favourites', pd.Series(dtype=float)), errors='coerce')

    return {
        'history_state': _collect_history_state(df),
        'train_columns': list(train_columns),
        'chapters_median': float(chapters_series.median()) if not chapters_series.dropna().empty else 0.0,
        'volumes_median': float(volumes_series.median()) if not volumes_series.dropna().empty else 0.0,
        'release_year_median': float(release_year_series.median()) if not release_year_series.dropna().empty else 2015.0,
        'average_score_median': float(average_score_series.median()) if not average_score_series.dropna().empty else 70.0,
        'popularity_median': float(popularity_series.median()) if not popularity_series.dropna().empty else 1000.0,
        'favourites_median': float(favourites_series.median()) if not favourites_series.dropna().empty else 10.0,
    }


def _fill_candidate_category_features(feature_row: Dict[str, Any], train_columns, *, fmt=None, country=None, source=None, genres=None, tag_ranks=None):
    genres = set(genres or [])
    tag_ranks = tag_ranks or {}
    for col in train_columns:
        if col.startswith('format_'):
            feature_row[col] = 1.0 if str(fmt) == col[len('format_'):] else 0.0
        elif col == 'format_nan':
            feature_row[col] = 1.0 if pd.isna(fmt) or fmt is None else 0.0
        elif col.startswith('countryOfOrigin_'):
            feature_row[col] = 1.0 if str(country) == col[len('countryOfOrigin_'):] else 0.0
        elif col == 'countryOfOrigin_nan':
            feature_row[col] = 1.0 if pd.isna(country) or country is None else 0.0
        elif source is not None and col.startswith('source_'):
            feature_row[col] = 1.0 if str(source) == col[len('source_'):] else 0.0
        elif source is not None and col == 'source_nan':
            feature_row[col] = 1.0 if pd.isna(source) or source is None else 0.0
        elif col.startswith('genre_'):
            feature_row[col] = 1.0 if col[len('genre_'):] in genres else 0.0
        elif col.startswith('tag_'):
            feature_row[col] = float(tag_ranks.get(col[len('tag_'):], 0.0)) / 100.0


def _build_common_historical_feature_values(history_state, fmt, genres, tags_dict, studios, producers, creators, related_ids_str, char_related_ids_str):
    user_mean = history_state['user_mean']
    hist_count = history_state['hist_count']

    genre_means = [_mean_or_nan(history_state['genre_scores'][g]) for g in genres if history_state['genre_scores'][g]['count'] > 0]
    genre_counts = [history_state['genre_scores'][g]['count'] / hist_count for g in genres if hist_count > 0]

    tag_means = []
    tag_weights = []
    tag_counts = []
    for tag_name, rank in tags_dict.items():
        if history_state['tag_scores'][tag_name]['count'] > 0:
            tag_means.append(_mean_or_nan(history_state['tag_scores'][tag_name]))
            tag_weights.append(rank + 1.0)
        if hist_count > 0:
            tag_counts.append(history_state['tag_scores'][tag_name]['count'] / hist_count)

    studio_means = [_mean_or_nan(history_state['studio_scores'][s]) for s in studios if history_state['studio_scores'][s]['count'] > 0]
    studio_counts = [history_state['studio_scores'][s]['count'] / hist_count for s in studios if hist_count > 0]

    producer_means = [_mean_or_nan(history_state['producer_scores'][p]) for p in producers if history_state['producer_scores'][p]['count'] > 0]
    producer_counts = [history_state['producer_scores'][p]['count'] / hist_count for p in producers if hist_count > 0]

    creator_means = [_mean_or_nan(history_state['creator_scores'][c]) for c in creators if history_state['creator_scores'][c]['count'] > 0]
    creator_counts = [history_state['creator_scores'][c]['count'] / hist_count for c in creators if hist_count > 0]

    hist_franchise_count, hist_franchise_mean = _candidate_related_stats(related_ids_str, history_state['scores_by_id'])
    hist_char_count, hist_char_mean = _candidate_related_stats(char_related_ids_str, history_state['scores_by_id'])
    global_count = history_state['diff_from_global']['count']

    return {
        'hist_user_mean': user_mean,
        'hist_user_std': history_state['user_std'],
        'recent_mean_score': history_state['recent_mean'],
        'hist_count': hist_count,
        'hist_format_affinity': _mean_or_nan(history_state['format_scores'][fmt]) if history_state['format_scores'][fmt]['count'] > 0 else user_mean,
        'hist_genre_affinity': float(np.mean(genre_means)) if genre_means else user_mean,
        'hist_genre_freq': _mean_or_zero(genre_counts),
        'hist_tag_affinity': float(np.average(tag_means, weights=tag_weights)) if tag_means else user_mean,
        'hist_tag_freq': _mean_or_zero(tag_counts),
        'hist_studio_affinity': float(np.mean(studio_means)) if studio_means else user_mean,
        'hist_studio_freq': _mean_or_zero(studio_counts),
        'hist_producer_affinity': float(np.mean(producer_means)) if producer_means else user_mean,
        'hist_producer_freq': _mean_or_zero(producer_counts),
        'hist_creator_affinity': float(np.mean(creator_means)) if creator_means else user_mean,
        'hist_creator_freq': _mean_or_zero(creator_counts),
        'hist_franchise_count': hist_franchise_count,
        'hist_franchise_mean': hist_franchise_mean if pd.notna(hist_franchise_mean) else user_mean,
        'hist_char_count': hist_char_count,
        'hist_char_mean': hist_char_mean if pd.notna(hist_char_mean) else user_mean,
        'hist_global_diff': history_state['diff_from_global']['sum'] / global_count if global_count > 0 else 0.0,
        'hist_global_mae': history_state['diff_from_global']['abs_sum'] / global_count if global_count > 0 else 0.0,
    }


def build_inference_feature_row(anime_data_dict, inference_context):
    train_columns = inference_context['train_columns']
    history_state = inference_context['history_state']

    studios_data = anime_data_dict.get('studios', {}).get('edges', [])
    main_studios = [s['node']['name'] for s in studios_data if s.get('isMain')]
    producers = [s['node']['name'] for s in studios_data if not s.get('isMain')]
    all_studios = [s['node']['name'] for s in studios_data]
    studio_str = ", ".join(main_studios) if main_studios else ", ".join(all_studios)
    producer_str = ", ".join(producers)

    staff_data = anime_data_dict.get('staff', {}).get('edges', [])
    creators = []
    for s in staff_data:
        role = (s.get('role') or '').lower()
        name = s.get('node', {}).get('name', {}).get('full', '')
        if name and ('director' in role or 'original creator' in role or 'original story' in role or 'series composition' in role or 'original character design' in role):
            creators.append(name)
    creator_str = ", ".join(creators) if creators else ""

    tags_dict = {}
    for t in anime_data_dict.get('tags', []):
        t_name = str(t['name']).replace('=', '-').replace(',', '')
        tags_dict[t_name] = t.get('rank', 0)

    relations_data = anime_data_dict.get('relations', {}).get('edges', [])
    related_ids = []
    char_related_ids = []
    valid_types = ['ADAPTATION', 'PREQUEL', 'SEQUEL', 'PARENT', 'SIDE_STORY', 'SUMMARY', 'ALTERNATIVE', 'SPIN_OFF', 'OTHER', 'SOURCE', 'COMPILATION', 'CONTAINS']
    for edge in relations_data:
        node_id = edge.get('node', {}).get('id')
        if not node_id:
            continue
        r_type = edge.get('relationType')
        if r_type == 'CHARACTER':
            char_related_ids.append(str(node_id))
        elif r_type in valid_types or r_type:
            related_ids.append(str(node_id))

    fmt = anime_data_dict.get('format')
    country = anime_data_dict.get('countryOfOrigin')
    source = anime_data_dict.get('source', 'UNKNOWN')
    genres = [str(g).strip() for g in anime_data_dict.get('genres', []) if str(g).strip()]
    studios = _get_list(studio_str)
    producers_list = _get_list(producer_str)
    creators_list = _get_list(creator_str)

    episodes = anime_data_dict.get('episodes') or 0
    duration = anime_data_dict.get('duration')
    if pd.isna(duration) or duration is None:
        duration = inference_context['duration_median']
    season_year = anime_data_dict.get('seasonYear')
    if pd.isna(season_year) or season_year is None:
        season_year = inference_context['season_year_median']
    average_score = anime_data_dict.get('averageScore')
    if pd.isna(average_score) or average_score in (None, 0):
        average_score = inference_context['average_score_median']
    popularity = anime_data_dict.get('popularity')
    if anime_data_dict.get('status') == 'NOT_YET_RELEASED':
        popularity = None
    popularity = np.log1p(inference_context['popularity_median'] if pd.isna(popularity) or popularity is None else popularity)
    favourites = anime_data_dict.get('favourites')
    if anime_data_dict.get('status') == 'NOT_YET_RELEASED':
        favourites = None
    favourites = np.log1p(inference_context['favourites_median'] if pd.isna(favourites) or favourites is None else favourites)

    feature_row = {
        'episodes': float(episodes),
        'duration': float(duration),
        'seasonYear': float(season_year),
        'averageScore': float(average_score) / 10.0,
        'popularity': float(popularity),
        'favourites': float(favourites),
        'isAdult': 1 if anime_data_dict.get('isAdult') else 0,
        'completion_ratio': 1.0 if not episodes else float(np.clip((episodes / episodes), 0, 1)),
        'repeat': 0,
    }
    feature_row.update(
        _build_common_historical_feature_values(
            history_state,
            fmt,
            genres,
            tags_dict,
            studios,
            producers_list,
            creators_list,
            ",".join(related_ids),
            ",".join(char_related_ids),
        )
    )
    _fill_candidate_category_features(
        feature_row,
        train_columns,
        fmt=fmt,
        country=country,
        source=source,
        genres=genres,
        tag_ranks=tags_dict,
    )
    return {col: feature_row.get(col, 0.0) for col in train_columns}


def build_manga_inference_feature_row(manga_data_dict, inference_context):
    train_columns = inference_context['train_columns']
    history_state = inference_context['history_state']

    staff_data = manga_data_dict.get('staff', {}).get('edges', [])
    authors = []
    for s in staff_data:
        role = (s.get('role') or '').lower()
        name = s.get('node', {}).get('name', {}).get('full', '')
        if name and ('story' in role or 'art' in role or 'original' in role):
            authors.append(name)
    author_str = ", ".join(authors) if authors else ""

    tags_dict = {}
    for t in manga_data_dict.get('tags', []):
        t_name = str(t['name']).replace('=', '-').replace(',', '')
        tags_dict[t_name] = t.get('rank', 0)

    relations_data = manga_data_dict.get('relations', {}).get('edges', [])
    related_ids = []
    char_related_ids = []
    valid_types = ['ADAPTATION', 'PREQUEL', 'SEQUEL', 'PARENT', 'SIDE_STORY', 'SUMMARY', 'ALTERNATIVE', 'SPIN_OFF', 'OTHER', 'SOURCE', 'COMPILATION', 'CONTAINS']
    for edge in relations_data:
        node_id = edge.get('node', {}).get('id')
        if not node_id:
            continue
        r_type = edge.get('relationType')
        if r_type == 'CHARACTER':
            char_related_ids.append(str(node_id))
        elif r_type in valid_types or r_type:
            related_ids.append(str(node_id))

    start_date = manga_data_dict.get('startDate', {}) or {}
    release_year = start_date.get('year')
    if pd.isna(release_year) or release_year is None:
        release_year = inference_context['release_year_median']

    fmt = manga_data_dict.get('format')
    country = manga_data_dict.get('countryOfOrigin')
    genres = [str(g).strip() for g in manga_data_dict.get('genres', []) if str(g).strip()]
    authors_list = _get_list(author_str)

    chapters = manga_data_dict.get('chapters') or 0
    volumes = manga_data_dict.get('volumes') or 0
    if not volumes and chapters:
        volumes = np.ceil(float(chapters) / 9.0)
    elif not volumes:
        volumes = inference_context['volumes_median']

    average_score = manga_data_dict.get('averageScore')
    if pd.isna(average_score) or average_score in (None, 0):
        average_score = inference_context['average_score_median']
    popularity = manga_data_dict.get('popularity')
    if manga_data_dict.get('status') == 'NOT_YET_RELEASED':
        popularity = None
    popularity = np.log1p(inference_context['popularity_median'] if pd.isna(popularity) or popularity is None else popularity)
    favourites = manga_data_dict.get('favourites')
    if manga_data_dict.get('status') == 'NOT_YET_RELEASED':
        favourites = None
    favourites = np.log1p(inference_context['favourites_median'] if pd.isna(favourites) or favourites is None else favourites)

    feature_row = {
        'chapters': float(chapters),
        'volumes': float(volumes),
        'releaseYear': float(release_year),
        'averageScore': float(average_score) / 10.0,
        'popularity': float(popularity),
        'favourites': float(favourites),
        'isAdult': 1 if manga_data_dict.get('isAdult') else 0,
        'completion_ratio': 1.0 if not chapters else float(np.clip((chapters / chapters), 0, 1)),
        'repeat': 0,
    }
    feature_row.update(
        _build_common_historical_feature_values(
            history_state,
            fmt,
            genres,
            tags_dict,
            authors_list,
            [],
            authors_list,
            ",".join(related_ids),
            ",".join(char_related_ids),
        )
    )
    _fill_candidate_category_features(
        feature_row,
        train_columns,
        fmt=fmt,
        country=country,
        genres=genres,
        tag_ranks=tags_dict,
    )
    return {col: feature_row.get(col, 0.0) for col in train_columns}

def build_inference_features(anime_data_dict, user_history_df, train_columns, inference_context: Optional[Dict[str, Any]] = None):
    """
    Given a new anime dict from API and user history df, construct the 1-row feature matrix
    Expected to return exactly the same columns as the model expects.
    This reconstructs the state just for this new anime using the ENTIRE user_history_df.
    """
    context = inference_context or prepare_anime_inference_context(user_history_df, train_columns)
    row = build_inference_feature_row(anime_data_dict, context)
    return pd.DataFrame([row], columns=context['train_columns']).fillna(0)

def engineer_manga_features(df: pd.DataFrame):
    """Takes cleaned manga dataframe and outputs model-ready dataset.
    Uses chapters instead of episodes/duration, releaseYear instead of seasonYear,
    and authors instead of studios/producers."""
    if df.empty:
        return df
        
    # Ensure chronological order
    if 'sort_date' not in df.columns:
        df['sort_date'] = df['completedAt'].fillna(df['updatedAt'])
        
    df['sort_date'] = pd.to_datetime(df['sort_date'], errors='coerce')
    df['sort_date'] = df['sort_date'].fillna(pd.Timestamp('1970-01-01'))
    df = df.sort_values('sort_date').reset_index(drop=True)
    
    # For manga, we reuse the same historical feature engine.
    # We need 'studios' and 'producers' columns for create_historical_features to work.
    # Map authors -> studios, leave producers empty.
    if 'studios' not in df.columns:
        df['studios'] = df.get('authors', '')
    if 'producers' not in df.columns:
        df['producers'] = ''
    if 'creators' not in df.columns:
        df['creators'] = df.get('authors', '')  # For manga, authors are the creators
    
    # Create historical features (running state logic is domain-agnostic)
    df = create_historical_features(df)
    
    # Fill NAs in historical features
    df['hist_user_mean'] = df['hist_user_mean'].fillna(df['user_score'].mean() if len(df) > 0 else 5.0)
    df['recent_mean_score'] = df['recent_mean_score'].fillna(df['hist_user_mean'])
    df['hist_franchise_count'] = df['hist_franchise_count'].fillna(0)
    df['hist_char_count'] = df['hist_char_count'].fillna(0)
    for col in ['hist_format_affinity', 'hist_genre_affinity', 'hist_tag_affinity', 'hist_studio_affinity', 'hist_producer_affinity', 'hist_creator_affinity', 'hist_franchise_mean', 'hist_char_mean']:
        df[col] = df[col].fillna(df['hist_user_mean'])
        
    # Manga Numeric Features
    df['chapters'] = df['chapters'].fillna(0)
    df['volumes'] = df.get('volumes', pd.Series([0]*len(df))).fillna(0)
    
    # Manhwas do not have volumes and shouldn't be penalized.
    # We impute volumes = chapters // 9 (a standard average for JP manga)
    # for any manga that has 0 volumes but > 0 chapters.
    mask_needs_volumes = (df['volumes'] == 0) & (df['chapters'] > 0)
    df.loc[mask_needs_volumes, 'volumes'] = np.ceil(df.loc[mask_needs_volumes, 'chapters'] / 9.0)
    df['releaseYear'] = df['releaseYear'].fillna(df['releaseYear'].median() if not df['releaseYear'].isna().all() else 2015.0)
    
    df['averageScore'] = df['averageScore'].replace(0, np.nan)
    avg_score_median = df['averageScore'].median() if not df['averageScore'].isna().all() else 70.0
    df['averageScore'] = df['averageScore'].fillna(avg_score_median) / 10.0
    
    pop_median = df['popularity'].median() if not df['popularity'].isna().all() else 1000.0
    if 'media_status' in df.columns:
        unreleased_mask = df['media_status'] == 'NOT_YET_RELEASED'
        df.loc[unreleased_mask, 'popularity'] = np.nan
    df['popularity'] = np.log1p(df['popularity'].fillna(pop_median))
    
    fav_median = df['favourites'].median() if not df['favourites'].isna().all() else 10.0
    if 'media_status' in df.columns:
        unreleased_mask = df['media_status'] == 'NOT_YET_RELEASED'
        df.loc[unreleased_mask, 'favourites'] = np.nan
    df['favourites'] = np.log1p(df['favourites'].fillna(fav_median))
    
    df['isAdult'] = df['isAdult'].fillna(0).astype(int)
    
    # New features: completion ratio, rewatch count
    df['progress'] = df['progress'].fillna(0)
    df['completion_ratio'] = (df['progress'] / df['chapters'].replace(0, np.nan)).fillna(1.0).clip(0, 1)
    df['repeat'] = df['repeat'].fillna(0).astype(int)
    
    # Categorical (One-Hot / Multi-Hot)
    categorical_cols = ['format', 'countryOfOrigin']
    df_cat = pd.get_dummies(df[categorical_cols], dummy_na=True, drop_first=False)
    
    # Genres Multi-Hot
    all_genres = set()
    for row_genres in df['genres']:
        all_genres.update(_get_list(row_genres))
    for g in all_genres:
        if not g: continue
        df_cat[f'genre_{g}'] = df['genres'].apply(lambda x: 1.0 if g in _get_list(x) else 0.0)
        
    # Tags Multi-Hot with Ranks
    all_tags = set()
    for row_tags in df['tags']:
        all_tags.update(_get_tags_with_ranks(row_tags).keys())
    for t in all_tags:
        if not t: continue
        df_cat[f'tag_{t}'] = df['tags'].apply(lambda x: _get_tags_with_ranks(x).get(t, 0.0) / 100.0)

    # Combine
    num_cols = ['chapters', 'volumes', 'releaseYear', 'averageScore', 'popularity', 'favourites', 'isAdult', 'completion_ratio', 'repeat']
    hist_cols = ['hist_user_mean', 'hist_user_std', 'recent_mean_score', 'hist_count', 'hist_format_affinity', 'hist_genre_affinity', 'hist_genre_freq', 
                 'hist_tag_affinity', 'hist_tag_freq', 'hist_studio_affinity', 'hist_studio_freq', 'hist_producer_affinity', 'hist_producer_freq', 'hist_creator_affinity', 'hist_creator_freq', 'hist_franchise_count', 'hist_franchise_mean', 'hist_char_count', 'hist_char_mean', 'hist_global_diff', 'hist_global_mae']
                 
    X = pd.concat([df[num_cols + hist_cols], df_cat], axis=1)
    y = df['user_score']
    
    return X, y

def build_manga_inference_features(manga_data_dict, user_history_df, train_columns, inference_context: Optional[Dict[str, Any]] = None):
    """
    Given a new manga dict from API and user manga history df, construct the 1-row feature matrix.
    """
    context = inference_context or prepare_manga_inference_context(user_history_df, train_columns)
    row = build_manga_inference_feature_row(manga_data_dict, context)
    return pd.DataFrame([row], columns=context['train_columns']).fillna(0)

