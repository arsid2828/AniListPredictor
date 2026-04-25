import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
BLOCKED_GENRES = {"Ecchi", "Hentai"}
BLOCKED_TAG_TERMS = {"ecchi", "hentai", "adult", "nsfw", "nudity", "sexual", "erotica"}

def _get_list(val):
    if pd.isna(val) or not val:
        return []
    values = []
    for item in str(val).split(","):
        item = item.strip()
        if not item or item in BLOCKED_GENRES:
            continue
        values.append(item)
    return values

def _get_tags_with_ranks(val):
    if pd.isna(val) or not val:
        return {}
    res = {}
    for x in str(val).split(","):
        x = x.strip()
        if not x: continue
        parts = x.rsplit('=', 1)
        tag_name = parts[0].strip().lower()
        if any(term in tag_name for term in BLOCKED_TAG_TERMS):
            continue
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
    num_cols = ['episodes', 'duration', 'seasonYear', 'averageScore', 'popularity', 'favourites', 'completion_ratio', 'repeat']
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

def build_inference_features(anime_data_dict, user_history_df, train_columns):
    """
    Given a new anime dict from API and user history df, construct the 1-row feature matrix
    Expected to return exactly the same columns as the model expects.
    This reconstructs the state just for this new anime using the ENTIRE user_history_df.
    """
    title_dict = anime_data_dict.get('title', {})
    
    studios_data = anime_data_dict.get('studios', {}).get('edges', [])
    main_studios = [s['node']['name'] for s in studios_data if s.get('isMain')]
    producers = [s['node']['name'] for s in studios_data if not s.get('isMain')]
    all_studios = [s['node']['name'] for s in studios_data]
    studio_str = ", ".join(main_studios) if main_studios else ", ".join(all_studios)
    producer_str = ", ".join(producers)
    
    # Extract key creators from staff
    staff_data = anime_data_dict.get('staff', {}).get('edges', [])
    creators = []
    for s in staff_data:
        role = (s.get('role') or '').lower()
        name = s.get('node', {}).get('name', {}).get('full', '')
        if name and ('director' in role or 'original creator' in role or 'original story' in role or 'series composition' in role or 'original character design' in role):
            creators.append(name)
    creator_str = ", ".join(creators) if creators else ""
    
    tags_str_list = []
    for t in anime_data_dict.get('tags', []):
        t_name = str(t['name']).replace('=', '-').replace(',', '')
        t_rank = t.get('rank', 0)
        tags_str_list.append(f"{t_name}={t_rank}")
    
    relations_data = anime_data_dict.get('relations', {}).get('edges', [])
    related_ids = []
    char_related_ids = []
    valid_types = ['ADAPTATION', 'PREQUEL', 'SEQUEL', 'PARENT', 'SIDE_STORY', 'SUMMARY', 'ALTERNATIVE', 'SPIN_OFF', 'OTHER', 'SOURCE', 'COMPILATION', 'CONTAINS']
    for edge in relations_data:
        node_id = edge.get('node', {}).get('id')
        if not node_id: continue
        r_type = edge.get('relationType')
        if r_type == 'CHARACTER':
            char_related_ids.append(str(node_id))
        elif r_type in valid_types or r_type:
            related_ids.append(str(node_id))
            
    row = {
        'user_score': np.nan, # Unknown target
        'mediaId': anime_data_dict.get('id'),
        'media_status': anime_data_dict.get('status'),
        'related_ids': ",".join(related_ids),
        'char_related_ids': ",".join(char_related_ids),
        'format': anime_data_dict.get('format'),
        'episodes': anime_data_dict.get('episodes'),
        'duration': anime_data_dict.get('duration'),
        'season': anime_data_dict.get('season'),
        'seasonYear': anime_data_dict.get('seasonYear'),
        'averageScore': anime_data_dict.get('averageScore'),
        'countryOfOrigin': anime_data_dict.get('countryOfOrigin'),
        'favourites': anime_data_dict.get('favourites'),
        'popularity': anime_data_dict.get('popularity'),
        'genres': ", ".join(anime_data_dict.get('genres', [])),
        'tags': ", ".join(tags_str_list),
        'studios': studio_str,
        'producers': producer_str,
        'progress': anime_data_dict.get('episodes') or 0,
        'repeat': 0,
        'source': anime_data_dict.get('source', 'UNKNOWN'),
        'creators': creator_str,
        'sort_date': pd.Timestamp('2100-01-01') # Put firmly at the absolute end of history
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
    num_cols = ['chapters', 'volumes', 'releaseYear', 'averageScore', 'popularity', 'favourites', 'completion_ratio', 'repeat']
    hist_cols = ['hist_user_mean', 'hist_user_std', 'recent_mean_score', 'hist_count', 'hist_format_affinity', 'hist_genre_affinity', 'hist_genre_freq', 
                 'hist_tag_affinity', 'hist_tag_freq', 'hist_studio_affinity', 'hist_studio_freq', 'hist_producer_affinity', 'hist_producer_freq', 'hist_creator_affinity', 'hist_creator_freq', 'hist_franchise_count', 'hist_franchise_mean', 'hist_char_count', 'hist_char_mean', 'hist_global_diff', 'hist_global_mae']
                 
    X = pd.concat([df[num_cols + hist_cols], df_cat], axis=1)
    y = df['user_score']
    
    return X, y

def build_manga_inference_features(manga_data_dict, user_history_df, train_columns):
    """
    Given a new manga dict from API and user manga history df, construct the 1-row feature matrix.
    """
    # Staff (author / artist)
    staff_data = manga_data_dict.get('staff', {}).get('edges', [])
    authors = []
    for s in staff_data:
        role = (s.get('role') or '').lower()
        name = s.get('node', {}).get('name', {}).get('full', '')
        if name and ('story' in role or 'art' in role or 'original' in role):
            authors.append(name)
    author_str = ", ".join(authors) if authors else ""
    
    tags_str_list = []
    for t in manga_data_dict.get('tags', []):
        t_name = str(t['name']).replace('=', '-').replace(',', '')
        t_rank = t.get('rank', 0)
        tags_str_list.append(f"{t_name}={t_rank}")
    
    relations_data = manga_data_dict.get('relations', {}).get('edges', [])
    related_ids = []
    char_related_ids = []
    valid_types = ['ADAPTATION', 'PREQUEL', 'SEQUEL', 'PARENT', 'SIDE_STORY', 'SUMMARY', 'ALTERNATIVE', 'SPIN_OFF', 'OTHER', 'SOURCE', 'COMPILATION', 'CONTAINS']
    for edge in relations_data:
        node_id = edge.get('node', {}).get('id')
        if not node_id: continue
        r_type = edge.get('relationType')
        if r_type == 'CHARACTER':
            char_related_ids.append(str(node_id))
        elif r_type in valid_types or r_type:
            related_ids.append(str(node_id))
    
    start_date = manga_data_dict.get('startDate', {}) or {}
    release_year = start_date.get('year')
            
    row = {
        'user_score': np.nan,
        'mediaId': manga_data_dict.get('id'),
        'media_status': manga_data_dict.get('status'),
        'related_ids': ",".join(related_ids),
        'char_related_ids': ",".join(char_related_ids),
        'format': manga_data_dict.get('format'),
        'chapters': manga_data_dict.get('chapters'),
        'volumes': manga_data_dict.get('volumes'),
        'releaseYear': release_year,
        'averageScore': manga_data_dict.get('averageScore'),
        'countryOfOrigin': manga_data_dict.get('countryOfOrigin'),
        'favourites': manga_data_dict.get('favourites'),
        'popularity': manga_data_dict.get('popularity'),
        'genres': ", ".join(manga_data_dict.get('genres', [])),
        'tags': ", ".join(tags_str_list),
        'authors': author_str,
        'progress': manga_data_dict.get('chapters') or 0,
        'repeat': 0,
        'sort_date': pd.Timestamp('2100-01-01')
    }
    
    new_df = pd.DataFrame([row])
    combined = pd.concat([user_history_df, new_df], ignore_index=True)
    
    X, _ = engineer_manga_features(combined)
    
    X_last = X.iloc[[-1]].copy()
    
    for col in train_columns:
        if col not in X_last.columns:
            X_last[col] = 0
            
    X_inference = X_last[train_columns]
    X_inference = X_inference.fillna(0)
    
    return X_inference

