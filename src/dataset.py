import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import logging
from .api import fetch_user_anime_list

logger = logging.getLogger(__name__)

DATA_DIR = Path("c:/Users/arsid/Desktop/AnilistProject/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def parse_date(date_dict):
    """Safely parse AniList fuzzyDate dict to datetime info."""
    if not date_dict:
        return None
    y, m, d = date_dict.get('year'), date_dict.get('month'), date_dict.get('day')
    if y and m and d:
        try:
            return pd.Timestamp(year=y, month=m, day=d)
        except ValueError:
            return None
    return None

def build_user_dataframe(username: str, force_refresh: bool = False, save_csv: bool = True) -> pd.DataFrame:
    """
    Fetch user list from cache/API, flatten it, and return a clean DataFrame.
    Filters out uncategorized or completely missing score items.
    """
    raw_data = fetch_user_anime_list(username, force_refresh=force_refresh)
    if not raw_data:
        logger.error(f"No data retrieved for {username}. Cannot build dataset.")
        return pd.DataFrame()

    rows = []
    
    for str_list in raw_data: # these are the top level list groups
        list_name = str_list.get('name')
        entries = str_list.get('entries', [])
        
        for entry in entries:
            media = entry.get('media', {})
            if not media:
                continue
                
            # Flatten studios to a comma-separated string
            studios_data = media.get('studios', {}).get('edges', [])
            # In feature engineering we will handle strings, here we just extract main ones
            main_studios = [s['node']['name'] for s in studios_data if s.get('isMain')]
            producers = [s['node']['name'] for s in studios_data if not s.get('isMain')]
            all_studios = [s['node']['name'] for s in studios_data]
            studio_str = ", ".join(main_studios) if main_studios else ", ".join(all_studios)
            producer_str = ", ".join(producers)
            
            # Flatten tags with their rank (percentage of affinity)
            tags_str_list = []
            for t in media.get('tags', []):
                t_name = str(t['name']).replace('=', '-').replace(',', '') # sanitize
                t_rank = t.get('rank', 0)
                tags_str_list.append(f"{t_name}={t_rank}")
            
            relations_data = media.get('relations', {}).get('edges', [])
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
            
            # Get best title
            title_dict = media.get('title', {})
            title = title_dict.get('english') or title_dict.get('romaji') or str(media.get('id'))
            
            row = {
                'username': username,
                'mediaId': entry.get('mediaId'),
                'title': title,
                'user_score': entry.get('score'),  # Already 0-10 format, 0 means unrated
                'user_status': entry.get('status'),
                'progress': entry.get('progress', 0),
                'repeat': entry.get('repeat', 0),
                'startedAt': parse_date(entry.get('startedAt')),
                'completedAt': parse_date(entry.get('completedAt')),
                'updatedAt': pd.to_datetime(entry.get('updatedAt'), unit='s') if entry.get('updatedAt') else None,
                
                # Media Info
                'format': media.get('format'),
                'episodes': media.get('episodes'),
                'duration': media.get('duration'),
                'season': media.get('season'),
                'seasonYear': media.get('seasonYear'),
                'averageScore': media.get('averageScore'),
                'meanScore': media.get('meanScore'),
                'favourites': media.get('favourites'),
                'isAdult': 1 if media.get('isAdult') else 0,
                'popularity': media.get('popularity'),
                'countryOfOrigin': media.get('countryOfOrigin'),
                'source': media.get('source'),
                'media_status': media.get('status'),
                'genres': ", ".join(media.get('genres', [])),
                'tags': ", ".join(tags_str_list),
                'related_ids': ",".join(related_ids),
                'char_related_ids': ",".join(char_related_ids),
                'studios': studio_str,
                'producers': producer_str
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    if df.empty:
        return df
        
    # Deduplica: se un utente ha un anime in "Completed" e anche in una "Custom List", GraphQL lo restituisce 2 volte.
    df = df.drop_duplicates(subset=['mediaId']).copy()
        
    # --- Clean up & Filtering ---
    # We are predicting user score. Filter out entries with score == 0 (unrated).
    # AniList POINT_10_DECIMAL format returns 0 for unrated entries.
    original_len = len(df)
    df = df[df['user_score'] > 0].copy()
    logger.info(f"Dropped {original_len - len(df)} unrated anime. Remaining: {len(df)}")
    
    if len(df) < 10:
        logger.warning(f"User {username} has less than 10 rated anime. ML models will be highly unreliable.")
        
    # Chronological sort
    # We will sort by completedAt first. If missing, fallback to updatedAt
    df['sort_date'] = df['completedAt'].fillna(df['updatedAt'])
    df = df.sort_values('sort_date').reset_index(drop=True)
    
    if save_csv:
        csv_path = DATA_DIR / f"{username}_clean.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")
        
    return df
