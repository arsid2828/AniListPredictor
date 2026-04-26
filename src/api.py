import requests
import json
import os
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from app.shared import CACHE_RETENTION_HOURS

CACHE_TTL_HOURS = CACHE_RETENTION_HOURS
REQUEST_TIMEOUT_SECONDS = 30

def _is_cache_valid(cache_path, ttl_hours=CACHE_TTL_HOURS):
    """Check if a cache file exists and is younger than ttl_hours."""
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < ttl_hours * 3600

API_URL = "https://graphql.anilist.co"

# We request in point 10 decimal format
USER_LIST_QUERY = """
query ($userName: String, $chunk: Int) {
  MediaListCollection(userName: $userName, type: ANIME, chunk: $chunk) {
    hasNextChunk
    lists {
      name
      status
      entries {
        mediaId
        status
        score(format: POINT_10_DECIMAL)
        progress
        repeat
        startedAt { year month day }
        completedAt { year month day }
        updatedAt
        media {
          id
          title { romaji english }
          coverImage {
            large
          }
          format
          episodes
          duration
          season
          seasonYear
          averageScore
          meanScore
          favourites
          isAdult
          popularity
          countryOfOrigin
          source
          status
          genres
          tags {
            name
            rank
            category
          }
          relations {
            edges {
              relationType
              node {
                id
              }
            }
          }
          studios {
            edges {
              isMain
              node { name }
            }
          }
          staff {
            edges {
              role
              node { name { full } }
            }
          }
        }
      }
    }
  }
}
"""

ANIME_SEARCH_QUERY = """
query ($search: String) {
  Page(page: 1, perPage: 10) {
    media(search: $search, type: ANIME, sort: SEARCH_MATCH) {
      id
      title { romaji english }
      format
      episodes
      duration
      season
      seasonYear
      averageScore
      meanScore
      favourites
      isAdult
      popularity
      countryOfOrigin
      source
      status
      genres
      tags {
        name
        rank
        category
      }
      coverImage {
        extraLarge
        large
      }
      relations {
        edges {
          relationType
          node {
            id
          }
        }
      }
      studios {
        edges {
          isMain
          node { name }
        }
      }
      staff {
        edges {
          role
          node { name { full } }
        }
      }
    }
  }
}
"""

ANIME_CANDIDATES_QUERY = """
query ($page: Int, $perPage: Int, $sort: [MediaSort], $genre: String, $tag: String, $seasonYear: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, sort: $sort, genre: $genre, tag: $tag, seasonYear: $seasonYear) {
      id
      title { romaji english }
      format
      episodes
      duration
      season
      seasonYear
      averageScore
      meanScore
      favourites
      isAdult
      popularity
      countryOfOrigin
      source
      status
      genres
      tags {
        name
        rank
        category
      }
      coverImage {
        extraLarge
        large
      }
      relations {
        edges {
          relationType
          node {
            id
          }
        }
      }
      studios {
        edges {
          isMain
          node { name }
        }
      }
      staff {
        edges {
          role
          node { name { full } }
        }
      }
    }
  }
}
"""

USER_MANGA_LIST_QUERY = """
query ($userName: String, $chunk: Int) {
  MediaListCollection(userName: $userName, type: MANGA, chunk: $chunk) {
    hasNextChunk
    lists {
      name
      status
      entries {
        mediaId
        status
        score(format: POINT_10_DECIMAL)
        progress
        repeat
        startedAt { year month day }
        completedAt { year month day }
        updatedAt
        media {
          id
          title { romaji english }
          coverImage {
            large
          }
          format
          chapters
          volumes
          startDate { year }
          averageScore
          meanScore
          favourites
          isAdult
          popularity
          countryOfOrigin
          status
          genres
          tags {
            name
            rank
            category
          }
          relations {
            edges {
              relationType
              node {
                id
              }
            }
          }
          staff {
            edges {
              role
              node { name { full } }
            }
          }
        }
      }
    }
  }
}
"""

MANGA_SEARCH_QUERY = """
query ($search: String) {
  Page(page: 1, perPage: 10) {
    media(search: $search, type: MANGA, sort: SEARCH_MATCH) {
      id
      title { romaji english }
      format
      chapters
      volumes
      startDate { year }
      averageScore
      meanScore
      favourites
      isAdult
      popularity
      countryOfOrigin
      status
      genres
      tags {
        name
        rank
        category
      }
      coverImage {
        extraLarge
        large
      }
      relations {
        edges {
          relationType
          node {
            id
          }
        }
      }
      staff {
        edges {
          role
          node { name { full } }
        }
      }
    }
  }
}
"""

MANGA_CANDIDATES_QUERY = """
query ($page: Int, $perPage: Int, $sort: [MediaSort], $genre: String, $tag: String) {
  Page(page: $page, perPage: $perPage) {
    media(type: MANGA, sort: $sort, genre: $genre, tag: $tag) {
      id
      title { romaji english }
      format
      chapters
      volumes
      startDate { year }
      averageScore
      meanScore
      favourites
      isAdult
      popularity
      countryOfOrigin
      status
      genres
      tags {
        name
        rank
        category
      }
      coverImage {
        extraLarge
        large
      }
      relations {
        edges {
          relationType
          node {
            id
          }
        }
      }
      staff {
        edges {
          role
          node { name { full } }
        }
      }
    }
  }
}
"""

def fetch_with_retry(query, variables, retries=3):
    """Fetch from AniList GraphQL with basic exponential backoff retry logic."""
    for attempt in range(retries):
        try:
            response = requests.post(
                API_URL, 
                json={"query": query, "variables": variables},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            
            # Rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning("AniList rate limit hit on attempt %s/%s. Sleeping for %s seconds.", attempt + 1, retries, retry_after)
                time.sleep(retry_after)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                error_message = data["errors"][0].get("message", "Unknown")
                logger.warning("AniList GraphQL returned an error: %s", error_message)
                raise ValueError(f"GraphQL Error: {error_message}")
                
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            logger.warning("AniList request failed on attempt %s/%s: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                sleep_time = 2 ** attempt
                logger.debug("Retrying AniList request in %s seconds.", sleep_time)
                time.sleep(sleep_time)
            else:
                raise e
    raise Exception("Max retries exceeded.")

def fetch_user_anime_list(username: str, force_refresh: bool = False):
    """
    Fetch all anime list entries for a user, using chunks to handle large lists.
    Caches the full unified output to disk.
    """
    cache_path = CACHE_DIR / f"user_list_{username.lower()}.json"
    
    if not force_refresh and _is_cache_valid(cache_path):
        logger.debug("Loading cached anime list for %s.", username)
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    logger.debug("Fetching anime list for %s.", username)
    all_lists = {}
    chunk = 1
    has_next_chunk = True
    
    while has_next_chunk:
        variables = {
            "userName": username,
            "chunk": chunk
        }
        
        logger.debug("Requesting anime list chunk %s.", chunk)
        try:
            data = fetch_with_retry(USER_LIST_QUERY, variables)
        except ValueError as e:
            if "User not found" in str(e) or "private" in str(e).lower():
                logger.warning("Cannot fetch anime list because the profile is unavailable or private.")
                return None
            raise
            
        collection = data.get("MediaListCollection", {})
        if not collection:
            break
            
        lists = collection.get("lists", [])
        for lst in lists:
            name = lst.get("name")
            if name not in all_lists:
                all_lists[name] = {"name": name, "status": lst.get("status"), "entries": []}
            all_lists[name]["entries"].extend(lst.get("entries", []))
            
        has_next_chunk = collection.get("hasNextChunk", False)
        chunk += 1
        
        # Polite sleep to avoid hitting limits during pagination
        time.sleep(1)
        
    unified_data = list(all_lists.values())
    
    # Save to cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
    return unified_data

def search_anime_by_title(title: str):
    """Search for an anime by title, returns top 10 matches."""
    variables = {"search": title}
    logger.debug("Searching anime title.")
    data = fetch_with_retry(ANIME_SEARCH_QUERY, variables)
    return data.get("Page", {}).get("media", [])

def get_candidate_anime_for_recommendations(limit: int = 500, genre: str = None, tag: str = None, year: int = None):
    """
    Fetch top anime by popularity and top anime by score to serve as candidates.
    Returns a unique list of media dictionaries.
    """
    limit_per_category = limit // 2
    per_page = 50
    pages_per_category = max(1, limit_per_category // per_page)
    
    unique_candidates = {}
    
    logger.debug("Fetching anime candidates with limit=%s.", limit)
    
    base_vars = {}
    if genre: base_vars["genre"] = genre
    if tag: base_vars["tag"] = tag
    if year: base_vars["seasonYear"] = int(year)

    for p in range(1, pages_per_category + 1):
        vars_pop = {"page": p, "perPage": per_page, "sort": ["POPULARITY_DESC"]}
        vars_pop.update(base_vars)
        data = fetch_with_retry(ANIME_CANDIDATES_QUERY, vars_pop)
        media_list = data.get("Page", {}).get("media", [])
        for m in media_list:
            if m: unique_candidates[m['id']] = m
        time.sleep(1) # Polite delay
        
    for p in range(1, pages_per_category + 1):
        vars_score = {"page": p, "perPage": per_page, "sort": ["SCORE_DESC"]}
        vars_score.update(base_vars)
        data = fetch_with_retry(ANIME_CANDIDATES_QUERY, vars_score)
        media_list = data.get("Page", {}).get("media", [])
        for m in media_list:
            if m: unique_candidates[m['id']] = m
        time.sleep(1) # Polite delay
        
    return list(unique_candidates.values())

def fetch_user_manga_list(username: str, force_refresh: bool = False):
    cache_path = CACHE_DIR / f"user_manga_list_{username.lower()}.json"
    if not force_refresh and _is_cache_valid(cache_path):
        logger.debug("Loading cached manga list for %s.", username)
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    logger.debug("Fetching manga list for %s.", username)
    all_lists = {}
    chunk = 1
    has_next_chunk = True
    
    while has_next_chunk:
        variables = {"userName": username, "chunk": chunk}
        logger.debug("Requesting manga list chunk %s.", chunk)
        try:
            data = fetch_with_retry(USER_MANGA_LIST_QUERY, variables)
        except ValueError as e:
            if "User not found" in str(e) or "private" in str(e).lower():
                logger.warning("Cannot fetch manga list because the profile is unavailable or private.")
                return None
            raise
            
        collection = data.get("MediaListCollection", {})
        if not collection: break
            
        lists = collection.get("lists", [])
        for lst in lists:
            name = lst.get("name")
            if name not in all_lists:
                all_lists[name] = {"name": name, "status": lst.get("status"), "entries": []}
            all_lists[name]["entries"].extend(lst.get("entries", []))
            
        has_next_chunk = collection.get("hasNextChunk", False)
        chunk += 1
        time.sleep(1)
        
    unified_data = list(all_lists.values())
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
    return unified_data

def search_manga_by_title(title: str):
    variables = {"search": title}
    logger.debug("Searching manga title.")
    data = fetch_with_retry(MANGA_SEARCH_QUERY, variables)
    return data.get("Page", {}).get("media", [])

def get_candidate_manga_for_recommendations(limit: int = 500, genre: str = None, tag: str = None):
    limit_per_category = limit // 2
    per_page = 50
    pages_per_category = max(1, limit_per_category // per_page)
    
    unique_candidates = {}
    
    logger.debug("Fetching manga candidates with limit=%s.", limit)
    
    base_vars = {}
    if genre: base_vars["genre"] = genre
    if tag: base_vars["tag"] = tag

    for p in range(1, pages_per_category + 1):
        vars_pop = {"page": p, "perPage": per_page, "sort": ["POPULARITY_DESC"]}
        vars_pop.update(base_vars)
        data = fetch_with_retry(MANGA_CANDIDATES_QUERY, vars_pop)
        for m in data.get("Page", {}).get("media", []):
            if m: unique_candidates[m['id']] = m
        time.sleep(1)
        
    for p in range(1, pages_per_category + 1):
        vars_score = {"page": p, "perPage": per_page, "sort": ["SCORE_DESC"]}
        vars_score.update(base_vars)
        data = fetch_with_retry(MANGA_CANDIDATES_QUERY, vars_score)
        for m in data.get("Page", {}).get("media", []):
            if m: unique_candidates[m['id']] = m
        time.sleep(1)
        
    return list(unique_candidates.values())

USER_ACTIVITY_QUERY = """
query ($userId: Int, $page: Int, $type: ActivityType) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    activities(userId: $userId, type: $type, sort: ID_DESC) {
      ... on ListActivity {
        id
        createdAt
        progress
        status
        media {
          title { romaji english }
          format
          episodes
          chapters
          duration
          countryOfOrigin
        }
      }
    }
  }
}
"""

USER_ID_QUERY = """
query ($name: String) {
  User(name: $name) { id }
}
"""

def _load_cached_json_list(cache_path: Path):
    if not cache_path.exists():
        return []
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        logger.warning("Failed to read cached JSON list from %s.", cache_path.name, exc_info=True)
        return []


def _merge_activities_preserving_latest(cached_activities, new_activities):
    merged_by_id = {}
    for act in new_activities + cached_activities:
        act_id = act.get("id")
        if act_id is None:
            continue
        if act_id not in merged_by_id:
            merged_by_id[act_id] = act
    merged = list(merged_by_id.values())
    merged.sort(key=lambda item: item.get("id", 0), reverse=True)
    return merged


def fetch_user_activity_history(username: str, media_type: str = "ANIME", force_refresh: bool = False):
    """Fetch user's activity history (anime or manga) from AniList."""
    cache_path = CACHE_DIR / f"user_activity_{media_type.lower()}_{username.lower()}.json"
    cached_activities = _load_cached_json_list(cache_path)
    
    if not force_refresh and _is_cache_valid(cache_path):
        logger.debug("Loading cached activity history for %s (%s).", username, media_type)
        return cached_activities
            
    logger.debug("Fetching AniList user ID.")
    try:
        data = fetch_with_retry(USER_ID_QUERY, {"name": username})
        user_id = data.get("User", {}).get("id")
        if not user_id:
            return None
    except Exception as e:
        logger.error(f"Failed to fetch user ID: {e}")
        return None
        
    activities = []
    cached_ids = {act.get("id") for act in cached_activities if act.get("id") is not None}
    page = 1
    has_next = True
    reached_cached_boundary = False
    
    logger.debug("Fetching activity history for %s.", username)
    
    while has_next and page <= 200:  # Max 10,000 activities limitation to avoid infinite loop
        variables = {"userId": user_id, "page": page, "type": f"{media_type.upper()}_LIST"}
        try:
            data = fetch_with_retry(USER_ACTIVITY_QUERY, variables)
        except Exception as e:
            logger.error(f"Failed to fetch activity page {page}: {e}")
            break
            
        page_info = data.get("Page", {}).get("pageInfo", {})
        acts = data.get("Page", {}).get("activities", [])

        if not acts:
            break

        for act in acts:
            act_id = act.get("id")
            if act_id in cached_ids:
                reached_cached_boundary = True
                break
            activities.append(act)

        has_next = page_info.get("hasNextPage", False)
        if reached_cached_boundary:
            break
        page += 1
        time.sleep(0.5)

    merged_activities = _merge_activities_preserving_latest(cached_activities, activities)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(merged_activities, f, ensure_ascii=False, indent=2)
        
    return merged_activities

if __name__ == "__main__":
    test_username = os.environ.get("TEST_USERNAME", "example_user")
    data = fetch_user_anime_list(test_username)
    print(f"Fetched list chunks: {len(data) if data else 0}")
