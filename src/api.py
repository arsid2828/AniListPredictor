import requests
import json
import os
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("c:/Users/arsid/Desktop/AnilistProject/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_HOURS = 24

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
query ($page: Int, $perPage: Int, $sort: [MediaSort]) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, sort: $sort) {
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
          format
          chapters
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
query ($page: Int, $perPage: Int, $sort: [MediaSort]) {
  Page(page: $page, perPage: $perPage) {
    media(type: MANGA, sort: $sort) {
      id
      title { romaji english }
      format
      chapters
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
                headers={"Accept": "application/json", "Content-Type": "application/json"}
            )
            
            # Rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                logger.warning(f"Rate limited by AniList. Attempt {attempt+1}/{retries}. Sleeping for {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                logger.error(f"GraphQL Errors: {data['errors']}")
                raise ValueError(f"GraphQL Error: {data['errors'][0].get('message', 'Unknown')}")
                
            return data["data"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if attempt < retries - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
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
        logger.info(f"Loading cached anime list for {username}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    logger.info(f"Fetching anime list for user: {username}")
    all_lists = {}
    chunk = 1
    has_next_chunk = True
    
    while has_next_chunk:
        variables = {
            "userName": username,
            "chunk": chunk
        }
        
        logger.info(f"Requesting chunk {chunk}...")
        try:
            data = fetch_with_retry(USER_LIST_QUERY, variables)
        except ValueError as e:
            if "User not found" in str(e) or "private" in str(e).lower():
                logger.error(f"Cannot fetch data for {username}: Profile might be private or non-existent.")
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
    logger.info(f"Searching for anime: {title}")
    data = fetch_with_retry(ANIME_SEARCH_QUERY, variables)
    return data.get("Page", {}).get("media", [])

def get_candidate_anime_for_recommendations(limit: int = 500):
    """
    Fetch top anime by popularity and top anime by score to serve as candidates.
    Returns a unique list of media dictionaries.
    """
    limit_per_category = limit // 2
    per_page = 50
    pages_per_category = max(1, limit_per_category // per_page)
    
    unique_candidates = {}
    
    logger.info("Fetching most popular anime candidates...")
    for p in range(1, pages_per_category + 1):
        vars_pop = {"page": p, "perPage": per_page, "sort": ["POPULARITY_DESC"]}
        data = fetch_with_retry(ANIME_CANDIDATES_QUERY, vars_pop)
        media_list = data.get("Page", {}).get("media", [])
        for m in media_list:
            if m: unique_candidates[m['id']] = m
        time.sleep(1) # Polite delay
        
    logger.info("Fetching highest rated anime candidates...")
    for p in range(1, pages_per_category + 1):
        vars_score = {"page": p, "perPage": per_page, "sort": ["SCORE_DESC"]}
        data = fetch_with_retry(ANIME_CANDIDATES_QUERY, vars_score)
        media_list = data.get("Page", {}).get("media", [])
        for m in media_list:
            if m: unique_candidates[m['id']] = m
        time.sleep(1) # Polite delay
        
    return list(unique_candidates.values())

def fetch_user_manga_list(username: str, force_refresh: bool = False):
    cache_path = CACHE_DIR / f"user_manga_list_{username.lower()}.json"
    if not force_refresh and _is_cache_valid(cache_path):
        logger.info(f"Loading cached manga list for {username}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    logger.info(f"Fetching manga list for user: {username}")
    all_lists = {}
    chunk = 1
    has_next_chunk = True
    
    while has_next_chunk:
        variables = {"userName": username, "chunk": chunk}
        logger.info(f"Requesting manga chunk {chunk}...")
        try:
            data = fetch_with_retry(USER_MANGA_LIST_QUERY, variables)
        except ValueError as e:
            if "User not found" in str(e) or "private" in str(e).lower():
                logger.error(f"Cannot fetch manga data for {username}: Profile might be private or non-existent.")
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
    logger.info(f"Searching for manga: {title}")
    data = fetch_with_retry(MANGA_SEARCH_QUERY, variables)
    return data.get("Page", {}).get("media", [])

def get_candidate_manga_for_recommendations(limit: int = 500):
    limit_per_category = limit // 2
    per_page = 50
    pages_per_category = max(1, limit_per_category // per_page)
    
    unique_candidates = {}
    
    logger.info("Fetching most popular manga candidates...")
    for p in range(1, pages_per_category + 1):
        vars_pop = {"page": p, "perPage": per_page, "sort": ["POPULARITY_DESC"]}
        data = fetch_with_retry(MANGA_CANDIDATES_QUERY, vars_pop)
        for m in data.get("Page", {}).get("media", []):
            if m: unique_candidates[m['id']] = m
        time.sleep(1)
        
    logger.info("Fetching highest rated manga candidates...")
    for p in range(1, pages_per_category + 1):
        vars_score = {"page": p, "perPage": per_page, "sort": ["SCORE_DESC"]}
        data = fetch_with_retry(MANGA_CANDIDATES_QUERY, vars_score)
        for m in data.get("Page", {}).get("media", []):
            if m: unique_candidates[m['id']] = m
        time.sleep(1)
        
    return list(unique_candidates.values())

if __name__ == "__main__":
    # Test script locally
    username = "arsid"  # Test with an example username or yours
    # data = fetch_user_anime_list(username)
    # print(f"Fetched list chunks: {len(data) if data else 0}")
