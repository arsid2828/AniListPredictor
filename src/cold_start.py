import numpy as np

# Static Genre Definitions
GENRES_LIST = [
    "Action", "Adventure", "Comedy", "Drama", "Ecchi", 
    "Fantasy", "Hentai", "Horror", "Mahou Shoujo", "Mecha", "Music", 
    "Mystery", "Psychological", "Romance", "Sci-Fi", 
    "Slice of Life", "Sports", "Supernatural", "Thriller"
]

FORMATS_LIST = ["TV", "MOVIE", "OVA", "ONA", "SPECIAL", "ANY"]
LENGTHS_LIST = ["SHORT (<14)", "MEDIUM (14-26)", "LONG (27+)", "ANY"]
ERAS_LIST = ["CLASSICS (Pre-2000)", "MODERN (2000-2015)", "RECENT (Post-2015)", "ANY"]

def build_cold_start_profile(favorite_anime_list, preferred_genres, avoided_genres, preferred_format, preferred_length, preferred_era):
    """
    Builds a mock user profile ("Cold Start Profile") based on explicitly stated preferences
    and implicitly derived traits from the 3-5 favorite anime.
    """
    profile = {
        "explicit": {
            "preferred_genres": preferred_genres or [],
            "avoided_genres": avoided_genres or [],
            "preferred_format": preferred_format,
            "preferred_length": preferred_length,
            "preferred_era": preferred_era
        },
        "implicit": {
            "genres_weight": {},
            "tags_weight": {},
            "format_weight": {},
            "avg_score_mean": 7.0,
            "popularity_mean": 0
        }
    }
    
    if not favorite_anime_list:
        return profile
        
    num_favorites = len(favorite_anime_list)
    genre_counts = {}
    tag_counts = {}
    format_counts = {}
    total_avg_score = 0
    total_popularity = 0
    
    for anime in favorite_anime_list:
        # Implicit Genres
        for g in anime.get('genres', []):
            genre_counts[g] = genre_counts.get(g, 0) + 1
            
        # Implicit Tags (Weight them by anilist rank if available)
        for t in anime.get('tags', []):
            t_name = t.get('name')
            t_rank = t.get('rank', 50)
            if t_rank > 60:  # Only count major themes
                tag_counts[t_name] = tag_counts.get(t_name, 0) + (t_rank / 100.0)
                
        # Implicit Formats
        fmt = anime.get('format')
        if fmt:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
            
        # Averages
        total_avg_score += (anime.get('averageScore') or 70) / 10.0
        total_popularity += (anime.get('popularity') or 0)
        
    # Normalize Implicit Weights
    for g, count in genre_counts.items():
        profile["implicit"]["genres_weight"][g] = count / num_favorites
        
    for t, weight in tag_counts.items():
        profile["implicit"]["tags_weight"][t] = weight / num_favorites
        
    for f, count in format_counts.items():
        profile["implicit"]["format_weight"][f] = count / num_favorites
        
    profile["implicit"]["avg_score_mean"] = total_avg_score / num_favorites
    profile["implicit"]["popularity_mean"] = total_popularity / num_favorites
    
    return profile

def check_length_match(candidate_eps, candidate_format, pref_length):
    if pref_length == "ANY" or not candidate_eps:
        return "ANY", 0.0
        
    if candidate_format == "MOVIE":
        return "MOVIE", 0.0 # Length logic applies less to movies
        
    eps = int(candidate_eps)
    
    if "SHORT" in pref_length: # < 14
        if eps < 14: return "MATCH", 1.0
        elif eps <= 26: return "MISMATCH_1", 0.0
        else: return "MISMATCH_2", -0.5
    elif "MEDIUM" in pref_length: # 14 - 26
        if 14 <= eps <= 26: return "MATCH", 1.0
        elif eps < 14: return "MISMATCH_1", 0.0
        elif eps <= 50: return "MISMATCH_1", 0.0
        else: return "MISMATCH_2", -0.5
    elif "LONG" in pref_length: # 27+
        if eps >= 27: return "MATCH", 1.0
        elif 14 <= eps <= 26: return "MISMATCH_1", 0.0
        else: return "MISMATCH_2", -1.0
        
    return "ANY", 0.0

def check_era_match(candidate_year, pref_era):
    if pref_era == "ANY" or not candidate_year:
        return 0.0
        
    year = int(candidate_year)
    
    if "CLASSICS" in pref_era:
        if year < 2000: return 1.0
        elif year <= 2010: return 0.0
        else: return -0.5
    elif "MODERN" in pref_era:
        if 2000 <= year <= 2015: return 1.0
        else: return -0.3
    elif "RECENT" in pref_era:
        if year > 2015: return 1.0
        elif year >= 2010: return 0.0
        else: return -0.5
        
    return 0.0

def content_based_heuristic_scorer(candidate, profile):
    """
    Returns a predicted score (0.0 to 10.0) and a list of human-readable explanations.
    """
    base_score = 5.0
    score = base_score
    explanations = []
    
    cand_genres = set(candidate.get('genres', []))
    cand_tags = {t['name']: t.get('rank', 50) for t in candidate.get('tags', [])}
    cand_format = candidate.get('format')
    cand_eps = candidate.get('episodes')
    cand_year = candidate.get('seasonYear')
    cand_avg_score = (candidate.get('averageScore') or 70) / 10.0
    
    # 1. EXPLICIT PREFERENCES
    # Avoided Genres (Severe Penalty)
    avoided_match = set(profile['explicit']['avoided_genres']).intersection(cand_genres)
    if avoided_match:
        penalty = len(avoided_match) * 2.5
        score -= penalty
        explanations.append(f"❌ Strongly discouraged (-{penalty}): Contains genres you avoid ({', '.join(avoided_match)}).")
        
    # Preferred Genres
    pref_match = set(profile['explicit']['preferred_genres']).intersection(cand_genres)
    if pref_match:
        bonus = len(pref_match) * 1.0
        score += bonus
        explanations.append(f"✅ Great Genre Match (+{bonus}): Contains your favorite genres ({', '.join(pref_match)}).")
        
    # Explicit Format
    pref_format = profile['explicit']['preferred_format']
    if pref_format != "ANY":
        if cand_format == pref_format:
            score += 0.8
            explanations.append(f"🎬 Preferred Format (+0.8): It is an anime {pref_format}.")
        elif cand_format:
            score -= 0.5
            explanations.append(f"⚠️ Different Format (-0.5): You preferred {pref_format}, but this is {cand_format}.")
            
    # Explicit Length
    pref_length = profile['explicit']['preferred_length']
    l_status, l_bonus = check_length_match(cand_eps, cand_format, pref_length)
    if l_status == "MATCH":
        score += 0.5
        explanations.append(f"⏱️ Ideal Length (+0.5): In line with your length preferences.")
    elif l_status == "MISMATCH_2":
        score += l_bonus
        explanations.append(f"⏳ Disliked Length ({l_bonus}): It is too long or too short for your tastes.")
        
    # Explicit Era
    pref_era = profile['explicit']['preferred_era']
    era_bonus = check_era_match(cand_year, pref_era)
    if era_bonus > 0:
        score += 0.5
        explanations.append(f"📅 Preferred Era (+0.5): Released in your preferred historical period ({cand_year}).")
    elif era_bonus < 0:
        score += era_bonus
        explanations.append(f"🕰️ Different Era ({era_bonus}): Released in a different period than what you sought.")

    # 2. IMPLICIT PREFERENCES (from favorites)
    # Implicit Genres matching (Bonus capped at +1.5 to not overthrow explicit)
    implicit_genre_score = 0.0
    for g in cand_genres:
        if g not in profile['explicit']['preferred_genres']: # Don't double count explicit
            implicit_genre_score += profile['implicit']['genres_weight'].get(g, 0.0) * 0.5
            
    if implicit_genre_score > 0:
        implicit_genre_score = min(implicit_genre_score, 1.5)
        score += implicit_genre_score
        explanations.append(f"🔍 Similar Traces (+{implicit_genre_score:.2f}): Shares genres with your initial favorite anime.")
        
    # Implicit Tags (Themes)
    implicit_tag_score = 0.0
    matched_tags = []
    for t_name, t_rank in cand_tags.items():
        if t_name in profile['implicit']['tags_weight']:
            # Multiply cand rank/100 by user tag weight
            weight = profile['implicit']['tags_weight'][t_name] * (t_rank / 100.0)
            implicit_tag_score += weight
            matched_tags.append(t_name)
            
    if implicit_tag_score > 0.3:
        implicit_tag_score = min(implicit_tag_score, 2.0)
        score += implicit_tag_score
        top_matched = ", ".join(matched_tags[:3])
        explanations.append(f"🎭 Similar Themes (+{implicit_tag_score:.2f}): Shares strong themes with your favorites (es. {top_matched}).")
        
    # Global Quality Base (Tie-breaker for overall goodness)
    global_diff = cand_avg_score - 7.0
    global_bonus = global_diff * 0.3 # If global 9.0 -> +0.6. If global 5.0 -> -0.6
    score += global_bonus
    if global_bonus > 0.3:
        explanations.append(f"🏆 Critically Acclaimed (+{global_bonus:.2f}): High global quality makes it a safe bet.")
    elif global_bonus < -0.3:
        explanations.append(f"📉 Low Global Rating ({global_bonus:.2f}): The community considers it mediocre.")
        
    # Clip final score
    final_score = float(np.clip(score, 0.0, 10.0))
    return final_score, explanations

def generate_cold_start_recommendations(profile, candidates_list):
    """
    Given the computed profile and a list of candidates from API,
    score them all and return the top 10 descending.
    """
    results = []
    
    avoided_genres = set(profile['explicit']['avoided_genres'])
    pref_format = profile['explicit']['preferred_format']
    
    for cand in candidates_list:
        cand_genres = set(cand.get('genres', []))
        if avoided_genres.intersection(cand_genres):
            continue
        
        # Hard filter format: if user picked a specific format, exclude mismatches
        if pref_format != "ANY" and cand.get('format') and cand.get('format') != pref_format:
            continue
            
        score, _ = content_based_heuristic_scorer(cand, profile)
        results.append((score, cand))
        
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:10]

# ============================
# MANGA COLD START
# ============================

MANGA_FORMATS_LIST = ["MANGA", "LIGHT_NOVEL", "ONE_SHOT", "ANY"]
MANGA_LENGTHS_LIST = ["SHORT (<20 cap)", "MEDIUM (20-100 cap)", "LONG (100+ cap)", "ANY"]

def build_manga_cold_start_profile(favorite_manga_list, preferred_genres, avoided_genres, preferred_format, preferred_length, preferred_era):
    """Builds a mock user profile for manga cold start."""
    profile = {
        "explicit": {
            "preferred_genres": preferred_genres or [],
            "avoided_genres": avoided_genres or [],
            "preferred_format": preferred_format,
            "preferred_length": preferred_length,
            "preferred_era": preferred_era
        },
        "implicit": {
            "genres_weight": {},
            "tags_weight": {},
            "format_weight": {},
            "avg_score_mean": 7.0,
            "popularity_mean": 0
        }
    }
    
    if not favorite_manga_list:
        return profile
        
    num_favorites = len(favorite_manga_list)
    genre_counts = {}
    tag_counts = {}
    format_counts = {}
    total_avg_score = 0
    total_popularity = 0
    
    for manga in favorite_manga_list:
        for g in manga.get('genres', []):
            genre_counts[g] = genre_counts.get(g, 0) + 1
            
        for t in manga.get('tags', []):
            t_name = t.get('name')
            t_rank = t.get('rank', 50)
            if t_rank > 60:
                tag_counts[t_name] = tag_counts.get(t_name, 0) + (t_rank / 100.0)
                
        fmt = manga.get('format')
        if fmt:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
            
        total_avg_score += (manga.get('averageScore') or 70) / 10.0
        total_popularity += (manga.get('popularity') or 0)
        
    for g, count in genre_counts.items():
        profile["implicit"]["genres_weight"][g] = count / num_favorites
    for t, weight in tag_counts.items():
        profile["implicit"]["tags_weight"][t] = weight / num_favorites
    for f, count in format_counts.items():
        profile["implicit"]["format_weight"][f] = count / num_favorites
        
    profile["implicit"]["avg_score_mean"] = total_avg_score / num_favorites
    profile["implicit"]["popularity_mean"] = total_popularity / num_favorites
    
    return profile

def check_manga_length_match(candidate_chapters, pref_length):
    if pref_length == "ANY" or not candidate_chapters:
        return "ANY", 0.0
        
    chaps = int(candidate_chapters)
    
    if "SHORT" in pref_length:
        if chaps < 20: return "MATCH", 1.0
        elif chaps <= 100: return "MISMATCH_1", 0.0
        else: return "MISMATCH_2", -0.5
    elif "MEDIUM" in pref_length:
        if 20 <= chaps <= 100: return "MATCH", 1.0
        elif chaps < 20: return "MISMATCH_1", 0.0
        elif chaps <= 200: return "MISMATCH_1", 0.0
        else: return "MISMATCH_2", -0.5
    elif "LONG" in pref_length:
        if chaps > 100: return "MATCH", 1.0
        elif 20 <= chaps <= 100: return "MISMATCH_1", 0.0
        else: return "MISMATCH_2", -1.0
        
    return "ANY", 0.0

def content_based_heuristic_scorer_manga(candidate, profile):
    """Returns a predicted score (0-10) and explanations for a manga candidate."""
    base_score = 5.0
    score = base_score
    explanations = []
    
    cand_genres = set(candidate.get('genres', []))
    cand_tags = {t['name']: t.get('rank', 50) for t in candidate.get('tags', [])}
    cand_format = candidate.get('format')
    cand_chapters = candidate.get('chapters')
    start_date = candidate.get('startDate', {}) or {}
    cand_year = start_date.get('year') if isinstance(start_date, dict) else None
    cand_avg_score = (candidate.get('averageScore') or 70) / 10.0
    
    # 1. EXPLICIT PREFERENCES
    avoided_match = set(profile['explicit']['avoided_genres']).intersection(cand_genres)
    if avoided_match:
        penalty = len(avoided_match) * 2.5
        score -= penalty
        explanations.append(f"❌ Strongly discouraged (-{penalty}): Contains genres you avoid ({', '.join(avoided_match)}).")
        
    pref_match = set(profile['explicit']['preferred_genres']).intersection(cand_genres)
    if pref_match:
        bonus = len(pref_match) * 1.0
        score += bonus
        explanations.append(f"✅ Great Genre Match (+{bonus}): Contains your favorite genres ({', '.join(pref_match)}).")
        
    pref_format = profile['explicit']['preferred_format']
    if pref_format != "ANY":
        if cand_format == pref_format:
            score += 0.8
            explanations.append(f"📖 Preferred Format (+0.8): It is a {pref_format}.")
        elif cand_format:
            score -= 0.5
            explanations.append(f"⚠️ Different Format (-0.5): You preferred {pref_format}, but this is {cand_format}.")
            
    pref_length = profile['explicit']['preferred_length']
    l_status, l_bonus = check_manga_length_match(cand_chapters, pref_length)
    if l_status == "MATCH":
        score += 0.5
        explanations.append(f"📏 Ideal Length (+0.5): In line with your chapter count preferences.")
    elif l_status == "MISMATCH_2":
        score += l_bonus
        explanations.append(f"📏 Disliked Length ({l_bonus}): Too many or too few chapters for your taste.")
        
    pref_era = profile['explicit']['preferred_era']
    era_bonus = check_era_match(cand_year, pref_era)
    if era_bonus > 0:
        score += 0.5
        explanations.append(f"📅 Preferred Era (+0.5): Published in your preferred historical period ({cand_year}).")
    elif era_bonus < 0:
        score += era_bonus
        explanations.append(f"🕰️ Different Era ({era_bonus}): Published in a different period than what you sought.")

    # 2. IMPLICIT PREFERENCES
    implicit_genre_score = 0.0
    for g in cand_genres:
        if g not in profile['explicit']['preferred_genres']:
            implicit_genre_score += profile['implicit']['genres_weight'].get(g, 0.0) * 0.5
    if implicit_genre_score > 0:
        implicit_genre_score = min(implicit_genre_score, 1.5)
        score += implicit_genre_score
        explanations.append(f"🔍 Similar Traces (+{implicit_genre_score:.2f}): Shares genres with your favorite manga.")
        
    implicit_tag_score = 0.0
    matched_tags = []
    for t_name, t_rank in cand_tags.items():
        if t_name in profile['implicit']['tags_weight']:
            weight = profile['implicit']['tags_weight'][t_name] * (t_rank / 100.0)
            implicit_tag_score += weight
            matched_tags.append(t_name)
    if implicit_tag_score > 0.3:
        implicit_tag_score = min(implicit_tag_score, 2.0)
        score += implicit_tag_score
        top_matched = ", ".join(matched_tags[:3])
        explanations.append(f"🎭 Similar Themes (+{implicit_tag_score:.2f}): Shares strong themes with your favorites (es. {top_matched}).")
        
    global_diff = cand_avg_score - 7.0
    global_bonus = global_diff * 0.3
    score += global_bonus
    if global_bonus > 0.3:
        explanations.append(f"🏆 Critically Acclaimed (+{global_bonus:.2f}): High global quality makes it a safe bet.")
    elif global_bonus < -0.3:
        explanations.append(f"📉 Low Global Rating ({global_bonus:.2f}): The community considers it mediocre.")
        
    final_score = float(np.clip(score, 0.0, 10.0))
    return final_score, explanations

def generate_manga_cold_start_recommendations(profile, candidates_list):
    """Score manga candidates and return top 10."""
    results = []
    avoided_genres = set(profile['explicit']['avoided_genres'])
    pref_format = profile['explicit']['preferred_format']
    
    for cand in candidates_list:
        cand_genres = set(cand.get('genres', []))
        if avoided_genres.intersection(cand_genres):
            continue
        
        # Hard filter format
        if pref_format != "ANY" and cand.get('format') and cand.get('format') != pref_format:
            continue
        
        score, _ = content_based_heuristic_scorer_manga(cand, profile)
        results.append((score, cand))
        
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:10]

