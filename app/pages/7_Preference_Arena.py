import json
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
ROOT_DIR = APP_DIR.parent
for path in (str(ROOT_DIR), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from shared import init_session_state, inject_css, render_app_disclaimer, render_user_badge
from src.api import CACHE_DIR
from src.dataset import DATA_DIR


ARENA_CSS = """
<style>
    .arena-cover-frame {
        width: min(100%, 280px);
        aspect-ratio: 2 / 3;
        overflow: hidden;
        border-radius: 14px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.9rem;
        margin-left: auto;
        margin-right: auto;
    }

    .arena-cover-frame img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    
    .arena-cover-empty {
        color: #9aa3b2;
        font-size: 0.95rem;
        text-align: center;
        padding: 1rem;
    }
    
    .arena-filter-box {
        padding: 1rem 1rem 0.35rem 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
    }
</style>
"""

ALL_GENRES = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Ecchi",
    "Fantasy",
    "Hentai",
    "Horror",
    "Mahou Shoujo",
    "Mecha",
    "Music",
    "Mystery",
    "Psychological",
    "Romance",
    "Sci-Fi",
    "Slice of Life",
    "Sports",
    "Supernatural",
    "Thriller",
]


def _load_media_pool(username: str, is_manga: bool):
    csv_name = f"{username.lower()}_manga_clean.csv" if is_manga else f"{username.lower()}_clean.csv"
    cache_name = f"user_manga_list_{username.lower()}.json" if is_manga else f"user_list_{username.lower()}.json"
    csv_path = DATA_DIR / csv_name
    cache_path = CACHE_DIR / cache_name

    media_lookup = {}
    cached_entries = []
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for group in raw:
                for entry in group.get("entries", []):
                    media = entry.get("media", {}) or {}
                    media_id = entry.get("mediaId") or media.get("id")
                    if media_id:
                        media_lookup[int(media_id)] = media
                        cached_entries.append(entry)
        except Exception:
            media_lookup = {}
            cached_entries = []

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if not df.empty:
            items = []
            for _, row in df.iterrows():
                media_id = int(row["mediaId"])
                media = media_lookup.get(media_id, {})
                title = media.get("title", {}).get("english") or media.get("title", {}).get("romaji") or row.get("title") or str(media_id)
                items.append(
                    {
                        "id": media_id,
                        "title": title,
                        "cover": (media.get("coverImage") or {}).get("large", ""),
                        "score": float(row.get("user_score", 0) or 0),
                        "format": row.get("format", ""),
                        "genres": [g.strip() for g in str(row.get("genres", "")).split(",") if g.strip()],
                        "tags": [tag.split("=")[0].strip() for tag in str(row.get("tags", "")).split(",") if tag.strip()],
                        "year": int(pd.to_numeric(row.get("seasonYear" if not is_manga else "releaseYear"), errors="coerce"))
                        if pd.notna(pd.to_numeric(row.get("seasonYear" if not is_manga else "releaseYear"), errors="coerce"))
                        else None,
                    }
                )
            if items:
                return items

    items = []
    seen_ids = set()
    for entry in cached_entries:
        media = entry.get("media", {}) or {}
        media_id = entry.get("mediaId") or media.get("id")
        score = float(entry.get("score") or 0)
        if not media_id or score <= 0 or media_id in seen_ids:
            continue
        seen_ids.add(media_id)
        title = media.get("title", {}).get("english") or media.get("title", {}).get("romaji") or str(media_id)
        items.append(
            {
                "id": media_id,
                "title": title,
                "cover": (media.get("coverImage") or {}).get("large", ""),
                "score": score,
                "format": media.get("format", ""),
                "genres": [g.strip() for g in media.get("genres", []) if g.strip()],
                "tags": [tag.get("name", "").strip() for tag in media.get("tags", []) if tag.get("name")],
                "year": media.get("seasonYear") or (media.get("startDate") or {}).get("year"),
            }
        )
    return items


def _get_state_key(username: str, is_manga: bool, mode: str):
    suffix = "manga" if is_manga else "anime"
    return f"arena_{mode}_{suffix}_{username.lower()}"


def _seed_stats(items):
    return {
        item["id"]: {
            "wins": 0,
            "losses": 0,
            "round_reached": 0,
            "title": item["title"],
            "score": float(item.get("score", 0) or 0),
        }
        for item in items
    }


def _pool_signature(items):
    return tuple(sorted(item["id"] for item in items))


def _collect_filter_options(items):
    pool_genres = {genre for item in items for genre in item.get("genres", []) if genre}
    genres = sorted(set(ALL_GENRES).union(pool_genres))
    tags = sorted({tag for item in items for tag in item.get("tags", []) if tag})
    years = sorted({int(item["year"]) for item in items if item.get("year")})
    scores = [float(item.get("score", 0) or 0) for item in items]
    return {
        "genres": genres,
        "tags": tags,
        "years": years,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 10.0,
    }


def _apply_filters(items, score_range, selected_genre, selected_year, selected_tag):
    filtered = []
    min_score, max_score = score_range
    for item in items:
        score = float(item.get("score", 0) or 0)
        if score < min_score or score > max_score:
            continue
        if selected_genre != "All" and selected_genre not in item.get("genres", []):
            continue
        if selected_year != "All" and item.get("year") != selected_year:
            continue
        if selected_tag != "All" and selected_tag not in item.get("tags", []):
            continue
        filtered.append(item)
    return filtered


def _init_gauntlet(items):
    participants = items[:]
    random.shuffle(participants)
    stats = _seed_stats(participants)
    return {
        "mode": "gauntlet",
        "pool_signature": _pool_signature(participants),
        "participants": participants,
        "stats": stats,
        "champion_idx": 0,
        "challenger_idx": 1,
        "history": [],
        "finished": len(participants) < 2,
    }


def _build_round(participants):
    matches = []
    i = 0
    while i < len(participants):
        left = participants[i]
        right = participants[i + 1] if i + 1 < len(participants) else None
        matches.append((left, right))
        i += 2
    return matches


def _init_tournament(items):
    participants = items[:]
    random.shuffle(participants)
    stats = _seed_stats(participants)
    return {
        "mode": "tournament",
        "pool_signature": _pool_signature(participants),
        "round_number": 1,
        "current_round": _build_round(participants),
        "match_index": 0,
        "round_winners": [],
        "stats": stats,
        "history": [],
        "finished": len(participants) < 2,
        "champion": participants[0]["id"] if len(participants) == 1 else None,
    }


def _record_result(stats, winner, loser, round_number):
    stats[winner["id"]]["wins"] += 1
    stats[winner["id"]]["round_reached"] = max(stats[winner["id"]]["round_reached"], round_number)
    if loser is not None:
        stats[loser["id"]]["losses"] += 1
        stats[loser["id"]]["round_reached"] = max(stats[loser["id"]]["round_reached"], round_number - 1)


def _advance_gauntlet(state, winner_id):
    champion = state["participants"][state["champion_idx"]]
    challenger = state["participants"][state["challenger_idx"]]
    winner = champion if champion["id"] == winner_id else challenger
    loser = challenger if winner is champion else champion

    _record_result(state["stats"], winner, loser, len(state["history"]) + 1)
    state["history"].append(
        {
            "winner": winner["title"],
            "winner_id": winner["id"],
            "winner_score": float(winner.get("score", 0) or 0),
            "loser": loser["title"],
            "loser_id": loser["id"],
            "loser_score": float(loser.get("score", 0) or 0),
        }
    )

    if winner is challenger:
        state["champion_idx"] = state["challenger_idx"]

    state["challenger_idx"] += 1
    if state["challenger_idx"] >= len(state["participants"]):
        state["finished"] = True


def _advance_tournament(state, winner_id):
    left, right = state["current_round"][state["match_index"]]
    if right is None:
        winner = left
        loser = None
    else:
        winner = left if left["id"] == winner_id else right
        loser = right if winner is left else left

    _record_result(state["stats"], winner, loser, state["round_number"])
    state["history"].append(
        {
            "round": state["round_number"],
            "winner": winner["title"],
            "winner_id": winner["id"],
            "winner_score": float(winner.get("score", 0) or 0),
            "loser": loser["title"] if loser else "BYE",
            "loser_id": loser["id"] if loser else None,
            "loser_score": float(loser.get("score", 0) or 0) if loser else None,
        }
    )
    state["round_winners"].append(winner)
    state["match_index"] += 1

    if state["match_index"] >= len(state["current_round"]):
        if len(state["round_winners"]) == 1:
            state["finished"] = True
            state["champion"] = state["round_winners"][0]["id"]
        else:
            next_round_items = state["round_winners"][:]
            state["round_number"] += 1
            state["current_round"] = _build_round(next_round_items)
            state["match_index"] = 0
            state["round_winners"] = []


def _render_card(item, button_key, label):
    if item.get("cover"):
        st.markdown(
            f"""
            <div class="arena-cover-frame">
                <img src="{item['cover']}" alt="{item['title']} cover">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="arena-cover-frame"><div class="arena-cover-empty">No cover available</div></div>', unsafe_allow_html=True)
    st.markdown(f"### {item['title']}")
    meta = []
    if item.get("format"):
        meta.append(str(item["format"]))
    if item.get("score"):
        meta.append(f"User Score: {item['score']:.1f}")
    if item.get("year"):
        meta.append(str(item["year"]))
    if item.get("genres"):
        meta.append(", ".join(item["genres"][:3]))
    if meta:
        st.caption(" | ".join(meta))
    return st.button(label, key=button_key, width="stretch")


def _render_ranking(state):
    rows = []
    champion_id = state.get("champion")
    if state["mode"] == "gauntlet" and state["finished"]:
        champion_id = state["participants"][state["champion_idx"]]["id"]

    max_stage = max((info["round_reached"] for info in state["stats"].values()), default=0)

    for item_id, info in state["stats"].items():
        stage = info["round_reached"]
        if item_id == champion_id:
            stage_label = "Champion"
        elif stage <= 0:
            stage_label = "Opening Round"
        elif state["mode"] == "tournament":
            if stage == max_stage - 1 and max_stage >= 2:
                stage_label = "Finalist"
            elif stage == max_stage - 2 and max_stage >= 3:
                stage_label = "Semifinalist"
            else:
                stage_label = f"Round {stage}"
        else:
            duel_wins = info["wins"]
            if duel_wins <= 0:
                stage_label = "Eliminated immediately"
            else:
                stage_label = f"Won {duel_wins} duel{'s' if duel_wins != 1 else ''}"
        rows.append(
            {
                "Title": info["title"],
                "Wins": info["wins"],
                "Losses": info["losses"],
                "Stage": stage_label,
                "Stage Order": info["wins"] if state["mode"] == "gauntlet" else stage,
                "Champion": "🏆" if item_id == champion_id else "",
            }
        )
    ranking_df = pd.DataFrame(rows).sort_values(["Champion", "Wins", "Stage Order"], ascending=[False, False, False])
    st.dataframe(ranking_df.drop(columns=["Stage Order"]), width="stretch", hide_index=True)


st.set_page_config(page_title="Preference Arena", layout="wide", page_icon="🏟️")
inject_css()
st.markdown(ARENA_CSS, unsafe_allow_html=True)
init_session_state()
render_user_badge()

st.title("🏟️ Preference Arena")
st.write("Choose between two titles at a time and build a preference ranking from your own AniList history.")

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Return to the main page, load a user profile, and train the model first.")
    st.stop()

media_type = st.radio("Type:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "manga" if is_manga else "anime"

mode_label = st.radio("Mode:", ["Progressive Challenge", "Tournament Bracket"], horizontal=True)
mode_key = "gauntlet" if "Progressive" in mode_label else "tournament"

pool = _load_media_pool(username, is_manga=is_manga)
if len(pool) < 2:
    st.warning(f"Not enough rated {media_label} found for {username}.")
    render_app_disclaimer()
    st.stop()

filter_options = _collect_filter_options(pool)

st.markdown('<div class="arena-filter-box">', unsafe_allow_html=True)
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
with filter_col1:
    score_range = st.slider(
        "User score range",
        min_value=float(filter_options["score_min"]),
        max_value=float(filter_options["score_max"]),
        value=(float(filter_options["score_min"]), float(filter_options["score_max"])),
        step=0.1,
    )
with filter_col2:
    selected_genre = st.selectbox("Genre", ["All"] + filter_options["genres"])
with filter_col3:
    selected_year = st.selectbox("Year", ["All"] + filter_options["years"])
with filter_col4:
    selected_tag = st.selectbox("Tag", ["All"] + filter_options["tags"])
st.markdown("</div>", unsafe_allow_html=True)

pool = _apply_filters(pool, score_range, selected_genre, selected_year, selected_tag)

if len(pool) < 2:
    st.warning("The current filters leave fewer than 2 titles. Broaden the filters to start a challenge.")
    render_app_disclaimer()
    st.stop()

st.caption(f"{len(pool)} rated {media_label} available after filters.")

state_key = _get_state_key(username, is_manga, mode_key)

control_col1, control_col2 = st.columns([2, 1])
with control_col1:
    st.info(
        "Progressive Challenge keeps the winner on screen against the next title. "
        "Tournament Bracket creates a classic 1v1 knockout until the final."
    )
with control_col2:
    if st.button("🔄 Start New Ranking", width="stretch"):
        st.session_state[state_key] = _init_gauntlet(pool) if mode_key == "gauntlet" else _init_tournament(pool)
        st.rerun()

if state_key not in st.session_state:
    st.session_state[state_key] = _init_gauntlet(pool) if mode_key == "gauntlet" else _init_tournament(pool)

state = st.session_state[state_key]

if state["mode"] != mode_key or state.get("pool_signature") != _pool_signature(pool):
    st.session_state[state_key] = _init_gauntlet(pool) if mode_key == "gauntlet" else _init_tournament(pool)
    state = st.session_state[state_key]

if not state["finished"]:
    if mode_key == "gauntlet":
        left = state["participants"][state["champion_idx"]]
        right = state["participants"][state["challenger_idx"]]
        st.subheader(f"Challenge {len(state['history']) + 1} / {len(state['participants']) - 1}")
    else:
        left, right = state["current_round"][state["match_index"]]
        st.subheader(f"Round {state['round_number']} - Match {state['match_index'] + 1} / {len(state['current_round'])}")
        if right is None:
            _advance_tournament(state, left["id"])
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        pick_left = _render_card(left, f"{state_key}_left_{left['id']}", f"Choose {left['title']}")
    with col2:
        pick_right = _render_card(right, f"{state_key}_right_{right['id']}", f"Choose {right['title']}")

    if pick_left:
        if mode_key == "gauntlet":
            _advance_gauntlet(state, left["id"])
        else:
            _advance_tournament(state, left["id"])
        st.rerun()

    if pick_right:
        if mode_key == "gauntlet":
            _advance_gauntlet(state, right["id"])
        else:
            _advance_tournament(state, right["id"])
        st.rerun()

else:
    st.success("Ranking complete!")
    if mode_key == "gauntlet":
        champion = state["participants"][state["champion_idx"]]
        st.markdown(f"## 🏆 Final Champion: {champion['title']}")
    else:
        champion_id = state.get("champion")
        champion_title = state["stats"].get(champion_id, {}).get("title", "Unknown")
        st.markdown(f"## 🏆 Tournament Winner: {champion_title}")

st.markdown("---")
st.subheader("Leaderboard")
_render_ranking(state)

with st.expander("📚 Recent Match History"):
    if not state["history"]:
        st.write("No matches played yet.")
    else:
        for row in reversed(state["history"][-20:]):
            if "round" in row:
                st.write(f"Round {row['round']}: **{row['winner']}** beat {row['loser']}")
            else:
                st.write(f"**{row['winner']}** beat {row['loser']}")

render_app_disclaimer()
