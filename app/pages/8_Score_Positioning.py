import json
import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import init_session_state, inject_css, render_app_disclaimer, render_user_badge
from src.api import CACHE_DIR
from src.dataset import DATA_DIR


PAGE_CSS = """
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


def _init_positioning_run(items, anchor_id):
    anchor_item = next(item for item in items if item["id"] == anchor_id)
    opponents = [item for item in items if item["id"] != anchor_id]
    random.shuffle(opponents)
    participants = [anchor_item] + opponents
    return {
        "pool_signature": _pool_signature(participants),
        "participants": participants,
        "anchor_item": anchor_item,
        "opponents": opponents,
        "stats": _seed_stats(participants),
        "anchor_id": anchor_id,
        "current_index": 0,
        "history": [],
        "finished": len(opponents) < 1,
    }


def _advance_positioning(state, prefer_anchor: bool):
    anchor_item = state["anchor_item"]
    challenger = state["opponents"][state["current_index"]]
    winner = anchor_item if prefer_anchor else challenger
    loser = challenger if prefer_anchor else anchor_item

    state["stats"][winner["id"]]["wins"] += 1
    state["stats"][winner["id"]]["round_reached"] += 1
    state["stats"][loser["id"]]["losses"] += 1
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

    state["current_index"] += 1
    if state["current_index"] >= len(state["opponents"]):
        state["finished"] = True


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
        meta.append(f"Current Score: {item['score']:.1f}")
    if item.get("year"):
        meta.append(str(item["year"]))
    if item.get("genres"):
        meta.append(", ".join(item["genres"][:3]))
    if meta:
        st.caption(" | ".join(meta))
    return st.button(label, key=button_key, width="stretch")


def _render_ranking(state):
    rows = []
    anchor_id = state.get("anchor_id")
    for item_id, info in state["stats"].items():
        duel_wins = info["wins"]
        if item_id == anchor_id:
            stage_label = "Target title"
        elif duel_wins <= 0:
            stage_label = "Eliminated immediately"
        else:
            stage_label = f"Won {duel_wins} duel{'s' if duel_wins != 1 else ''}"
        rows.append(
            {
                "Title": info["title"],
                "Wins": info["wins"],
                "Losses": info["losses"],
                "Stage": stage_label,
                "Target": "🎯" if item_id == anchor_id else "",
            }
        )
    ranking_df = pd.DataFrame(rows).sort_values(["Target", "Wins"], ascending=[False, False])
    st.dataframe(ranking_df, width="stretch", hide_index=True)


def _render_score_helper(state):
    anchor_id = state.get("anchor_id")
    anchor_stats = state["stats"].get(anchor_id)
    if not anchor_stats:
        return

    anchor_title = anchor_stats["title"]
    anchor_score = float(anchor_stats.get("score", 0) or 0)
    if anchor_score <= 0:
        return

    tolerance = 0.05
    beat_higher_scores = []
    lost_lower_scores = []
    beat_lower_count = 0
    lost_higher_count = 0
    same_score_count = 0

    for row in state["history"]:
        if row.get("winner_id") == anchor_id:
            opponent_score = float(row.get("loser_score") or 0)
            if opponent_score > anchor_score + tolerance:
                beat_higher_scores.append(opponent_score)
            elif opponent_score < anchor_score - tolerance:
                beat_lower_count += 1
            else:
                same_score_count += 1
        elif row.get("loser_id") == anchor_id:
            opponent_score = float(row.get("winner_score") or 0)
            if opponent_score < anchor_score - tolerance:
                lost_lower_scores.append(opponent_score)
            elif opponent_score > anchor_score + tolerance:
                lost_higher_count += 1
            else:
                same_score_count += 1

    raise_pressure = sum(score - anchor_score for score in beat_higher_scores)
    lower_pressure = sum(anchor_score - score for score in lost_lower_scores)

    st.markdown("### Score Positioning Helper")

    if raise_pressure > lower_pressure + tolerance:
        target_score = min(max(beat_higher_scores or [anchor_score]), 10.0)
        st.info(
            f"**{anchor_title}** is currently at **{anchor_score:.1f}**. It beat **{len(beat_higher_scores)}** higher-rated title(s) "
            f"and only lost to lower-rated titles **{len(lost_lower_scores)}** time(s). "
            f"Suggestion: move it upward toward **{target_score:.1f}**."
        )
    elif lower_pressure > raise_pressure + tolerance:
        target_score = max(min(lost_lower_scores or [anchor_score]), 0.0)
        st.info(
            f"**{anchor_title}** is currently at **{anchor_score:.1f}**. It lost to **{len(lost_lower_scores)}** lower-rated title(s) "
            f"and only beat higher-rated titles **{len(beat_higher_scores)}** time(s). "
            f"Suggestion: move it downward toward **{target_score:.1f}**."
        )
    else:
        st.info(
            f"**{anchor_title}** is currently at **{anchor_score:.1f}**. Its results are mostly coherent with your current score: "
            f"{beat_lower_count} expected win(s) over lower-rated titles, {lost_higher_count} expected loss(es) to higher-rated titles, "
            f"and {same_score_count} match(es) against similarly rated titles. Suggestion: keep it around **{anchor_score:.1f}**."
        )


st.set_page_config(page_title="Score Positioning", layout="wide", page_icon="🎯")
inject_css()
st.markdown(PAGE_CSS, unsafe_allow_html=True)
init_session_state()
render_user_badge()

st.title("🎯 Score Positioning")
st.write("Pick one title, compare it against the rest of your filtered list, and use the results to understand whether its current score should go up, down, or stay where it is.")

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Return to the main page, load a user profile, and train the model first.")
    st.stop()

media_type = st.radio("Type:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "manga" if is_manga else "anime"

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
    st.warning("The current filters leave fewer than 2 titles. Broaden the filters to start a positioning run.")
    render_app_disclaimer()
    st.stop()

st.caption(f"{len(pool)} rated {media_label} available after filters.")

anchor_options = {item["title"]: item["id"] for item in sorted(pool, key=lambda x: x["title"].lower())}
selected_anchor_title = st.selectbox(
    "Title to position",
    options=list(anchor_options.keys()),
    index=0,
    help="This title starts every duel and is compared against the rest of your filtered pool.",
)
selected_anchor_id = anchor_options[selected_anchor_title]

state_key = f"score_positioning_{'manga' if is_manga else 'anime'}_{username.lower()}"
if st.button("🔄 Start New Positioning Run", width="stretch"):
    st.session_state[state_key] = _init_positioning_run(pool, selected_anchor_id)
    st.rerun()

if state_key not in st.session_state:
    st.session_state[state_key] = _init_positioning_run(pool, selected_anchor_id)

state = st.session_state[state_key]
if state.get("pool_signature") != _pool_signature(pool) or state.get("anchor_id") != selected_anchor_id:
    st.session_state[state_key] = _init_positioning_run(pool, selected_anchor_id)
    state = st.session_state[state_key]

_render_score_helper(state)

if not state["finished"]:
    left = state["anchor_item"]
    right = state["opponents"][state["current_index"]]
    st.subheader(f"Challenge {state['current_index'] + 1} / {len(state['opponents'])}")

    col1, col2 = st.columns(2)
    with col1:
        pick_left = _render_card(left, f"{state_key}_left_{left['id']}_{state['current_index']}", f"Choose {left['title']}")
    with col2:
        pick_right = _render_card(right, f"{state_key}_right_{right['id']}_{state['current_index']}", f"Choose {right['title']}")

    if pick_left:
        _advance_positioning(state, prefer_anchor=True)
        st.rerun()

    if pick_right:
        _advance_positioning(state, prefer_anchor=False)
        st.rerun()
else:
    st.success("Positioning run complete!")
    anchor_stats = state["stats"].get(state["anchor_id"], {})
    st.markdown(
        f"## 🎯 {state['anchor_item']['title']} finished the run with "
        f"**{anchor_stats.get('wins', 0)}** preferred duel(s) and **{anchor_stats.get('losses', 0)}** loss(es)."
    )
    _render_score_helper(state)

st.markdown("---")
st.subheader("Leaderboard")
_render_ranking(state)

with st.expander("📚 Recent Match History"):
    if not state["history"]:
        st.write("No matches played yet.")
    else:
        for row in reversed(state["history"][-20:]):
            st.write(
                f"**{row['winner']}** ({row['winner_score']:.1f}) beat "
                f"{row['loser']} ({row['loser_score']:.1f})"
            )

render_app_disclaimer()
