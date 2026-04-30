import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import (
    USERNAME_VALIDATION_MESSAGE,
    init_session_state,
    inject_css,
    is_valid_username,
    render_app_disclaimer,
    render_user_badge,
)
from src.analytics import compute_activity_stats
from src.api import CACHE_DIR, fetch_user_activity_history, get_last_api_error

logger = logging.getLogger(__name__)

ANIME_REFERENCE_MINUTES = 24.0
ANIME_COLOR = "#667eea"
MANGA_COLOR = "#f299c1"


def _load_history_rows(username, media_type):
    history_path = CACHE_DIR / f"user_activity_{media_type.lower()}_{username.lower()}.json"
    if not history_path.exists():
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        acts = json.load(f)
    stats = compute_activity_stats(
        acts,
        media_type=media_type,
        anime_reference_minutes=ANIME_REFERENCE_MINUTES,
    )
    if not stats:
        return None
    rows = stats.get("rows")
    if rows is None or rows.empty:
        return None
    return rows.copy()


def _build_combined_period_stats(rows, period_col):
    grouped = (
        rows.groupby([period_col, "media_type"], as_index=False)
        .agg(
            equivalent_units=("equivalent_units", "sum"),
            raw_units=("episodes", "sum"),
            duration_hours=("duration_hours", "sum"),
        )
        .sort_values(period_col)
    )
    grouped["media_label"] = grouped["media_type"].map({"ANIME": "Anime", "MANGA": "Manga"})
    grouped["raw_label"] = grouped.apply(
        lambda r: (
            f"{int(r['raw_units'])} episodes"
            if r["media_type"] == "ANIME"
            else f"{int(r['raw_units'])} chapters"
        ),
        axis=1,
    )
    return grouped


def _render_combined_bar(period_stats, x_col, title):
    fig = px.bar(
        period_stats,
        x=x_col,
        y="equivalent_units",
        color="media_label",
        barmode="stack",
        title=title,
        labels={
            x_col: "Month" if x_col == "month" else "Year",
            "equivalent_units": "Anime Episode Equivalent",
            "media_label": "Media",
        },
        color_discrete_map={"Anime": ANIME_COLOR, "Manga": MANGA_COLOR},
        custom_data=["raw_label", "duration_hours", "media_label"],
        template="plotly_dark",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{customdata[2]}: %{y:.2f} episode equivalents<br>"
            "Raw amount: %{customdata[0]}<br>"
            "Estimated hours: %{customdata[1]:.1f}<extra></extra>"
        )
    )
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


st.set_page_config(page_title="Activity History", layout="wide", page_icon="📔")
inject_css()
init_session_state()
render_user_badge()

st.title("📔 Analytical Activity History")
st.markdown(
    "Explore your AniList activity over time with a unified view for anime and manga. "
    "Anime is weighted by runtime, while manga-like formats are normalized with format-aware reading weights."
)

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Return to the main page and enter your username to start.")
    st.stop()

if not is_valid_username(username):
    st.error(USERNAME_VALIDATION_MESSAGE)
    st.stop()

display_mode = st.radio("Show Activity:", ["🎬 Anime", "📖 Manga", "🔀 Both"], horizontal=True)
selected_modes = ["ANIME", "MANGA"] if "Both" in display_mode else (["MANGA"] if "Manga" in display_mode else ["ANIME"])

button_label = (
    "⬇️ Download / Refresh Anime and Manga History"
    if len(selected_modes) == 2
    else f"⬇️ Download / Refresh {'Manga' if selected_modes[0] == 'MANGA' else 'Anime'} History"
)

if st.button(button_label):
    pretty = "Anime and Manga" if len(selected_modes) == 2 else ("Manga" if selected_modes[0] == "MANGA" else "Anime")
    with st.spinner(f"Downloading {pretty} history from AniList... Please wait."):
        got_any = False
        failures = []
        for media_type in selected_modes:
            acts = fetch_user_activity_history(username, media_type=media_type, force_refresh=True)
            got_any = got_any or bool(acts)
            if not acts:
                last_error = get_last_api_error()
                if last_error:
                    failures.append(f"{media_type.title()}: {last_error}")
        if got_any:
            st.success(f"{pretty} history downloaded successfully. Reloading...")
            time.sleep(1)
            st.rerun()
        elif failures:
            st.error("History download failed.\n\n" + "\n".join(failures))
        else:
            st.warning("No activity found or private profile.")

try:
    frames = []
    for media_type in selected_modes:
        rows = _load_history_rows(username, media_type)
        if rows is not None and not rows.empty:
            frames.append(rows)

    if not frames:
        requested = "anime and manga" if len(selected_modes) == 2 else ("manga" if selected_modes[0] == "MANGA" else "anime")
        st.info(f"Click the button to download the {requested} history.")
        render_app_disclaimer()
        st.stop()

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["media_label"] = all_rows["media_type"].map({"ANIME": "Anime", "MANGA": "Manga"})
    all_rows["title_label"] = all_rows.apply(
        lambda r: f"[{'Anime' if r['media_type'] == 'ANIME' else 'Manga'}] {r['title']}",
        axis=1,
    )

    day_stats = (
        all_rows.groupby("date", as_index=False)
        .agg(
            equivalent_units=("equivalent_units", "sum"),
            duration_hours=("duration_hours", "sum"),
            titles=("title_label", lambda x: sorted(set(x))),
        )
        .sort_values("date")
    )

    max_day_row = day_stats.loc[day_stats["equivalent_units"].idxmax()]
    mode_label = "Anime + Manga" if len(selected_modes) == 2 else ("Manga" if selected_modes[0] == "MANGA" else "Anime")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🏆 Record Day",
            str(max_day_row["date"]),
            f"{max_day_row['equivalent_units']:.2f} episode equivalents ({max_day_row['duration_hours']:.1f} estimated hours)",
        )
    with col2:
        st.metric("View Mode", mode_label, "Anime weighted by runtime, reading weighted by format")

    with st.expander(f"📚 What did you consume on {max_day_row['date']}?"):
        for title in max_day_row["titles"]:
            st.write(f"- {title}")

    st.markdown("#### Monthly Activity")
    month_stats = _build_combined_period_stats(all_rows, "month")
    fig_month = _render_combined_bar(
        month_stats,
        "month",
        "Monthly Activity (Anime episode equivalent)",
    )
    st.plotly_chart(fig_month, width="stretch")

    st.markdown("#### Yearly Activity")
    year_stats = _build_combined_period_stats(all_rows, "year")
    fig_year = _render_combined_bar(
        year_stats,
        "year",
        "Yearly Activity (Anime episode equivalent)",
    )
    st.plotly_chart(fig_year, width="stretch")

    with st.expander("ℹ️ How is the mixed chart calculated?"):
        st.markdown(
            f"""
            The combined charts normalize manga and anime onto the same scale:

            - **Anime**: weighted by runtime using **{ANIME_REFERENCE_MINUTES:.0f} minutes = 1 episode equivalent**
            - **Movies / long specials** therefore count more than a standard 24-minute TV episode
            - **Manga (JP)**: about **2.5 chapters = 1 episode equivalent**
            - **Manhwa / Manhua (KR/CN)**: about **2 chapters = 1 episode equivalent**
            - **Novel**: about **1.5 chapters = 1 episode equivalent**
            - **One-shot**: about **1 chapter = 1 episode equivalent**

            This keeps the mixed charts more faithful to real consumption time without making movies and long-form reading look identical to short TV episodes.
            """
        )

except Exception:
    logger.exception("Error generating activity statistics.")
    st.error("Error generating statistics. Please try reloading the page.")

render_app_disclaimer()
