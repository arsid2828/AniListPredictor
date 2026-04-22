import json
import logging
import sys
import time
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import USERNAME_VALIDATION_MESSAGE, init_session_state, inject_css, is_valid_username, render_app_disclaimer, render_user_badge
from src.analytics import compute_activity_stats
from src.api import CACHE_DIR, fetch_user_activity_history

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Activity History", layout="wide", page_icon="📔")
inject_css()
init_session_state()
render_user_badge()

st.title("📔 Analytical Activity History")
st.markdown("Pure statistics based on your real AniList history (episodes watched and chapters read). Automatically excludes *Plan to Watch* activities and similar noise.")

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Return to the main page and enter your username to start.")
    st.stop()

if not is_valid_username(username):
    st.error(USERNAME_VALIDATION_MESSAGE)
    st.stop()

media_type_raw = st.radio("Select Media:", ["🎬 Anime", "📖 Manga"], horizontal=True)
media_type = "MANGA" if "Manga" in media_type_raw else "ANIME"
media_label = "Manga" if media_type == "MANGA" else "Anime"
unit_label = "Chapters" if media_type == "MANGA" else "Episodes"

history_path = CACHE_DIR / f"user_activity_{media_type.lower()}_{username.lower()}.json"

if st.button(f"⬇️ Download / Refresh {media_label} History (may take a while for active users)"):
    with st.spinner(f"Downloading {media_label} history from AniList... Please wait."):
        acts = fetch_user_activity_history(username, media_type=media_type, force_refresh=True)
        if acts:
            st.success(f"{media_label} history downloaded successfully. Reloading...")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("No activity found or private profile.")

if history_path.exists():
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            acts = json.load(f)

        stats = compute_activity_stats(acts, media_type=media_type)
        if stats:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"🏆 Record Day (most {unit_label.lower()})",
                    f"{stats['max_day']['date']}",
                    f"{stats['max_day']['episodes']} {unit_label[:2].lower()}. ({stats['max_day']['hours']:.1f} estimated hours)",
                )

            with st.expander(f"📚 What did you go through on {stats['max_day']['date']}?"):
                for title in stats["max_day"]["titles"]:
                    st.write(f"- {title}")

            st.markdown(f"#### {unit_label} by Month")
            if media_type == "ANIME":
                fig_month = go.Figure(
                    data=[
                        go.Bar(name=unit_label, x=stats["month_stats"]["month"], y=stats["month_stats"]["episodes"]),
                        go.Scatter(
                            name="Hours",
                            x=stats["month_stats"]["month"],
                            y=stats["month_stats"]["duration_hours"],
                            yaxis="y2",
                            mode="lines+markers",
                            line=dict(color="#eb3349"),
                        ),
                    ]
                )
                fig_month.update_layout(
                    template="plotly_dark",
                    yaxis=dict(title=unit_label, side="left"),
                    yaxis2=dict(title="Hours", overlaying="y", side="right"),
                    barmode="group",
                    legend=dict(x=0, y=1.2, orientation="h"),
                    height=400,
                )
            else:
                fig_month = px.bar(
                    stats["month_stats"],
                    x="month",
                    y="episodes",
                    title=f"{unit_label} Read per Month",
                    labels={"episodes": unit_label, "month": "Month"},
                    template="plotly_dark",
                )
                fig_month.update_layout(height=400)

            st.plotly_chart(fig_month, use_container_width=True)

            st.markdown(f"#### {unit_label} by Year")
            if media_type == "ANIME":
                fig_year = px.bar(
                    stats["year_stats"],
                    x="year",
                    y=["episodes", "duration_hours"],
                    barmode="group",
                    labels={"value": "Amount", "variable": "Metric", "year": "Year"},
                    template="plotly_dark",
                    height=400,
                )
            else:
                fig_year = px.bar(
                    stats["year_stats"],
                    x="year",
                    y="episodes",
                    labels={"episodes": unit_label, "year": "Year"},
                    text_auto=True,
                    template="plotly_dark",
                    height=400,
                )

            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info(f"Not enough {media_label.lower()} activity to generate analytical statistics.")
    except Exception:
        logger.exception("Error generating activity statistics.")
        st.error("Error generating statistics. Please try reloading the page.")
else:
    st.info(f"Click the button to download the {media_label} history.")

render_app_disclaimer()
