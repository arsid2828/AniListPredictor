import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import (
    MAX_USERNAME_LEN,
    USERNAME_VALIDATION_MESSAGE,
    init_session_state,
    inject_css,
    is_valid_username,
    normalize_limited_text,
    render_app_disclaimer,
    render_user_badge,
)
from src.analytics import compute_compatibility
from src.dataset import build_user_dataframe, build_user_manga_dataframe

st.set_page_config(page_title="User Compatibility", layout="wide", page_icon="👥")
inject_css()
init_session_state()
render_user_badge()

st.title("👥 Compatibility Between Two Users")
st.write("Compare the tastes of two AniList users based on commonly rated titles.")

media_type = st.radio("Type:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "Manga" if is_manga else "Anime"

col1, col2 = st.columns(2)
with col1:
    user1 = st.text_input(
        "👤 Username 1:",
        value=st.session_state.get("username", ""),
        key="compat_u1",
        max_chars=MAX_USERNAME_LEN,
    )
with col2:
    user2 = st.text_input("👤 Username 2:", key="compat_u2", max_chars=MAX_USERNAME_LEN)

user1_clean = normalize_limited_text(user1, MAX_USERNAME_LEN)
user2_clean = normalize_limited_text(user2, MAX_USERNAME_LEN)

if user1_clean and not is_valid_username(user1_clean):
    st.warning(USERNAME_VALIDATION_MESSAGE)
if user2_clean and not is_valid_username(user2_clean):
    st.warning(USERNAME_VALIDATION_MESSAGE)

if user1_clean and user2_clean and st.button("🔍 Calculate Compatibility", use_container_width=True):
    if not is_valid_username(user1_clean) or not is_valid_username(user2_clean):
        st.stop()

    if user1_clean.lower() == user2_clean.lower():
        st.warning("You must enter two different usernames!")
        st.stop()

    with st.spinner(f"Downloading {media_label.lower()} data for both users..."):
        if is_manga:
            df1 = build_user_manga_dataframe(user1_clean, force_refresh=False, save_csv=False)
            df2 = build_user_manga_dataframe(user2_clean, force_refresh=False, save_csv=False)
        else:
            df1 = build_user_dataframe(user1_clean, force_refresh=False, save_csv=False)
            df2 = build_user_dataframe(user2_clean, force_refresh=False, save_csv=False)

    if df1 is None or df1.empty:
        st.error(f"No data found for {user1_clean}. Private or non-existent profile?")
        st.stop()
    if df2 is None or df2.empty:
        st.error(f"No data found for {user2_clean}. Private or non-existent profile?")
        st.stop()

    result = compute_compatibility(df1, df2, user1_clean, user2_clean)

    if result["status"] == "insufficient":
        st.warning(f"Only {result['common_count']} common {media_label.lower()} found - at least 3 are needed for meaningful analysis.")
        st.stop()

    st.markdown("---")

    compat = result["compatibility_pct"]
    if compat >= 75:
        compat_color = "#00b09b"
        compat_emoji = "💚"
        compat_text = "Anime Soulmates!"
    elif compat >= 50:
        compat_color = "#f2994a"
        compat_emoji = "🤝"
        compat_text = "Good Compatibility"
    else:
        compat_color = "#eb3349"
        compat_emoji = "⚡"
        compat_text = "Different Tastes"

    gauge_col, stats_col = st.columns([2, 3])

    with gauge_col:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=compat,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{compat_emoji} Compatibility", "font": {"size": 20}},
                number={"suffix": "%", "font": {"size": 50}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": compat_color},
                    "steps": [
                        {"range": [0, 33], "color": "rgba(235, 51, 67, 0.1)"},
                        {"range": [33, 66], "color": "rgba(242, 153, 74, 0.1)"},
                        {"range": [66, 100], "color": "rgba(0, 176, 155, 0.1)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with stats_col:
        st.markdown(f"### {compat_text}")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(f"Common {media_label}", result["common_count"])
        with m2:
            st.metric("Correlation", f"{result['correlation']:.2f}")
        with m3:
            st.metric("Mean Difference", f"{result['mae']:.2f}")
        with m4:
            st.metric("Averages", f"{result['mean_1']:.1f} vs {result['mean_2']:.1f}")

    st.markdown("---")

    common_df = result["common_df"]
    fig_scatter = px.scatter(
        common_df,
        x="user_score_1",
        y="user_score_2",
        hover_data=["title"],
        labels={"user_score_1": f"Score {user1_clean}", "user_score_2": f"Score {user2_clean}"},
        title=f"Score Comparison - {user1_clean} vs {user2_clean}",
        color_discrete_sequence=["#667eea"],
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, 10],
            y=[0, 10],
            mode="lines",
            line=dict(dash="dash", color="rgba(255,255,255,0.3)"),
            name="Perfect Agreement",
            showlegend=True,
        )
    )
    fig_scatter.update_layout(
        template="plotly_dark",
        height=450,
        xaxis=dict(range=[0, 10.5]),
        yaxis=dict(range=[0, 10.5]),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    common_df["diff"] = (common_df["user_score_1"] - common_df["user_score_2"]).abs()
    common_df = common_df.sort_values("diff", ascending=False)

    st.subheader("🔥 Biggest Disagreements")
    for _, row in common_df.head(10).iterrows():
        title = row.get("title", row.get("title_1", "N/A"))
        diff = row["diff"]
        score1, score2 = row["user_score_1"], row["user_score_2"]
        badge = "🟢" if diff < 1 else ("🟡" if diff < 2 else "🔴")
        st.write(f"{badge} **{title}** - {user1_clean}: {score1:.1f} vs {user2_clean}: {score2:.1f} (Δ {diff:.1f})")

    st.subheader("💚 Most Agreed Upon")
    for _, row in common_df.tail(10).iloc[::-1].iterrows():
        title = row.get("title", row.get("title_1", "N/A"))
        diff = row["diff"]
        score1, score2 = row["user_score_1"], row["user_score_2"]
        st.write(f"🤝 **{title}** - {user1_clean}: {score1:.1f} vs {user2_clean}: {score2:.1f} (Δ {diff:.1f})")

render_app_disclaimer()
