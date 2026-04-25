import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.shared import init_session_state, inject_css, render_app_disclaimer, render_user_badge
from src.analytics import (
    compute_franchise_stats,
    compute_genre_stats,
    compute_score_timeline,
    compute_studio_stats,
    compute_user_bias,
)
from src.dataset import DATA_DIR

st.set_page_config(page_title="Profile Analysis", layout="wide", page_icon="📊")
inject_css()
init_session_state()
render_user_badge()

st.title("📊 User Profile Analysis")

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Return to the main page, enter your username, and train the model.")
    st.stop()

media_type = st.radio("Type:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "Manga" if is_manga else "Anime"

csv_name = f"{username}_manga_clean.csv" if is_manga else f"{username}_clean.csv"
csv_path = DATA_DIR / csv_name

if not csv_path.exists():
    st.warning(f"No {media_label.lower()} dataset found for {username}. Train the model first.")
    st.stop()

df = pd.read_csv(csv_path)
df["sort_date"] = pd.to_datetime(df["sort_date"], errors="coerce")

st.markdown(f"### **{username}** Profile - {len(df)} rated {media_label.lower()}")

with st.expander("📚 Metric Glossary"):
    st.markdown(
        """
    *   **Average Score**: Your personal average rating.
    *   **Std Deviation**: How much your ratings vary. A high deviation means many 10s and many 1s. A low deviation means your scores stay close together.
    *   **Tendency**: Compares your rating style with AniList's global average.
        *   **Generous**: You usually rate higher than other users.
        *   **Strict**: You usually rate lower than other users.
    *   **Contrarian Index**: Measures how much your tastes diverge from the crowd. The higher it is, the more unconventional your ratings are.
    """
    )

bias = compute_user_bias(df)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Average Score", bias["mean_score"], help="Your average score calculated over all rated titles.")
with col2:
    st.metric("Std Deviation", bias["std_score"], help="Variability of your scores. A deviation around 1.5 is normal.")
with col3:
    st.metric("Rated Titles", bias["total_rated"])
with col4:
    colors = {"generous": "#00b09b", "strict": "#eb3349", "balanced": "#a0a0b0"}
    direction_emoji = {"generous": "😇", "strict": "😤", "balanced": "😐"}
    label = bias["direction"].title()
    color = colors.get(bias["direction"], "#ffffff")

    st.markdown(
        f"""
    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; border-left: 5px solid {color};">
        <div style="font-size: 0.8rem; color: #a0a0b0; text-transform: uppercase;">Tendency</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: {color};">
            {direction_emoji.get(bias['direction'], '')} {label}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col5:
    st.metric("Contrarian Index", bias["contrarian_index"], help="The higher it is, the more your scores diverge from the community average.")

st.caption(
    f"📏 Average deviation from the community: **{bias['mean_diff']:+.2f}** points | "
    f"Mean absolute deviation: **{bias['abs_mean_diff']:.2f}**"
)

st.markdown("---")

col_hist, col_box = st.columns(2)

with col_hist:
    fig_hist = px.histogram(
        df,
        x="user_score",
        nbins=20,
        title=f"{media_label} Score Distribution",
        color_discrete_sequence=["#667eea"],
        labels={"user_score": "User Score"},
    )
    fig_hist.update_layout(template="plotly_dark", height=350, bargap=0.05)
    fig_hist.add_vline(
        x=df["user_score"].mean(),
        line_dash="dash",
        line_color="#f2994a",
        annotation_text=f"Average: {df['user_score'].mean():.2f}",
    )
    st.plotly_chart(fig_hist, width="stretch")

with col_box:
    fig_box = px.box(
        df,
        y="user_score",
        title="Score Box Plot",
        color_discrete_sequence=["#764ba2"],
        labels={"user_score": "User Score"},
    )
    fig_box.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_box, width="stretch")

st.markdown("---")
timeline = compute_score_timeline(df)

if not timeline.empty:
    fig_timeline = go.Figure()
    fig_timeline.add_trace(
        go.Scatter(
            x=timeline["index"],
            y=timeline["user_score"],
            mode="markers",
            name="Single Rating",
            marker=dict(size=5, color="rgba(102, 126, 234, 0.4)"),
            text=timeline["title"],
            hovertemplate="%{text}<br>Score: %{y:.1f}",
        )
    )
    fig_timeline.add_trace(
        go.Scatter(
            x=timeline["index"],
            y=timeline["rolling_mean"],
            mode="lines",
            name="Rolling Average (10)",
            line=dict(color="#f2994a", width=3),
        )
    )
    fig_timeline.update_layout(
        title="⏳ Temporal Evolution of Scores",
        xaxis_title=f"{media_label} consumed (chronological order)",
        yaxis_title="Score",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_timeline, width="stretch")

st.markdown("---")
col_radar, col_genre_bar = st.columns(2)

genre_stats = compute_genre_stats(df)
if not genre_stats.empty:
    radar_data = genre_stats[genre_stats["count"] >= 3].head(12)

    with col_radar:
        if not radar_data.empty:
            fig_radar = go.Figure(
                data=go.Scatterpolar(
                    r=radar_data["mean_score"].values,
                    theta=radar_data["genre"].values,
                    fill="toself",
                    fillcolor="rgba(102, 126, 234, 0.2)",
                    line=dict(color="#667eea", width=2),
                    name="Average Score",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10]), bgcolor="rgba(0,0,0,0)"),
                title="🎯 Genre Radar (min. 3 titles)",
                template="plotly_dark",
                height=450,
            )
            st.plotly_chart(fig_radar, width="stretch")

    with col_genre_bar:
        fig_genre = px.bar(
            genre_stats,
            x="mean_score",
            y="genre",
            orientation="h",
            color="count",
            color_continuous_scale="Viridis",
            title="Average Score per Genre",
            labels={"mean_score": "Average Score", "genre": "Genre", "count": "No. of Titles"},
        )
        fig_genre.update_layout(template="plotly_dark", height=450, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_genre, width="stretch")

st.markdown("---")
st.markdown(f"### 🏆 Top 10 Favorite Franchises ({media_label})")

franchise_stats = compute_franchise_stats(df, username, is_manga=is_manga)

if not franchise_stats.empty:
    top_franchises = franchise_stats.head(10).copy()
    top_franchises = top_franchises.sort_values("final_score", ascending=True)

    fig_franchise = px.bar(
        top_franchises,
        x="final_score",
        y="franchise",
        orientation="h",
        color="final_score",
        color_continuous_scale="Sunsetdark",
        title="Top Franchises by Score + Presence Bonus",
        labels={"final_score": "Franchise Score", "franchise": "Franchise"},
        hover_data={"avg_score": True, "total_entries": True, "total_progress": True},
    )
    fig_franchise.update_layout(template="plotly_dark", height=500)
    fig_franchise.update_traces(
        hovertemplate="<b>%{y}</b><br>Final Score: %{x:.2f}<br>Average Score: %{customdata[0]:.2f}<br>Total Entries Watched/Read: %{customdata[1]}<br>Total Episodes/Chapters: %{customdata[2]}<extra></extra>"
    )
    st.plotly_chart(fig_franchise, width="stretch")

    with st.expander("ℹ️ How is the franchise score calculated?"):
        st.markdown(
            """
        The franchise score is not a simple average of your ratings. It also considers **how much** of that franchise you consumed, balancing multi-season works (for example *Attack on Titan*) with very long single-series works (for example *One Piece*).

        **Calculation rules:**
        1. **Average Score**: The arithmetic mean of all your ratings inside the franchise.
        2. **Presence Bonus**: An extra bonus rewards how much content you consumed:
           - **+0.05 points** for each entry (season, movie, OVA, special, and so on).
           - **+0.1 points** every 50 episodes, or every 100 chapters for manga.
        """
        )
else:
    st.info("Not enough relational data found to calculate franchises, or no linked series were found.")

st.markdown("---")
studio_col_name = "authors" if is_manga and "authors" in df.columns else "studios"
entity_label = "Authors" if is_manga else "Studios"

studio_stats = compute_studio_stats(df, col=studio_col_name, min_count=2)
if not studio_stats.empty:
    top_studios = studio_stats.head(15)
    fig_studio = px.bar(
        top_studios,
        x="mean_score",
        y="name",
        orientation="h",
        color="count",
        color_continuous_scale="Plasma",
        title=f"🏢 Top {entity_label} by Average Score (min. 2 titles)",
        labels={"mean_score": "Average Score", "name": entity_label, "count": "No. of Titles"},
    )
    fig_studio.update_layout(template="plotly_dark", height=450, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_studio, width="stretch")

if "seasonYear" in df.columns or "releaseYear" in df.columns:
    st.markdown("---")
    year_col = "releaseYear" if "releaseYear" in df.columns and is_manga else "seasonYear"
    if year_col in df.columns:
        year_df = df.dropna(subset=[year_col]).copy()
        year_df[year_col] = year_df[year_col].astype(int)
        year_stats = year_df.groupby(year_col).agg(mean_score=("user_score", "mean"), count=("user_score", "count")).reset_index()
        year_stats = year_stats[year_stats["count"] >= 1]

        if not year_stats.empty:
            fig_year = px.bar(
                year_stats,
                x=year_col,
                y="mean_score",
                color="count",
                color_continuous_scale="Turbo",
                title="📅 Average Score by Release Year",
                labels={year_col: "Year", "mean_score": "Average Score", "count": "No. of Titles"},
            )
            fig_year.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_year, width="stretch")

render_app_disclaimer()
