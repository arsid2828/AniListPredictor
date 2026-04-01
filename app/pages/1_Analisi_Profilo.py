import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.dataset import DATA_DIR
from src.analytics import compute_genre_stats, compute_studio_stats, compute_score_timeline, compute_user_bias
from app.shared import inject_css, init_session_state, render_user_badge
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analisi Profilo", layout="wide", page_icon="📊")
inject_css()
init_session_state()
render_user_badge()

st.title("📊 Analisi del Profilo Utente")

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Torna alla pagina principale, inserisci il tuo username e traina il modello.")
    st.stop()

# Media toggle
media_type = st.radio("Tipo:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "Manga" if is_manga else "Anime"

csv_name = f"{username}_manga_clean.csv" if is_manga else f"{username}_clean.csv"
csv_path = DATA_DIR / csv_name

if not csv_path.exists():
    st.warning(f"Nessun dataset {media_label.lower()} trovato per {username}. Traina prima il modello!")
    st.stop()

df = pd.read_csv(csv_path)
df['sort_date'] = pd.to_datetime(df['sort_date'], errors='coerce')

st.markdown(f"### Profilo **{username}** — {len(df)} {media_label.lower()} valutati")

# ========== SUMMARY METRICS ==========
bias = compute_user_bias(df)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Media Voto", bias['mean_score'])
with col2:
    st.metric("Deviazione Std", bias['std_score'])
with col3:
    st.metric("Titoli Valutati", bias['total_rated'])
with col4:
    direction_emoji = {"generoso": "😇", "severo": "😤", "allineato": "😐"}
    st.metric("Tendenza", f"{direction_emoji.get(bias['direction'], '')} {bias['direction'].title()}")
with col5:
    st.metric("Indice Contrarian", bias['contrarian_index'])

st.caption(f"📏 Deviazione media dalla community: **{bias['mean_diff']:+.2f}** punti | "
           f"Scostamento assoluto medio: **{bias['abs_mean_diff']:.2f}**")

st.markdown("---")

# ========== SCORE DISTRIBUTION ==========
col_hist, col_box = st.columns(2)

with col_hist:
    fig_hist = px.histogram(df, x='user_score', nbins=20, 
                           title=f"Distribuzione Voti {media_label}",
                           color_discrete_sequence=['#667eea'],
                           labels={'user_score': 'Voto Utente'})
    fig_hist.update_layout(template='plotly_dark', height=350, bargap=0.05)
    fig_hist.add_vline(x=df['user_score'].mean(), line_dash="dash", line_color="#f2994a",
                       annotation_text=f"Media: {df['user_score'].mean():.2f}")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_box:
    fig_box = px.box(df, y='user_score', title=f"Box Plot Voti",
                    color_discrete_sequence=['#764ba2'],
                    labels={'user_score': 'Voto Utente'})
    fig_box.update_layout(template='plotly_dark', height=350)
    st.plotly_chart(fig_box, use_container_width=True)

# ========== SCORE TIMELINE ==========
st.markdown("---")
timeline = compute_score_timeline(df)

if not timeline.empty:
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=timeline['index'], y=timeline['user_score'],
        mode='markers', name='Voto Singolo',
        marker=dict(size=5, color='rgba(102, 126, 234, 0.4)'),
        text=timeline['title'], hovertemplate='%{text}<br>Voto: %{y:.1f}'
    ))
    fig_timeline.add_trace(go.Scatter(
        x=timeline['index'], y=timeline['rolling_mean'],
        mode='lines', name='Media Mobile (10)',
        line=dict(color='#f2994a', width=3)
    ))
    fig_timeline.update_layout(
        title=f"⏳ Evoluzione Temporale dei Voti",
        xaxis_title=f"{media_label} visti (ordine cronologico)",
        yaxis_title="Voto",
        template='plotly_dark', height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ========== GENRE RADAR ==========
st.markdown("---")
col_radar, col_genre_bar = st.columns(2)

genre_stats = compute_genre_stats(df)
if not genre_stats.empty:
    # Filter genres with at least 3 entries for radar
    radar_data = genre_stats[genre_stats['count'] >= 3].head(12)
    
    with col_radar:
        if not radar_data.empty:
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_data['mean_score'].values,
                theta=radar_data['genre'].values,
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='#667eea', width=2),
                name='Score Medio'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10]),
                    bgcolor='rgba(0,0,0,0)'
                ),
                title="🎯 Radar Generi (min. 3 titoli)",
                template='plotly_dark', height=450
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col_genre_bar:
        fig_genre = px.bar(genre_stats, x='mean_score', y='genre', orientation='h',
                          color='count', color_continuous_scale='Viridis',
                          title="Voto Medio per Genere",
                          labels={'mean_score': 'Score Medio', 'genre': 'Genere', 'count': 'N° Titoli'})
        fig_genre.update_layout(template='plotly_dark', height=450, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_genre, use_container_width=True)

# ========== STUDIO / AUTHOR RANKING ==========
st.markdown("---")
studio_col_name = 'authors' if is_manga and 'authors' in df.columns else 'studios'
entity_label = "Autori" if is_manga else "Studi"

studio_stats = compute_studio_stats(df, col=studio_col_name, min_count=2)
if not studio_stats.empty:
    top_studios = studio_stats.head(15)
    fig_studio = px.bar(top_studios, x='mean_score', y='name', orientation='h',
                       color='count', color_continuous_scale='Plasma',
                       title=f"🏢 Top {entity_label} per Voto Medio (min. 2 titoli)",
                       labels={'mean_score': 'Score Medio', 'name': entity_label, 'count': 'N° Titoli'})
    fig_studio.update_layout(template='plotly_dark', height=450, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_studio, use_container_width=True)

# ========== YEAR HEATMAP ==========
if 'seasonYear' in df.columns or 'releaseYear' in df.columns:
    st.markdown("---")
    year_col = 'releaseYear' if 'releaseYear' in df.columns and is_manga else 'seasonYear'
    if year_col in df.columns:
        year_df = df.dropna(subset=[year_col]).copy()
        year_df[year_col] = year_df[year_col].astype(int)
        year_stats = year_df.groupby(year_col).agg(
            mean_score=('user_score', 'mean'),
            count=('user_score', 'count')
        ).reset_index()
        year_stats = year_stats[year_stats['count'] >= 2]
        
        if not year_stats.empty:
            fig_year = px.bar(year_stats, x=year_col, y='mean_score',
                             color='count', color_continuous_scale='Turbo',
                             title="📅 Voto Medio per Anno di Uscita",
                             labels={year_col: 'Anno', 'mean_score': 'Score Medio', 'count': 'N° Titoli'})
            fig_year.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_year, use_container_width=True)
