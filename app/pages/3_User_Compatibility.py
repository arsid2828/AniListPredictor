import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.dataset import build_user_dataframe, build_user_manga_dataframe
from src.analytics import compute_compatibility
from app.shared import inject_css, init_session_state, render_user_badge
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="User Compatibility", layout="wide", page_icon="👥")
inject_css()
init_session_state()
render_user_badge()

st.title("👥 Compatibility Between Two Users")
st.write("Compare the tastes of two AniList users based on commonly rated titles.")

media_type = st.radio("Tipo:", ["🎬 Anime", "📖 Manga"], horizontal=True)
is_manga = "Manga" in media_type
media_label = "Manga" if is_manga else "Anime"

col1, col2 = st.columns(2)
with col1:
    user1 = st.text_input("👤 Username 1:", value=st.session_state.get("username", ""), key="compat_u1")
with col2:
    user2 = st.text_input("👤 Username 2:", key="compat_u2")

if user1 and user2 and st.button("🔍 Calculate Compatibility", use_container_width=True):
    if user1.lower() == user2.lower():
        st.warning("You must enter two different usernames!")
        st.stop()
    
    with st.spinner(f"Scarico i dati {media_label.lower()} di entrambi gli utenti..."):
        if is_manga:
            df1 = build_user_manga_dataframe(user1, force_refresh=False, save_csv=False)
            df2 = build_user_manga_dataframe(user2, force_refresh=False, save_csv=False)
        else:
            df1 = build_user_dataframe(user1, force_refresh=False, save_csv=False)
            df2 = build_user_dataframe(user2, force_refresh=False, save_csv=False)
    
    if df1 is None or df1.empty:
        st.error(f"No data found for {user1}. Private or non-existent profile?")
        st.stop()
    if df2 is None or df2.empty:
        st.error(f"No data found for {user2}. Private or non-existent profile?")
        st.stop()
    
    result = compute_compatibility(df1, df2, user1, user2)
    
    if result['status'] == 'insufficient':
        st.warning(f"Only {result['common_count']} {media_label.lower()} in common — at least 3 are needed for meaningful analysis.")
        st.stop()
    
    # ========== RESULTS ==========
    st.markdown("---")
    
    # Compatibility percentage with gauge
    compat = result['compatibility_pct']
    
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
        compat_text = "Gusti Diversi"
    
    gauge_col, stats_col = st.columns([2, 3])
    
    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=compat,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{compat_emoji} Compatibilità", 'font': {'size': 20}},
            number={'suffix': '%', 'font': {'size': 50}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': compat_color},
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(235, 51, 67, 0.1)'},
                    {'range': [33, 66], 'color': 'rgba(242, 153, 74, 0.1)'},
                    {'range': [66, 100], 'color': 'rgba(0, 176, 155, 0.1)'}
                ]
            }
        ))
        fig_gauge.update_layout(height=300, template='plotly_dark')
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with stats_col:
        st.markdown(f"### {compat_text}")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(f"{media_label} in Comune", result['common_count'])
        with m2:
            st.metric("Correlazione", f"{result['correlation']:.2f}")
        with m3:
            st.metric("Differenza Media", f"{result['mae']:.2f}")
        with m4:
            st.metric("Medie", f"{result['mean_1']:.1f} vs {result['mean_2']:.1f}")
    
    st.markdown("---")
    
    common_df = result['common_df']
    
    # Scatter plot: User1 vs User2 scores
    fig_scatter = px.scatter(
        common_df, x='user_score_1', y='user_score_2',
        hover_data=['title'],
        labels={'user_score_1': f'Voto {user1}', 'user_score_2': f'Voto {user2}'},
        title=f"Confronto Voti — {user1} vs {user2}",
        color_discrete_sequence=['#667eea']
    )
    # Perfect agreement line
    fig_scatter.add_trace(go.Scatter(
        x=[0, 10], y=[0, 10], mode='lines',
        line=dict(dash='dash', color='rgba(255,255,255,0.3)'),
        name='Accordo Perfetto', showlegend=True
    ))
    fig_scatter.update_layout(template='plotly_dark', height=450,
                             xaxis=dict(range=[0, 10.5]), yaxis=dict(range=[0, 10.5]))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Biggest disagreements
    common_df['diff'] = (common_df['user_score_1'] - common_df['user_score_2']).abs()
    common_df = common_df.sort_values('diff', ascending=False)
    
    st.subheader("🔥 The Biggest Disagreements")
    disagree = common_df.head(10)
    for _, row in disagree.iterrows():
        title = row.get('title', row.get('title_1', 'N/A'))
        d = row['diff']
        s1, s2 = row['user_score_1'], row['user_score_2']
        bar = "🟢" if d < 1 else ("🟡" if d < 2 else "🔴")
        st.write(f"{bar} **{title}** — {user1}: {s1:.1f} vs {user2}: {s2:.1f} (Δ {d:.1f})")
    
    st.subheader("💚 Most Agreed Upon")
    agree = common_df.tail(10).iloc[::-1]
    for _, row in agree.iterrows():
        title = row.get('title', row.get('title_1', 'N/A'))
        d = row['diff']
        s1, s2 = row['user_score_1'], row['user_score_2']
        st.write(f"🤝 **{title}** — {user1}: {s1:.1f} vs {user2}: {s2:.1f} (Δ {d:.1f})")
