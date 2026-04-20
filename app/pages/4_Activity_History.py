import streamlit as st
import pandas as pd
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.api import CACHE_DIR, fetch_user_activity_history
from src.analytics import compute_activity_stats
from app.shared import inject_css, init_session_state, render_user_badge
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Activity History", layout="wide", page_icon="📅")
inject_css()
init_session_state()
render_user_badge()

st.title("📅 Analytical Activity History")
st.markdown("Pure statistics based on your real AniList history (episodes watched and chapters read). Automatically excludes *Plan to Watch* activities etc.")

username = st.session_state.get("username", "")
if not username:
    st.info("👈 Return to the main page and enter your username to start.")
    st.stop()

# Media toggle
media_type_raw = st.radio("Select Media:", ["🎬 Anime", "📖 Manga"], horizontal=True)
media_type = "MANGA" if "Manga" in media_type_raw else "ANIME"
media_label = "Manga" if media_type == "MANGA" else "Anime"
unit_label = "Capitoli" if media_type == "MANGA" else "Episodes"

history_path = CACHE_DIR / f"user_activity_{media_type.lower()}_{username.lower()}.json"

if st.button(f"⬇️ Scarica / Aggiorna History {media_label} (richiede tempo per utenti attivi)"):
    with st.spinner(f"Scaricamento history {media_label} da AniList in corso... Attendere."):
        acts = fetch_user_activity_history(username, media_type=media_type, force_refresh=True)
        if acts:
            st.success(f"{media_label} History downloaded successfully! Reloading...")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("No activity found or private profile.")

if history_path.exists():
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            acts = json.load(f)
            
        stats = compute_activity_stats(acts, media_type=media_type)
        if stats:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"🏆 Giorno Record (più {unit_label.lower()})", 
                          f"{stats['max_day']['date']}", 
                          f"{stats['max_day']['episodes']} {unit_label[:2].lower()}. ({stats['max_day']['hours']:.1f} ore stimate)")
                
            with st.expander(f"📚 Cosa hai esplorato il {stats['max_day']['date']}?"):
                for title in stats['max_day']['titles']:
                    st.write(f"- {title}")
            
            st.markdown(f"#### {unit_label} per Month")
            if media_type == "ANIME":
                fig_month = go.Figure(data=[
                    go.Bar(name=unit_label, x=stats['month_stats']['month'], y=stats['month_stats']['episodes']),
                    go.Scatter(name='Ore', x=stats['month_stats']['month'], y=stats['month_stats']['duration_hours'], yaxis='y2', mode='lines+markers', line=dict(color='#eb3349'))
                ])
                fig_month.update_layout(
                    template='plotly_dark',
                    yaxis=dict(title=f'{unit_label}', side='left'),
                    yaxis2=dict(title='Ore', overlaying='y', side='right'),
                    barmode='group',
                    legend=dict(x=0, y=1.2, orientation='h'),
                    height=400
                )
            else:
                # Per i manga mostriamo solo i capitoli visto che le ore sono ipotetiche (ma manteniamo il plot in bar semplice)
                fig_month = px.bar(stats['month_stats'], x='month', y='episodes', title=f"{unit_label} Letti per Mese",
                                   labels={'episodes': unit_label, 'month': 'Mese'}, template='plotly_dark')
                fig_month.update_layout(height=400)
            
            st.plotly_chart(fig_month, use_container_width=True)
            
            st.markdown(f"#### {unit_label} per Year")
            if media_type == "ANIME":
                fig_year = px.bar(stats['year_stats'], x='year', y=['episodes', 'duration_hours'], barmode='group', 
                                  labels={'value': 'Quantità', 'variable': 'Metrica', 'year': 'Anno'},
                                  template='plotly_dark', height=400)
            else:
                fig_year = px.bar(stats['year_stats'], x='year', y='episodes', 
                                  labels={'episodes': unit_label, 'year': 'Anno'}, text_auto=True,
                                  template='plotly_dark', height=400)
                
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info(f"Not enough {media_label.lower()} activity to generate analytical statistics.")
    except Exception as e:
        st.error(f"Error generating statistics: {e}")
else:
    st.info(f"Click the button to download the {media_label} history.")
