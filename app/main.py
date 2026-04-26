from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent


pages = {
    "Main": [
        st.Page(str(APP_DIR / "home_dashboard.py"), title="Prediction Studio", icon="🏠", default=True),
    ],
    "Insights": [
        st.Page(str(APP_DIR / "pages" / "1_Profile_Analysis.py"), title="Profile Analysis", icon="📊"),
        st.Page(str(APP_DIR / "pages" / "2_Compare_Titles.py"), title="Compare Titles", icon="⚖️"),
        st.Page(str(APP_DIR / "pages" / "3_User_Compatibility.py"), title="User Compatibility", icon="🤝"),
        st.Page(str(APP_DIR / "pages" / "4_Activity_History.py"), title="Activity History", icon="📈"),
    ],
    "Preference Tools": [
        st.Page(str(APP_DIR / "pages" / "7_Preference_Arena.py"), title="Preference Arena", icon="🏟️"),
        st.Page(str(APP_DIR / "pages" / "8_Score_Positioning.py"), title="Score Positioning", icon="🎯"),
    ],
    "Legal": [
        st.Page(str(APP_DIR / "pages" / "5_Privacy_Policy.py"), title="Privacy Policy", icon="🔒"),
        st.Page(str(APP_DIR / "pages" / "6_Data_Source_And_Legal.py"), title="Data Source & Legal", icon="📘"),
    ],
}


navigation = st.navigation(pages, position="sidebar")
navigation.run()
