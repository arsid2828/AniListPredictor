import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
ROOT_DIR = APP_DIR.parent
for path in (str(ROOT_DIR), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from shared import (
    ADMIN_EMAIL,
    delete_artifact_by_type,
    delete_local_user_artifacts,
    get_all_runtime_usernames,
    get_profile_owner_registry,
    get_user_artifact_summary,
    init_session_state,
    inject_css,
    register_profile_for_viewer,
    render_app_disclaimer,
    render_user_badge,
    require_admin_user,
    unregister_profile_for_all_viewers,
    unregister_profile_for_viewer,
)


def _viewer_registry_rows():
    rows = []
    registry = get_profile_owner_registry()
    for viewer_key, profiles in sorted(registry.items()):
        viewer_label = viewer_key if "@" in viewer_key else f"legacy:{viewer_key[:12]}..."
        for profile in sorted({str(profile).strip().lower() for profile in profiles if str(profile).strip()}):
            rows.append({"viewer": viewer_label, "profile": profile})
    return pd.DataFrame(rows)


def _artifact_preview(path_str: str):
    path = Path(path_str)
    if not path.exists():
        st.info("File not found.")
        return

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            st.caption(f"JSON list with {len(data)} items")
            st.json(data[:3] if data else [])
        else:
            st.json(data)
        return

    if suffix == ".csv":
        df = pd.read_csv(path)
        st.caption(f"CSV rows: {len(df)}")
        st.dataframe(df.head(50), width="stretch")
        return

    if suffix == ".pkl":
        data = joblib.load(path)
        if isinstance(data, dict):
            preview = {}
            for key, value in data.items():
                if key == "model":
                    preview[key] = type(value).__name__
                elif key == "train_columns" and isinstance(value, list):
                    preview[key] = {"count": len(value), "sample": value[:12]}
                elif key == "metrics" and isinstance(value, list):
                    preview[key] = value[:5]
                elif key == "feature_importance" and isinstance(value, list):
                    preview[key] = value[:10]
                elif key == "shap_explanations" and isinstance(value, list):
                    preview[key] = value[:10]
                else:
                    preview[key] = value
            st.json(preview)
        else:
            st.write(type(data).__name__)
        return

    st.code(path.read_text(encoding="utf-8", errors="replace")[:4000])


st.set_page_config(page_title="Admin", layout="wide", page_icon="🛠️")
inject_css()
init_session_state()
require_admin_user()
render_user_badge()

st.title("🛠️ Admin")
st.caption(f"Owner-only runtime data console for `{ADMIN_EMAIL}`.")

with st.expander("Saved Profiles Registry", expanded=True):
    registry_df = _viewer_registry_rows()
    if registry_df.empty:
        st.info("No saved-profile assignments found yet.")
    else:
        st.dataframe(registry_df, width="stretch", hide_index=True)

    col_add, col_remove = st.columns(2)
    with col_add:
        with st.form("admin_add_profile_assignment"):
            st.subheader("Add Saved Profile")
            viewer_email = st.text_input("Viewer email", placeholder="user@example.com")
            profile_username = st.text_input("AniList username", placeholder="anilist_user")
            add_submit = st.form_submit_button("Add assignment", width="stretch")
            if add_submit:
                if register_profile_for_viewer(viewer_email, profile_username):
                    st.success("Saved profile assignment added.")
                    st.rerun()
                else:
                    st.error("Enter both a valid viewer email and username.")

    with col_remove:
        with st.form("admin_remove_profile_assignment"):
            st.subheader("Remove Saved Profile")
            viewer_email_remove = st.text_input("Viewer email ", placeholder="user@example.com")
            profile_username_remove = st.text_input("AniList username ", placeholder="anilist_user")
            remove_submit = st.form_submit_button("Remove assignment", width="stretch")
            if remove_submit:
                if unregister_profile_for_viewer(viewer_email_remove, profile_username_remove):
                    st.success("Saved profile assignment removed.")
                    st.rerun()
                else:
                    st.error("Assignment not found.")

runtime_usernames = get_all_runtime_usernames()
st.markdown("---")
st.subheader("Runtime User Inventory")

if not runtime_usernames:
    st.info("No runtime user data found.")
    render_app_disclaimer()
    st.stop()

selected_username = st.selectbox("Select AniList username", runtime_usernames, index=0)
artifact_rows = pd.DataFrame(get_user_artifact_summary(selected_username))
viewer_registry = get_profile_owner_registry()
assigned_viewers = sorted(
    viewer for viewer, profiles in viewer_registry.items() if selected_username in {str(item).strip().lower() for item in profiles}
)

col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.metric("Artifacts found", int(artifact_rows["exists"].sum()))
with col_meta2:
    st.metric("Assigned saved profiles", len(assigned_viewers))
with col_meta3:
    total_size = int(artifact_rows.loc[artifact_rows["exists"], "size_bytes"].sum())
    st.metric("Stored size", f"{total_size / 1024:.1f} KB")

st.caption("Assigned viewers: " + (", ".join(assigned_viewers) if assigned_viewers else "none"))
st.dataframe(artifact_rows, width="stretch", hide_index=True)

danger1, danger2 = st.columns(2)
with danger1:
    if st.button("Delete all artifacts for this username", type="primary", width="stretch"):
        delete_local_user_artifacts(selected_username)
        st.success(f"Deleted all runtime artifacts for {selected_username}.")
        st.rerun()
with danger2:
    if st.button("Remove this username from all saved-profile lists", width="stretch"):
        if unregister_profile_for_all_viewers(selected_username):
            st.success(f"Removed {selected_username} from all saved-profile registries.")
            st.rerun()
        else:
            st.info("That username was not assigned in the saved-profile registry.")

st.markdown("#### Artifact Explorer")
for row in artifact_rows.to_dict("records"):
    label = row["artifact_type"].replace("_", " ").title()
    status = "present" if row["exists"] else "missing"
    with st.expander(f"{label} — {status}", expanded=False):
        st.code(row["path"])
        if row["exists"]:
            st.caption(f"Modified: {row['modified_at']} | Size: {row['size_bytes']} bytes")
            _artifact_preview(row["path"])
            delete_key = f"delete_{selected_username}_{row['artifact_type']}"
            if st.button(f"Delete {label}", key=delete_key, width="stretch"):
                if delete_artifact_by_type(selected_username, row["artifact_type"]):
                    st.success(f"{label} deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete this artifact.")
        else:
            st.info("No file currently stored for this artifact.")

render_app_disclaimer()
