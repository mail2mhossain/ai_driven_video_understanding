# app.py
# Minimal Streamlit UI with spinner, status, progress bar, and logs.
# Select a video -> Describe -> 3-step progress -> Markdown summary.

import time
import streamlit as st
# If you need a temp file path from the upload, uncomment:
import tempfile
from pathlib import Path
from workflow_executer import execute_workflow

st.set_page_config(page_title="Video Understanding", page_icon="🎬", layout="centered")
st.title("🎬 AI-Driven Video Understanding")

def save_uploaded_to_temp(uploaded_file) -> str:
    # keep original extension if possible
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    # delete=False so we can reopen it on Windows
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

uploaded_video = st.file_uploader(
    "Select a video file",
    type=["mp4", "mkv", "mov", "avi", "webm", "m4v"],
)

if st.button("Describe", type="primary", disabled=uploaded_video is None):
    with st.spinner("Describing..."):
        status = st.status("Starting…", expanded=True)
        progress = st.progress(0)
        log_area = st.empty()
        logs = []

        def ui_report(step=None, total=None, message="", done=False):
            if done:
                status.update(label="Completed", state="complete", expanded=False)
                progress.progress(100)
                return
            if step is not None and total:
                pct = max(0, min(100, int(step / total * 100)))
                progress.progress(pct)
                status.update(label=message, state="running", expanded=True)
            logs.append(f"➡️ {message}")
            log_area.write("\n".join(logs[-10:]))

        # ---- get a filesystem path for the uploaded file ----
        temp_path = save_uploaded_to_temp(uploaded_video)
        try:
            # pass a Path or str, whichever your function expects
            summary_md = execute_workflow(video_path=temp_path, report_cb=ui_report)
        finally:
            # clean up the temp file after processing
            import os
            os.remove(temp_path)

    st.markdown("## Description")
    st.markdown(summary_md)

    file_stem = Path(uploaded_video.name).stem if uploaded_video and uploaded_video.name else "summary"
    st.download_button(
        label="⬇️ Export summary as Markdown",
        data=summary_md.encode("utf-8"),
        file_name=f"{file_stem}.md",
        mime="text/markdown",
    )

else:
    st.caption("Select a video file, then click **Describe**.")
