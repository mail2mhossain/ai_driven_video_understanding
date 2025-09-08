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

def run_pipeline_stub(report_cb):
    """Replace this stub with your real probe/ASR/OCR/VLM/summarization pipeline.
       Call report_cb(step=..., total=..., message=...) along the way.
       Finally, return markdown text.
    """
    total_steps = 3

    # Step 1 — Probe & decide
    report_cb(step=1, total=total_steps, message="Deciding pipeline…")
    time.sleep(0.4)
    decision = "ASR_OCR_VLM"

    # Step 2 — Chunking & extraction
    report_cb(step=2, total=total_steps, message=f"Decided: {decision}")
    time.sleep(0.6)
    report_cb(step=2, total=total_steps, message="Generating chunking & extracting signals…")
    time.sleep(0.6)

    # Step 3 — Summarization
    report_cb(step=3, total=total_steps, message="Summarizing…")
    time.sleep(0.6)

    summary_md = (
        "### Global Summary\n"
        "- The video explains a workflow using narration, on-screen text, and visuals.\n"
        "- Key steps are demonstrated in short segments.\n\n"
        "### Timeline\n"
        "[00:00–00:30]: Introduction and context.\n\n"
        "[00:30–01:00]: Method steps and results.\n\n"
        "### Entities\n"
        "- Presenter — demonstrates and explains.\n\n"
        "### Notable Numbers & Quotes\n"
        "None.\n\n"
        "### Uncertainties\n"
        "None.\n"
    )
    return summary_md

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

else:
    st.caption("Select a video file, then click **Describe**.")
