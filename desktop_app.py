import sys
import os
import shutil
import time
import tempfile
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QTextBrowser,
    QGroupBox,
    QMessageBox,
)

# --------------------------------------------------------------------------------------
# Import your real pipeline; if missing, a stub is used for quick testing.
# --------------------------------------------------------------------------------------
try:
    from workflow_executer import execute_workflow as _execute_workflow
except Exception:
    def _execute_workflow(video_path: str, report_cb):
        total_steps = 3
        report_cb(step=1, total=total_steps, message="Deciding pipeline…")
        time.sleep(0.4)
        decision = "ASR_OCR_VLM"

        report_cb(step=2, total=total_steps, message=f"Decided: {decision}")
        time.sleep(0.6)
        report_cb(step=2, total=total_steps, message="Generating chunking & extracting signals…")
        time.sleep(0.6)

        report_cb(step=3, total=total_steps, message="Summarizing…")
        time.sleep(0.6)
        report_cb(done=True, message="Completed")

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


# --------------------------------------------------------------------------------------
# Worker thread that copies the selected file to a temp path (to mirror Streamlit
# uploader semantics), runs the pipeline, reports progress/logs, and cleans up.
# --------------------------------------------------------------------------------------
class WorkflowWorker(QObject):
    progress = pyqtSignal(int, int, str)   # step, total, message
    log = pyqtSignal(str)
    finished = pyqtSignal(str)            # summary_md
    error = pyqtSignal(str)

    def __init__(self, source_path: str):
        super().__init__()
        self.source_path = source_path
        self._temp_path: Optional[str] = None

    def _report_cb(self, step: Optional[int] = None, total: Optional[int] = None, message: str = "", done: bool = False):
        if message:
            self.log.emit(message)
        if done:
            # Force progress to 100 when done is signalled by the pipeline
            self.progress.emit(total or 1, total or 1, message or "Completed")
            return
        if total:
            safe_step = 0 if step is None else step
            self.progress.emit(safe_step, total, message or "")

    def run(self):
        try:
            # Create a temp copy (mirrors the original Streamlit upload->temp file design)
            suffix = Path(self.source_path).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                self._temp_path = tmp.name
            self.log.emit("Copying video to a temporary file…")
            shutil.copy2(self.source_path, self._temp_path)

            # Execute the workflow
            summary_md = _execute_workflow(video_path=self._temp_path, report_cb=self._report_cb)
            self.finished.emit(summary_md)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Clean up the temp file
            if self._temp_path and os.path.exists(self._temp_path):
                try:
                    os.remove(self._temp_path)
                except Exception:
                    pass


# --------------------------------------------------------------------------------------
# Main Window UI
# --------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 AI-Driven Video Understanding")
        self.resize(900, 700)

        self._thread: Optional[QThread] = None
        self._worker: Optional[WorkflowWorker] = None
        self._logs = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        # File picker row
        picker_box = QGroupBox("Select a video file")
        picker_layout = QHBoxLayout(picker_box)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose a video (mp4, mkv, mov, avi, webm, m4v)…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.on_browse)
        picker_layout.addWidget(self.path_edit)
        picker_layout.addWidget(browse_btn)
        root.addWidget(picker_box)

        # Controls (Describe)
        controls_row = QHBoxLayout()
        self.describe_btn = QPushButton("Describe")
        self.describe_btn.setEnabled(False)
        self.describe_btn.clicked.connect(self.on_describe)
        controls_row.addStretch(1)
        controls_row.addWidget(self.describe_btn)
        root.addLayout(controls_row)

        # Status + Progress
        status_box = QGroupBox("Status")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("Idle. Select a video file, then click Describe.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        root.addWidget(status_box)

        # Logs
        logs_box = QGroupBox("Logs (last 10)")
        logs_layout = QVBoxLayout(logs_box)
        self.logs_view = QTextEdit()
        self.logs_view.setReadOnly(True)
        logs_layout.addWidget(self.logs_view)
        root.addWidget(logs_box)

        # Markdown summary (rendered as HTML)
        summary_box = QGroupBox("Description")
        summary_layout = QVBoxLayout(summary_box)
        self.summary_view = QTextBrowser()
        self.summary_view.setOpenExternalLinks(True)
        summary_layout.addWidget(self.summary_view)
        root.addWidget(summary_box, stretch=1)

        self.path_edit.textChanged.connect(self._on_path_changed)

    # ----------------------- UI Handlers -----------------------
    def on_browse(self):
        filters = "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", str(Path.home()), filters)
        if path:
            self.path_edit.setText(path)

    def _on_path_changed(self, _: str):
        p = Path(self.path_edit.text().strip())
        self.describe_btn.setEnabled(p.exists() and p.is_file())

    def on_describe(self):
        source_path = self.path_edit.text().strip()
        if not source_path:
            QMessageBox.warning(self, "No file", "Please choose a video file first.")
            return

        # Reset UI
        self._logs.clear()
        self.logs_view.clear()
        self.summary_view.clear()
        self.status_label.setText("Starting…")
        self.progress_bar.setValue(0)
        self.describe_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Start worker thread
        self._thread = QThread()
        self._worker = WorkflowWorker(source_path)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_progress(self, step: int, total: int, message: str):
        pct = max(0, min(100, int(step / total * 100))) if total else 0
        self.progress_bar.setValue(pct)
        if message:
            self.status_label.setText(message)

    def _on_log(self, message: str):
        if not message:
            return
        self._logs.append(f"➡️ {message}")
        self._logs = self._logs[-10:]
        self.logs_view.setPlainText("\n".join(self._logs))
        self.logs_view.verticalScrollBar().setValue(self.logs_view.verticalScrollBar().maximum())

    def _on_finished(self, summary_md: str):
        self.progress_bar.setValue(100)
        self.status_label.setText("Completed")
        self._render_markdown(summary_md)
        self.describe_btn.setEnabled(True)
        QApplication.restoreOverrideCursor()

    def _on_error(self, err: str):
        QApplication.restoreOverrideCursor()
        self.describe_btn.setEnabled(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred:\n\n{err}")

    def _cleanup_thread(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    # ----------------------- Helpers -----------------------
    def _render_markdown(self, md_text: str):
        try:
            import markdown  # type: ignore
            html = markdown.markdown(
                md_text or "",
                extensions=["fenced_code", "tables", "sane_lists"],
                output_format="html5",
            )
            self.summary_view.setHtml(html)
        except Exception:
            self.summary_view.setPlainText(md_text or "")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
