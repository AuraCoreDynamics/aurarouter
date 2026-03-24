"""First-launch onboarding wizard for AuraRouter.

Displays a short, visual walkthrough of core concepts and optionally
detects a running Ollama server for one-click initial configuration.
The wizard writes a flag file so it only appears once.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

_FLAG_PATH = Path.home() / ".auracore" / "aurarouter" / "onboarding_complete"


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def needs_onboarding() -> bool:
    """Return ``True`` if the onboarding wizard has not been completed."""
    return not _FLAG_PATH.exists()


def mark_onboarding_complete() -> None:
    """Write the flag file so the wizard does not appear again."""
    _FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FLAG_PATH.write_text("done", encoding="utf-8")


# ------------------------------------------------------------------
# Page content (HTML for QTextBrowser)
# ------------------------------------------------------------------

_PAGE_WELCOME = """\
<h2 style="text-align:center;">Welcome to AuraRouter</h2>
<p style="text-align:center; font-size:14px;">
AuraRouter is an intelligent prompt router that sends each task to the
best available model &mdash; local or cloud &mdash; with automatic
fallback.
</p>
<hr>
<p style="text-align:center;">
<b>How it works in 30 seconds:</b>
</p>
<pre style="font-size:13px; line-height:1.6;">
   You type a task
        |
   [ Router ] classifies it
        |
   Simple? -----> [ Coder ] executes directly
        |
   Complex? ----> [ Planner ] breaks it into steps
                      |
                  [ Coder ] executes each step
                      |
                  [ Reviewer ] checks the result
</pre>
<p style="text-align:center;">Each box above is a <b>role</b> that you
map to one or more models.</p>
"""

_PAGE_ROLES = """\
<h2>Roles: Who Does What</h2>
<p>Think of roles as job titles.  You assign a model (or a chain of
models) to each role.</p>

<table border="1" cellpadding="6" cellspacing="0" width="100%">
<tr style="background:#e0e0e0;">
    <th>Role</th><th>Job</th><th>Example Model</th></tr>
<tr><td><b>Router</b></td>
    <td>Reads your prompt and decides how complex it is</td>
    <td>Small local model (fast, cheap)</td></tr>
<tr><td><b>Reasoning</b></td>
    <td>Creates a step-by-step plan for complex tasks</td>
    <td>Medium model with good logic</td></tr>
<tr><td><b>Coding</b></td>
    <td>Writes the actual output (code, text, analysis)</td>
    <td>Best model you have access to</td></tr>
<tr><td><b>Reviewer</b></td>
    <td>Checks the output and asks for corrections</td>
    <td>Optional; any capable model</td></tr>
</table>

<p>You only <i>need</i> to configure <b>router</b>, <b>reasoning</b>,
and <b>coding</b>.  A single model can fill all three roles to start
with &mdash; you can specialize later.</p>
"""

_PAGE_FALLBACK = """\
<h2>Fallback Chains</h2>
<p>Each role has an ordered list of models.  AuraRouter tries the first
model; if it fails, it falls back to the next.</p>

<pre style="font-size:13px; line-height:1.8;">
  coding:
    1. local_qwen        (try first &mdash; free, private)
           |
         fail?
           |
    2. cloud_claude      (fallback &mdash; powerful, costs $)
</pre>

<p><b>Why this matters:</b></p>
<ul>
  <li>Sensitive data stays local whenever possible.</li>
  <li>Complex tasks still succeed via cloud fallback.</li>
  <li>You control cost by choosing which models appear in each chain
      and in what order.</li>
</ul>

<p>You manage chains in the <b>Configuration</b> tab &rarr;
<i>Routing</i> section.</p>
"""

_PAGE_SETUP = """\
<h2>Quick Setup</h2>
{detection_block}
<h3>Minimal Config</h3>
<p>You need at least one model assigned to the three required roles
(<b>router</b>, <b>reasoning</b>, <b>coding</b>).</p>

<p>The simplest starting point:</p>
<pre style="font-size:12px;">
models:
  local_qwen:
    provider: ollama
    endpoint: http://localhost:11434/api/generate
    model_name: qwen2.5-coder:7b

roles:
  router:    [local_qwen]
  reasoning: [local_qwen]
  coding:    [local_qwen]
</pre>

<p>You can add cloud fallbacks, more roles, and budget limits later.</p>
"""

_PAGE_READY = """\
<h2 style="text-align:center;">You're Ready!</h2>
<p style="text-align:center; font-size:14px;">
Head to the <b>Execute</b> tab, type a task, and press
<b>Ctrl+Enter</b>.
</p>
<br>
<p style="text-align:center;">Quick tips:</p>
<ul>
  <li><b>Ctrl+Enter</b> &mdash; run the task</li>
  <li><b>Ctrl+N</b> &mdash; clear everything for a new prompt</li>
  <li><b>Escape</b> &mdash; cancel a running task</li>
</ul>
<br>
<p style="text-align:center;">
Need help later?  Open the <b>Help</b> panel from the toolbar
(or press <b>F1</b>) to search all topics.</p>
"""


# ------------------------------------------------------------------
# Ollama detection
# ------------------------------------------------------------------

def _detect_ollama() -> bool:
    """Return ``True`` if an Ollama server appears to be reachable."""
    try:
        # Try the Ollama HTTP API (fast, no import needed).
        import httpx  # noqa: F811
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        pass
    # Fallback: check if the CLI exists.
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


# ------------------------------------------------------------------
# Wizard dialog
# ------------------------------------------------------------------

class OnboardingWizard(QDialog):
    """Multi-page onboarding dialog shown on first launch."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to AuraRouter")
        self.setMinimumSize(620, 480)
        self.setModal(True)

        self._pages: list[str] = []
        self._build_ui()

    # ----------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Stacked pages
        self._stack = QStackedWidget()
        self._pages = [
            _PAGE_WELCOME,
            _PAGE_ROLES,
            _PAGE_FALLBACK,
            self._build_setup_page_html(),
            _PAGE_READY,
        ]
        for html in self._pages:
            browser = QTextBrowser()
            browser.setOpenExternalLinks(True)
            browser.setHtml(html)
            self._stack.addWidget(browser)

        root.addWidget(self._stack, 1)

        # Page indicator
        self._page_label = QLabel()
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._page_label)

        # Navigation buttons
        nav = QHBoxLayout()

        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)

        nav.addStretch()

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self._on_skip)
        nav.addWidget(self._skip_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        root.addLayout(nav)

        self._update_buttons()

    def _build_setup_page_html(self) -> str:
        """Build the setup page, injecting Ollama detection results."""
        ollama_found = _detect_ollama()
        if ollama_found:
            block = (
                '<p style="color:green; font-weight:bold;">'
                "&#10004; Ollama detected on localhost:11434.</p>"
                "<p>You can use the Configuration tab to add an Ollama "
                "model right away.</p>"
            )
        else:
            block = (
                '<p style="color:#b26a00; font-weight:bold;">'
                "&#9888; Ollama not detected.</p>"
                "<p>Install Ollama from <code>https://ollama.com</code>, "
                "then run <code>ollama pull qwen2.5-coder:7b</code> to "
                "download a starter model.</p>"
            )
        return _PAGE_SETUP.format(detection_block=block)

    # ----------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------

    def _update_buttons(self) -> None:
        idx = self._stack.currentIndex()
        total = self._stack.count()
        self._back_btn.setEnabled(idx > 0)
        is_last = idx == total - 1
        self._next_btn.setText("Finish" if is_last else "Next")
        self._page_label.setText(f"Page {idx + 1} of {total}")

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx < self._stack.count() - 1:
            self._stack.setCurrentIndex(idx + 1)
            self._update_buttons()
        else:
            self._finish()

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_buttons()

    def _on_skip(self) -> None:
        self._finish()

    def _finish(self) -> None:
        mark_onboarding_complete()
        self.accept()
