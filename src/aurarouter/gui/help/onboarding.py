"""Interactive setup wizard for AuraRouter.

Six-page wizard: Welcome, Local Models, On-Prem Providers, Cloud Providers,
Role Configuration, Ready.  Writes a flag file so it only appears once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QProgressBar, QPushButton, QScrollArea, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from aurarouter.gui.theme import DARK_PALETTE, SPACING, TYPOGRAPHY

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader

_FLAG_PATH = Path.home() / ".auracore" / "aurarouter" / "onboarding_complete"
_P, _S, _T = DARK_PALETTE, SPACING, TYPOGRAPHY

# === Public helpers (preserved API) ===

def needs_onboarding() -> bool:
    return not _FLAG_PATH.exists()

def mark_onboarding_complete() -> None:
    _FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FLAG_PATH.write_text("done", encoding="utf-8")

# === Shared state ===

PERSONA_SETTINGS: dict[str, dict] = {
    "performance": {
        "speculative_decoding_enabled": True,
        "monologue_enabled": False,
        "sovereignty_enforcement_enabled": False,
        "rag_enrichment_enabled": True,
        "sessions_enabled": True,
        "sessions_condensation_threshold": 0.80,
    },
    "privacy": {
        "speculative_decoding_enabled": False,
        "monologue_enabled": False,
        "sovereignty_enforcement_enabled": True,
        "rag_enrichment_enabled": False,
        "sessions_enabled": True,
        "sessions_condensation_threshold": 0.70,
    },
    "researcher": {
        "speculative_decoding_enabled": True,
        "monologue_enabled": True,
        "sovereignty_enforcement_enabled": False,
        "rag_enrichment_enabled": True,
        "sessions_enabled": True,
        "sessions_condensation_threshold": 0.90,
    },
}


def _apply_persona(state: "WizardState", persona: str) -> None:
    """Write persona-preset values into *state*."""
    settings = PERSONA_SETTINGS.get(persona)
    if settings is None:
        return
    state.persona = persona
    state.persona_settings = dict(settings)


@dataclass
class WizardState:
    hardware: object = None
    downloaded_models: list[dict] = field(default_factory=list)
    ollama_models: list[dict] = field(default_factory=list)
    custom_endpoints: list[dict] = field(default_factory=list)
    cloud_providers: list[dict] = field(default_factory=list)
    role_assignments: dict[str, list[str]] = field(default_factory=dict)
    persona: str | None = None
    persona_settings: dict = field(default_factory=dict)

    def all_model_ids(self) -> list[str]:
        ids = [m["model_id"] for m in self.downloaded_models]
        ids += [m["model_id"] for m in self.ollama_models]
        ids += [m["model_id"] for m in self.custom_endpoints]
        return ids

def _sanitize_id(prefix: str, name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", f"{prefix}_{name}")

# === Workers ===

class _HWWorker(QObject):
    finished = Signal(object); error = Signal(str)
    def run(self):
        try:
            from aurarouter.gui.help.setup_helpers import detect_hardware
            self.finished.emit(detect_hardware())
        except Exception as e: self.error.emit(str(e))

class _OllamaWorker(QObject):
    finished = Signal(dict); error = Signal(str)
    def run(self):
        try:
            from aurarouter.gui.help.setup_helpers import detect_ollama
            self.finished.emit(detect_ollama())
        except Exception as e: self.error.emit(str(e))

class _CloudWorker(QObject):
    finished = Signal(list); error = Signal(str)
    def run(self):
        try:
            from aurarouter.gui.help.setup_helpers import detect_cloud_providers
            self.finished.emit(detect_cloud_providers())
        except Exception as e: self.error.emit(str(e))

class _PipWorker(QObject):
    finished = Signal(bool, str); error = Signal(str)
    def __init__(self, pkg: str):
        super().__init__(); self.pkg = pkg
    def run(self):
        try:
            from aurarouter.gui.help.setup_helpers import pip_install_sync
            ok, msg = pip_install_sync(self.pkg); self.finished.emit(ok, msg)
        except Exception as e: self.error.emit(str(e))

class _TestWorker(QObject):
    finished = Signal(str); error = Signal(str)
    def __init__(self, cl: "ConfigLoader"):
        super().__init__(); self._cl = cl
    def run(self):
        try:
            from aurarouter.fabric import ComputeFabric
            chain = self._cl.get_role_chain("router")
            if not chain:
                self.finished.emit("No router role configured."); return
            r = ComputeFabric(self._cl).execute("router", "Say hello", json_mode=False)
            self.finished.emit(str(r)[:300])
        except Exception as e: self.error.emit(str(e))

# === Thread mixin ===

class _TM:
    def _init_threads(self):
        self._threads: list[QThread] = []; self._workers: list[QObject] = []
    def _start_worker(self, w, on_ok, on_err):
        t = QThread(); w.moveToThread(t)
        t.started.connect(w.run)
        w.finished.connect(on_ok); w.error.connect(on_err)
        w.finished.connect(t.quit); w.error.connect(t.quit)
        self._threads.append(t); self._workers.append(w); t.start()
    def _cleanup_threads(self):
        for t in self._threads:
            if t.isRunning(): t.quit(); t.wait(3000)
            t.deleteLater()
        for w in self._workers: w.deleteLater()
        self._threads.clear(); self._workers.clear()

# === Style helpers ===

def _stitle(t):
    l = QLabel(t); l.setStyleSheet(f"font-size:{_T.size_h2}px;font-weight:bold;color:{_P.accent};margin-top:{_S.md}px;margin-bottom:{_S.sm}px;"); return l

def _sdesc(t):
    l = QLabel(t); l.setWordWrap(True); l.setStyleSheet(f"color:{_P.text_secondary};"); return l

def _scroll(w):
    s = QScrollArea(); s.setWidgetResizable(True); s.setWidget(w); s.setStyleSheet("QScrollArea{border:none;}"); return s

# === Page 0: Persona Chooser ===

_PERSONA_CARDS = [
    ("performance", "\u26a1 Performance First",
     "Local-first speed with speculative decoding and RAG enrichment."),
    ("privacy",     "\U0001f512 Privacy First",
     "Sovereignty enforcement on by default. Cloud routing restricted."),
    ("researcher",  "\U0001f52c Researcher",
     "Full pipeline: monologue reasoning, speculative decoding, and RAG."),
]


class PersonaChooserPage(QWidget):
    """Page 0 &mdash; choose a persona preset or go advanced."""

    def __init__(self, state: "WizardState", parent=None):
        super().__init__(parent)
        self._state = state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(_S.xl, _S.xl, _S.xl, _S.xl)
        lay.setSpacing(_S.md)

        t = QLabel("Welcome to AuraRouter")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(
            f"font-size:{_T.size_h1}px;font-weight:bold;color:{_P.accent};"
        )
        lay.addWidget(t)

        sub = QLabel("Choose a configuration preset to get started quickly.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{_P.text_secondary};")
        lay.addWidget(sub)

        lay.addSpacing(_S.md)

        for persona_key, label, desc in _PERSONA_CARDS:
            card = self._make_card(persona_key, label, desc)
            lay.addWidget(card)

        lay.addSpacing(_S.sm)

        adv = QPushButton("\u2192 Advanced: Let me configure everything manually")
        adv.setFlat(True)
        adv.setStyleSheet(
            f"color:{_P.accent};text-decoration:underline;border:none;"
            f"font-size:{_T.size_body}px;"
        )
        adv.clicked.connect(lambda: _apply_persona(self._state, ""))
        lay.addWidget(adv, alignment=Qt.AlignmentFlag.AlignCenter)

        lay.addStretch()

        skip_lbl = QLabel("Skip \u2014 I'll configure in Settings later")
        skip_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        skip_lbl.setStyleSheet(f"color:{_P.text_disabled};font-size:{_T.size_small}px;")
        lay.addWidget(skip_lbl)

    def _make_card(self, persona_key: str, label: str, desc: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            f"QWidget{{background:{_P.bg_secondary};border:1px solid {_P.border};"
            f"border-radius:8px;padding:{_S.md}px;}}"
            f"QWidget:hover{{border:1px solid {_P.accent};}}"
        )
        vl = QVBoxLayout(card)
        vl.setSpacing(_S.sm)

        title_row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"font-size:{_T.size_h2}px;font-weight:bold;border:none;"
        )
        title_row.addWidget(lbl)
        title_row.addStretch()

        select_btn = QPushButton("Select")
        select_btn.setFixedWidth(80)
        select_btn.clicked.connect(
            lambda _=False, k=persona_key: _apply_persona(self._state, k)
        )
        title_row.addWidget(select_btn)
        vl.addLayout(title_row)

        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet(f"color:{_P.text_secondary};border:none;")
        vl.addWidget(d)

        return card

# === Page 1: Local Models ===

class LocalModelsPage(QWidget, _TM):
    def __init__(self, state: WizardState, parent=None):
        super().__init__(parent); self._state = state; self._init_threads()
        self._hw_done = False; self._recs: list[dict] = []; self._sidecar_pkg: str | None = None
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(_S.lg, _S.lg, _S.lg, _S.lg); lay.setSpacing(_S.md)
        # Section A
        lay.addWidget(_stitle("Hardware Detection"))
        self._hw_lbl = QLabel("Detecting hardware..."); self._hw_lbl.setStyleSheet(f"color:{_P.text_secondary};")
        lay.addWidget(self._hw_lbl)
        self._hw_bar = QProgressBar(); self._hw_bar.setRange(0,0); self._hw_bar.setMaximumHeight(8); lay.addWidget(self._hw_bar)
        # Section B
        lay.addWidget(_stitle("Download a Model"))
        self._tbl = QTableWidget(0,4); self._tbl.setHorizontalHeaderLabels(["Model","Size","Reason",""])
        h = self._tbl.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch); h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in (1,3): h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self._tbl.setMinimumHeight(150)
        lay.addWidget(self._tbl)
        bb = QPushButton("Browse HuggingFace..."); bb.setFixedWidth(180); bb.clicked.connect(self._browse_hf); lay.addWidget(bb)
        # Section C
        self._gpu_w = QWidget(); gl = QVBoxLayout(self._gpu_w); gl.setContentsMargins(0,0,0,0)
        gl.addWidget(_stitle("GPU Acceleration"))
        self._gpu_lbl = _sdesc("NVIDIA GPU detected but no acceleration sidecar installed."); gl.addWidget(self._gpu_lbl)
        self._gpu_btn = QPushButton("Install GPU Acceleration"); self._gpu_btn.setFixedWidth(220)
        self._gpu_btn.clicked.connect(self._install_sidecar); gl.addWidget(self._gpu_btn)
        self._gpu_st = QLabel(""); gl.addWidget(self._gpu_st)
        self._gpu_w.setVisible(False); lay.addWidget(self._gpu_w)
        lay.addStretch(); outer.addWidget(_scroll(inner))

    def showEvent(self, e):
        super().showEvent(e)
        if not self._hw_done: self._start_worker(_HWWorker(), self._on_hw, self._on_hw_err)

    def _on_hw(self, hw):
        self._hw_done = True; self._state.hardware = hw; self._hw_bar.setVisible(False)
        gpu = getattr(hw, "gpu_name", None) or "No GPU"; vram = getattr(hw, "vram_mb", 0) or 0
        cores = getattr(hw, "cpu_cores", 0) or 0; nv = getattr(hw, "has_nvidia", False)
        if vram:
            self._hw_lbl.setText(f"GPU: {gpu} ({vram} MB VRAM) | CPU cores: {cores}")
            self._hw_lbl.setStyleSheet(f"color:{_P.success};")
        else:
            self._hw_lbl.setText(f"No GPU detected -- will use CPU ({cores} cores)")
            self._hw_lbl.setStyleSheet(f"color:{_P.warning};")
        try:
            from aurarouter.gui.help.setup_helpers import get_recommended_models
            self._recs = get_recommended_models(hw)
        except Exception: self._recs = []
        self._tbl.setRowCount(0)
        for i, m in enumerate(self._recs):
            r = self._tbl.rowCount(); self._tbl.insertRow(r)
            self._tbl.setItem(r, 0, QTableWidgetItem(m.get("display_name","")))
            self._tbl.setItem(r, 1, QTableWidgetItem(f"{m.get('size_gb','?')}"))
            self._tbl.setItem(r, 2, QTableWidgetItem(m.get("reason","")))
            b = QPushButton("Download"); b.clicked.connect(lambda _=False, idx=i: self._dl(idx))
            self._tbl.setCellWidget(r, 3, b)
        if nv:
            try:
                from aurarouter.gui.help.setup_helpers import suggest_cuda_sidecar
                p = suggest_cuda_sidecar(hw)
                if p: self._sidecar_pkg = p; self._gpu_w.setVisible(True)
            except Exception: pass

    def _on_hw_err(self, msg):
        self._hw_done = True; self._hw_bar.setVisible(False)
        self._hw_lbl.setText(f"Detection failed: {msg}"); self._hw_lbl.setStyleSheet(f"color:{_P.error};")

    def _dl(self, idx):
        if idx >= len(self._recs): return
        m = self._recs[idx]; repo, fn = m.get("repo",""), m.get("filename","")
        if not repo or not fn: return
        b = self._tbl.cellWidget(idx, 3)
        if b: b.setEnabled(False); b.setText("...")
        try: from aurarouter.gui.download_dialog import _DownloadWorker
        except ImportError: return
        w = _DownloadWorker(repo=repo, filename=fn)
        w.progress.connect(lambda dl, tot, r=idx: self._dlp(r, dl, tot))
        self._start_worker(w, lambda p, r=idx, md=m: self._dld(r, p, md), lambda e, r=idx: self._dle(r, e))

    def _dlp(self, row, dl, tot):
        b = self._tbl.cellWidget(row, 3)
        if b and tot > 0: b.setText(f"{int(dl*100/tot)}%")

    def _dld(self, row, path, m):
        b = self._tbl.cellWidget(row, 3)
        if b: b.setText("Done"); b.setStyleSheet(f"color:{_P.success};")
        mid = _sanitize_id("local", m.get("display_name", f"model_{row}"))
        self._state.downloaded_models.append({"model_id": mid, "display_name": m.get("display_name",""),
            "path": path, "provider": "llamacpp-server", "repo": m.get("repo",""), "filename": m.get("filename","")})

    def _dle(self, row, msg):
        b = self._tbl.cellWidget(row, 3)
        if b: b.setText("Retry"); b.setEnabled(True); b.setStyleSheet(f"color:{_P.error};")

    def _browse_hf(self):
        try:
            from aurarouter.gui.download_dialog import DownloadDialog
            DownloadDialog(parent=self).exec()
        except Exception: pass

    def _install_sidecar(self):
        if not self._sidecar_pkg: return
        self._gpu_btn.setEnabled(False); self._gpu_st.setText("Installing...")
        w = _PipWorker(self._sidecar_pkg)
        def done(ok, msg):
            if ok:
                self._gpu_st.setText("Installed."); self._gpu_st.setStyleSheet(f"color:{_P.success};")
            else:
                self._gpu_st.setText(f"Failed: {msg}"); self._gpu_st.setStyleSheet(f"color:{_P.error};")
                self._gpu_btn.setEnabled(True)
        self._start_worker(w, done, lambda m: done(False, m))

# === Page 2: On-Prem Providers ===

class OnPremProvidersPage(QWidget, _TM):
    def __init__(self, state: WizardState, parent=None):
        super().__init__(parent); self._state = state; self._init_threads()
        self._checked = False; self._cbs: list[tuple[QCheckBox, str]] = []
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(_S.lg, _S.lg, _S.lg, _S.lg); lay.setSpacing(_S.md)
        # Ollama
        lay.addWidget(_stitle("Ollama"))
        self._ol_st = QLabel("Checking..."); self._ol_st.setStyleSheet(f"color:{_P.text_secondary};"); lay.addWidget(self._ol_st)
        self._ol_area = QWidget(); self._ol_lay = QVBoxLayout(self._ol_area); self._ol_lay.setContentsMargins(0,0,0,0)
        self._ol_area.setVisible(False); lay.addWidget(self._ol_area)
        br = QHBoxLayout()
        self._ol_use = QPushButton("Use Selected"); self._ol_use.setFixedWidth(140)
        self._ol_use.clicked.connect(self._use_ollama); self._ol_use.setVisible(False); br.addWidget(self._ol_use)
        rc = QPushButton("Re-check"); rc.setFixedWidth(100); rc.clicked.connect(self._detect); br.addWidget(rc)
        br.addStretch(); lay.addLayout(br)
        # Custom endpoint
        lay.addSpacing(_S.lg); lay.addWidget(_stitle("Custom Endpoint"))
        lay.addWidget(_sdesc("Add a custom model endpoint (Ollama, llama.cpp server, or OpenAI-compatible)."))
        er = QHBoxLayout(); er.addWidget(QLabel("URL:"))
        self._ep_url = QLineEdit(); self._ep_url.setPlaceholderText("http://localhost:11434/api/generate"); er.addWidget(self._ep_url); lay.addLayout(er)
        pr = QHBoxLayout(); pr.addWidget(QLabel("Provider:"))
        self._ep_prov = QComboBox(); self._ep_prov.addItems(["ollama","llamacpp-server","openapi"]); pr.addWidget(self._ep_prov); pr.addStretch(); lay.addLayout(pr)
        nr = QHBoxLayout(); nr.addWidget(QLabel("Model:"))
        self._ep_name = QLineEdit(); self._ep_name.setPlaceholderText("qwen2.5-coder:7b"); nr.addWidget(self._ep_name); lay.addLayout(nr)
        ab = QPushButton("Test && Add"); ab.setFixedWidth(120); ab.clicked.connect(self._add_ep); lay.addWidget(ab)
        self._ep_st = QLabel(""); lay.addWidget(self._ep_st)
        lay.addStretch(); outer.addWidget(_scroll(inner))

    def showEvent(self, e):
        super().showEvent(e)
        if not self._checked: self._detect()

    def _detect(self):
        self._checked = True; self._ol_st.setText("Checking..."); self._ol_st.setStyleSheet(f"color:{_P.text_secondary};")
        self._start_worker(_OllamaWorker(), self._on_det, self._on_err)

    def _on_det(self, r):
        if r.get("available"):
            models = r.get("models", [])
            self._ol_st.setText(f"Ollama at {r.get('endpoint','localhost:11434')} -- {len(models)} model(s)")
            self._ol_st.setStyleSheet(f"color:{_P.success};")
            while self._ol_lay.count():
                w = self._ol_lay.takeAt(0).widget()
                if w: w.deleteLater()
            self._cbs.clear()
            for m in models:
                n = m if isinstance(m, str) else m.get("name", str(m))
                cb = QCheckBox(n); cb.setChecked(True); self._ol_lay.addWidget(cb); self._cbs.append((cb, n))
            self._ol_area.setVisible(bool(models)); self._ol_use.setVisible(bool(models))
        else:
            self._ol_st.setText("Ollama not detected. Install from https://ollama.com")
            self._ol_st.setStyleSheet(f"color:{_P.warning};")
            self._ol_area.setVisible(False); self._ol_use.setVisible(False)

    def _on_err(self, m):
        self._ol_st.setText(f"Error: {m}"); self._ol_st.setStyleSheet(f"color:{_P.error};")

    def _use_ollama(self):
        for cb, name in self._cbs:
            if cb.isChecked():
                mid = _sanitize_id("ollama", name)
                if not any(m["model_id"] == mid for m in self._state.ollama_models):
                    self._state.ollama_models.append({"model_id": mid, "display_name": f"Ollama: {name}",
                        "provider": "ollama", "model_name": name, "endpoint": "http://localhost:11434/api/generate"})
        self._ol_use.setText("Added!"); self._ol_use.setStyleSheet(f"color:{_P.success};")

    def _add_ep(self):
        url, name, prov = self._ep_url.text().strip(), self._ep_name.text().strip(), self._ep_prov.currentText()
        if not url or not name:
            self._ep_st.setText("Fill in URL and model name."); self._ep_st.setStyleSheet(f"color:{_P.warning};"); return
        mid = _sanitize_id("custom", name)
        self._state.custom_endpoints.append({"model_id": mid, "display_name": f"Custom: {name}",
            "provider": prov, "model_name": name, "endpoint": url})
        self._ep_st.setText(f"Added {mid}"); self._ep_st.setStyleSheet(f"color:{_P.success};"); self._ep_name.clear()

# === Page 3: Cloud Providers ===

class CloudProvidersPage(QWidget, _TM):
    def __init__(self, state: WizardState, parent=None):
        super().__init__(parent); self._state = state; self._init_threads(); self._done = False
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        inner = QWidget(); self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(_S.lg, _S.lg, _S.lg, _S.lg); self._lay.setSpacing(_S.md)
        self._lay.addWidget(_stitle("Cloud Providers"))
        self._lay.addWidget(_sdesc("Cloud providers extend AuraRouter with powerful remote models. Each is an optional pip package."))
        self._cards = QVBoxLayout(); self._cards.setSpacing(_S.md); self._lay.addLayout(self._cards)
        self._det_lbl = QLabel("Detecting..."); self._det_lbl.setStyleSheet(f"color:{_P.text_secondary};"); self._lay.addWidget(self._det_lbl)
        self._lay.addStretch(); outer.addWidget(_scroll(inner))

    def showEvent(self, e):
        super().showEvent(e)
        if not self._done: self._done = True; self._start_worker(_CloudWorker(), self._on_ok, self._on_err)

    def _on_ok(self, provs):
        self._det_lbl.setVisible(False)
        while self._cards.count():
            w = self._cards.takeAt(0).widget()
            if w: w.deleteLater()
        for p in provs: self._cards.addWidget(self._card(p))

    def _on_err(self, m):
        self._det_lbl.setText(f"Error: {m}"); self._det_lbl.setStyleSheet(f"color:{_P.error};")

    def _card(self, prov):
        c = QWidget(); c.setStyleSheet(f"QWidget{{background:{_P.bg_secondary};border:1px solid {_P.border};border-radius:6px;padding:{_S.md}px;}}")
        vl = QVBoxLayout(c); vl.setSpacing(_S.sm)
        dn = prov.get("display_name", prov.get("name","?")); t = QLabel(dn)
        t.setStyleSheet(f"font-size:{_T.size_h2}px;font-weight:bold;border:none;"); vl.addWidget(t)
        if not prov.get("installed"):
            pkg = prov.get("package",""); ib = QPushButton(f"Install ({pkg})"); ib.setFixedWidth(220)
            sl = QLabel(""); sl.setStyleSheet("border:none;")
            def do_inst(_=False, pk=pkg, b=ib, s=sl, pr=prov):
                b.setEnabled(False); s.setText("Installing..."); s.setStyleSheet(f"color:{_P.text_secondary};border:none;")
                def ok(success, msg, btn=b, st=s, p=pr):
                    if success:
                        st.setText("Installed!"); st.setStyleSheet(f"color:{_P.success};border:none;"); btn.setVisible(False)
                        p["installed"] = True; self._add_key_row(btn.parent(), p)
                    else: st.setText(f"Failed: {msg}"); st.setStyleSheet(f"color:{_P.error};border:none;"); btn.setEnabled(True)
                self._start_worker(_PipWorker(pk), ok, lambda m, b2=b, s2=s, p2=pr: ok(False, m, b2, s2, p2))
            ib.clicked.connect(do_inst); vl.addWidget(ib); vl.addWidget(sl)
        else: self._add_key_row(c, prov)
        return c

    def _add_key_row(self, parent, prov):
        lo = parent.layout()
        if not lo: return
        r = QHBoxLayout(); r.addWidget(QLabel("API Key:"))
        ki = QLineEdit(); ki.setEchoMode(QLineEdit.EchoMode.Password); ki.setPlaceholderText("Enter API key"); r.addWidget(ki)
        sb = QPushButton("Save"); sb.setFixedWidth(60); sl = QLabel(""); sl.setStyleSheet("border:none;")
        def save(_=False, k=ki, s=sl, p=prov):
            v = k.text().strip()
            if not v: s.setText("Key required"); s.setStyleSheet(f"color:{_P.warning};border:none;"); return
            nm = p.get("name","unknown")
            self._state.cloud_providers.append({"name": nm, "display_name": p.get("display_name",nm), "api_key": v, "package": p.get("package","")})
            s.setText("\u2714 Saved"); s.setStyleSheet(f"color:{_P.success};border:none;"); sb.setEnabled(False)
        sb.clicked.connect(save); r.addWidget(sb); r.addWidget(sl); lo.addLayout(r)

# === Page 4: Role Configuration ===

class RoleConfigPage(QWidget):
    _ROLES = ["router", "reasoning", "coding", "reviewer"]

    def __init__(self, state: WizardState, parent=None):
        super().__init__(parent); self._state = state
        lay = QVBoxLayout(self); lay.setContentsMargins(_S.lg, _S.lg, _S.lg, _S.lg); lay.setSpacing(_S.md)
        lay.addWidget(_stitle("Role Configuration"))
        lay.addWidget(_sdesc("Assign models to roles. Each role determines which model handles a particular part of the routing pipeline."))
        self._tbl = QTableWidget(4, 2); self._tbl.setHorizontalHeaderLabels(["Role","Model"])
        h = self._tbl.horizontalHeader(); h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch); self._tbl.verticalHeader().setVisible(False)
        self._tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self._tbl.setMinimumHeight(170)
        self._combos: dict[str, QComboBox] = {}
        for i, role in enumerate(self._ROLES):
            self._tbl.setItem(i, 0, QTableWidgetItem(role))
            cb = QComboBox(); self._combos[role] = cb; self._tbl.setCellWidget(i, 1, cb)
        lay.addWidget(self._tbl)
        self._flow = QLabel(""); self._flow.setWordWrap(True)
        self._flow.setStyleSheet(f"color:{_P.text_secondary};margin-top:{_S.md}px;"); lay.addWidget(self._flow)
        lay.addStretch()

    def showEvent(self, e):
        super().showEvent(e); self._refresh()

    def _refresh(self):
        ids = self._state.all_model_ids(); opts = ["(none)"] + ids
        for role in self._ROLES:
            cb = self._combos[role]; cur = cb.currentText()
            cb.blockSignals(True); cb.clear(); cb.addItems(opts)
            idx = cb.findText(cur)
            if idx >= 0: cb.setCurrentIndex(idx)
            cb.blockSignals(False)
        if ids:
            for role, default in [("router", ids[0]), ("reasoning", ids[-1]), ("coding", ids[-1]), ("reviewer", "(none)")]:
                cb = self._combos[role]
                if cb.currentText() == "(none)" or cb.currentIndex() <= 0:
                    i = cb.findText(default)
                    if i >= 0: cb.setCurrentIndex(i)
        parts = [f"{r} -> {self._combos[r].currentText()}" for r in self._ROLES if self._combos[r].currentText() not in ("", "(none)")]
        self._flow.setText("Routing: " + " | ".join(parts) if parts else "No models assigned yet.")

    def collect_assignments(self) -> dict[str, list[str]]:
        r: dict[str, list[str]] = {}
        for role in self._ROLES:
            m = self._combos[role].currentText()
            if m and m != "(none)": r[role] = [m]
        return r

# === Page 5: Ready ===

class ReadyPage(QWidget, _TM):
    def __init__(self, state: WizardState, config_loader: Optional["ConfigLoader"] = None, parent=None):
        super().__init__(parent); self._state = state; self._cl = config_loader; self._init_threads()
        lay = QVBoxLayout(self); lay.setContentsMargins(_S.xl, _S.xl, _S.xl, _S.xl); lay.setSpacing(_S.md)
        t = QLabel("You're Ready!"); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(f"font-size:{_T.size_h1}px;font-weight:bold;color:{_P.accent};"); lay.addWidget(t)
        self._sum = QLabel(""); self._sum.setWordWrap(True); self._sum.setStyleSheet(f"color:{_P.text_secondary};"); lay.addWidget(self._sum)
        lay.addSpacing(_S.lg)
        self._test_btn = QPushButton("Test Your Setup"); self._test_btn.setFixedWidth(160)
        self._test_btn.clicked.connect(self._test); lay.addWidget(self._test_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._res = QLabel(""); self._res.setWordWrap(True); self._res.setStyleSheet(f"color:{_P.text_secondary};"); lay.addWidget(self._res)
        lay.addStretch()

    def showEvent(self, e):
        super().showEvent(e)
        lines = []
        for label, lst in [("downloaded", self._state.downloaded_models), ("Ollama", self._state.ollama_models),
                           ("custom endpoint", self._state.custom_endpoints), ("cloud provider", self._state.cloud_providers)]:
            if lst: lines.append(f"\u2022 {len(lst)} {label}(s)")
        n = len(self._state.all_model_ids())
        lines.append(f"\nTotal models: {n}" if n else "No models configured yet.")
        self._sum.setText("\n".join(lines))

    def _test(self):
        if not self._cl:
            self._res.setText("No config loader."); self._res.setStyleSheet(f"color:{_P.warning};"); return
        self._test_btn.setEnabled(False); self._res.setText("Testing..."); self._res.setStyleSheet(f"color:{_P.text_secondary};")
        self._start_worker(_TestWorker(self._cl), self._ok, self._err)

    def _ok(self, r):
        self._test_btn.setEnabled(True); self._res.setText(f"Result: {r}"); self._res.setStyleSheet(f"color:{_P.success};")

    def _err(self, m):
        self._test_btn.setEnabled(True); self._res.setText(f"Failed: {m}"); self._res.setStyleSheet(f"color:{_P.error};")

# === Navigation bar ===

class _WizardNavBar(QWidget):
    back_clicked = Signal(); skip_clicked = Signal(); next_clicked = Signal()

    def __init__(self, n: int, parent=None):
        super().__init__(parent); self._n = n
        lay = QHBoxLayout(self); lay.setContentsMargins(_S.md, _S.sm, _S.md, _S.sm)
        self._back = QPushButton("Back"); self._back.setFixedWidth(70); self._back.clicked.connect(self.back_clicked.emit); lay.addWidget(self._back)
        lay.addStretch()
        self._dots: list[QLabel] = []
        dl = QHBoxLayout(); dl.setSpacing(_S.sm)
        for _ in range(n):
            d = QLabel(); d.setAlignment(Qt.AlignmentFlag.AlignCenter); d.setFixedSize(14,14); self._dots.append(d); dl.addWidget(d)
        lay.addLayout(dl); lay.addStretch()
        self._skip = QPushButton("Skip"); self._skip.setFixedWidth(70); self._skip.clicked.connect(self.skip_clicked.emit); lay.addWidget(self._skip)
        self._next = QPushButton("Next"); self._next.setFixedWidth(70); self._next.setDefault(True); self._next.clicked.connect(self.next_clicked.emit); lay.addWidget(self._next)
        self.set_page(0)

    def set_page(self, idx):
        self._back.setEnabled(idx > 0); self._next.setText("Finish" if idx == self._n - 1 else "Next")
        for i, d in enumerate(self._dots):
            if i <= idx:
                d.setText("\u25cf"); d.setStyleSheet(f"color:{_P.accent};font-size:12px;")
            else:
                d.setText("\u25cb"); d.setStyleSheet(f"color:{_P.text_disabled};font-size:12px;")

# === Main wizard ===

class OnboardingWizard(QDialog):
    """Six-page interactive setup wizard for AuraRouter."""

    def __init__(self, parent: Optional[QWidget] = None, config_loader: Optional["ConfigLoader"] = None, mode: str = "first_run"):
        super().__init__(parent); self._cl = config_loader; self._mode = mode; self._state = WizardState()
        self.setWindowTitle("AuraRouter Setup"); self.setMinimumSize(750, 550); self.setModal(True)
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self._stack = QStackedWidget()
        self._pages = [PersonaChooserPage(self._state), LocalModelsPage(self._state), OnPremProvidersPage(self._state),
                       CloudProvidersPage(self._state), RoleConfigPage(self._state), ReadyPage(self._state, self._cl)]
        for p in self._pages: self._stack.addWidget(p)
        root.addWidget(self._stack, 1)
        self._nav = _WizardNavBar(len(self._pages))
        self._nav.back_clicked.connect(self._back); self._nav.skip_clicked.connect(self._finish); self._nav.next_clicked.connect(self._next)
        root.addWidget(self._nav)
        if mode == "relaunch" and config_loader:
            for role in ["router","reasoning","coding","reviewer"]:
                ch = config_loader.get_role_chain(role)
                if ch: self._state.role_assignments[role] = ch

    def _next(self):
        i = self._stack.currentIndex()
        if i < self._stack.count() - 1: self._stack.setCurrentIndex(i+1); self._nav.set_page(i+1)
        else: self._finish()

    def _back(self):
        i = self._stack.currentIndex()
        if i > 0: self._stack.setCurrentIndex(i-1); self._nav.set_page(i-1)

    def _finish(self):
        rp = self._pages[4]
        if isinstance(rp, RoleConfigPage): self._state.role_assignments = rp.collect_assignments()
        if self._cl: self._flush()
        if self._mode == "first_run": mark_onboarding_complete()
        self.accept()

    def _flush(self):
        cl = self._cl
        if not cl: return
        for m in self._state.downloaded_models:
            cl.set_model(m["model_id"], {"provider": m.get("provider","llamacpp-server"), "model_path": m.get("path",""), "model_name": m.get("display_name",""), "tags": ["local"]})
        for m in self._state.ollama_models:
            cl.set_model(m["model_id"], {"provider": "ollama", "endpoint": m.get("endpoint","http://localhost:11434/api/generate"), "model_name": m.get("model_name",""), "tags": ["local"]})
        for m in self._state.custom_endpoints:
            cl.set_model(m["model_id"], {"provider": m.get("provider","openapi"), "endpoint": m.get("endpoint",""), "model_name": m.get("model_name",""), "tags": ["local"]})
        for role, chain in self._state.role_assignments.items():
            cl.set_role_chain(role, chain)
        try: cl.save()
        except Exception: pass

    def closeEvent(self, event):
        for p in self._pages:
            if hasattr(p, "_cleanup_threads"): p._cleanup_threads()
        event.accept()
