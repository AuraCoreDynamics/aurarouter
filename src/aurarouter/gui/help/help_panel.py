"""Help panel widget for the AuraRouter GUI.

Provides a searchable, filterable help viewer with a topic list on the
left (40%) and a detail pane on the right (60%).  Related-topic links
are clickable.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.help.content import HELP, HelpTopic


class HelpPanel(QWidget):
    """Searchable help browser embedded in the GUI."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_topic: Optional[HelpTopic] = None
        self._build_ui()
        self._populate_list(HELP.all_topics())
        # Select first topic if available.
        if self._topic_list.count() > 0:
            self._topic_list.setCurrentRow(0)

    # ----------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # -- Search bar + category filter --
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Type to search topics...")
        self._search_input.textChanged.connect(self._on_search)
        toolbar.addWidget(self._search_input, 1)

        toolbar.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.addItems(["All", "concept", "panel", "howto", "glossary"])
        self._category_combo.currentTextChanged.connect(self._on_filter)
        toolbar.addWidget(self._category_combo)

        root.addLayout(toolbar)

        # -- Splitter: topic list (left) | detail (right) --
        splitter = QSplitter()

        # Left: topic list
        self._topic_list = QListWidget()
        self._topic_list.currentItemChanged.connect(self._on_topic_selected)
        splitter.addWidget(self._topic_list)

        # Right: detail browser
        self._detail_browser = QTextBrowser()
        self._detail_browser.setOpenLinks(False)
        self._detail_browser.anchorClicked.connect(self._on_link_clicked)
        splitter.addWidget(self._detail_browser)

        # 40/60 split
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        root.addWidget(splitter, 1)

    # ----------------------------------------------------------
    # List population
    # ----------------------------------------------------------

    def _populate_list(self, topics: list[HelpTopic]) -> None:
        self._topic_list.blockSignals(True)
        self._topic_list.clear()
        for topic in topics:
            item = QListWidgetItem(f"[{topic.category}]  {topic.title}")
            item.setData(Qt.ItemDataRole.UserRole, topic.id)
            self._topic_list.addItem(item)
        self._topic_list.blockSignals(False)

    # ----------------------------------------------------------
    # Filtering / search
    # ----------------------------------------------------------

    def _on_search(self, text: str) -> None:
        cat = self._category_combo.currentText()
        topics = HELP.search(text) if text.strip() else HELP.all_topics()
        if cat != "All":
            topics = [t for t in topics if t.category == cat]
        self._populate_list(topics)
        if self._topic_list.count() > 0:
            self._topic_list.setCurrentRow(0)
        else:
            self._detail_browser.setHtml(
                "<p style='color:gray;'>No topics match your search.</p>"
            )

    def _on_filter(self, category: str) -> None:
        query = self._search_input.text().strip()
        if category == "All":
            topics = HELP.search(query) if query else HELP.all_topics()
        else:
            base = HELP.search(query) if query else HELP.all_topics()
            topics = [t for t in base if t.category == category]
        self._populate_list(topics)
        if self._topic_list.count() > 0:
            self._topic_list.setCurrentRow(0)
        elif not topics:
            self._detail_browser.setHtml(
                "<p style='color:gray;'>No topics in this category.</p>"
            )

    # ----------------------------------------------------------
    # Topic display
    # ----------------------------------------------------------

    def _on_topic_selected(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        topic_id = current.data(Qt.ItemDataRole.UserRole)
        topic = HELP.get(topic_id)
        if topic is None:
            return
        self._current_topic = topic
        self._show_topic(topic)

    def _show_topic(self, topic: HelpTopic) -> None:
        """Render a topic with its body and related-topic links."""
        html_parts = [topic.body]

        # Related topics as clickable links
        if topic.related:
            html_parts.append("<hr><p><b>Related topics:</b></p><ul>")
            for rid in topic.related:
                related = HELP.get(rid)
                if related:
                    html_parts.append(
                        f'<li><a href="topic://{rid}">{related.title}</a></li>'
                    )
            html_parts.append("</ul>")

        self._detail_browser.setHtml("\n".join(html_parts))

    def _on_link_clicked(self, url) -> None:  # type: ignore[override]
        """Navigate to a related topic when its link is clicked."""
        url_str = url.toString()
        if url_str.startswith("topic://"):
            topic_id = url_str[len("topic://"):]
            topic = HELP.get(topic_id)
            if topic is None:
                return
            # Select the topic in the list if visible, otherwise just show it.
            for i in range(self._topic_list.count()):
                item = self._topic_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == topic_id:
                    self._topic_list.setCurrentRow(i)
                    return
            # Topic not in current filtered list — show directly and reset filter.
            self._search_input.clear()
            self._category_combo.setCurrentText("All")
            self._populate_list(HELP.all_topics())
            for i in range(self._topic_list.count()):
                item = self._topic_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == topic_id:
                    self._topic_list.setCurrentRow(i)
                    return

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def show_topic(self, topic_id: str) -> None:
        """Programmatically navigate to a specific topic."""
        self._search_input.clear()
        self._category_combo.setCurrentText("All")
        self._populate_list(HELP.all_topics())
        for i in range(self._topic_list.count()):
            item = self._topic_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == topic_id:
                self._topic_list.setCurrentRow(i)
                return
