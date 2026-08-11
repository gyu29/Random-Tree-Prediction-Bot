import os
import webbrowser

from PySide6 import QtWidgets

from ui.widgets import BasePage, Card


class DocsPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Docs")
        self.action_button("Open README", self.open_readme)
        card = Card("README")
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        card.layout.addWidget(self.text)
        self.root.addWidget(card, 1)
        self._readme_cache = None

    def refresh(self):
        if self._readme_cache is None:
            try:
                with open(os.path.join(self.window.project_root, "README.md"), "r", encoding="utf-8") as readme:
                    self._readme_cache = readme.read()
            except OSError as error:
                # Not cached: a transient read failure gets retried on the next visit
                # instead of being shown forever.
                self.text.setPlainText(str(error))
                return
        self.text.setPlainText(self._readme_cache)

    def open_readme(self):
        webbrowser.open(os.path.join(self.window.project_root, "README.md"))
