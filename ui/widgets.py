"""Small reusable Qt widgets and the background-worker plumbing."""
import contextlib
import io

from PySide6 import QtCore, QtWidgets


@contextlib.contextmanager
def silence_stdout():
    """Suppresses prints from library/app code invoked inside the block (progress logs
    meant for a terminal, not the GUI). Output is discarded -- use capture_stdout when
    the printed text itself needs to be kept."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def capture_stdout():
    """Like silence_stdout, but yields the buffer so callers can keep the captured text
    (e.g. to show a training/analysis log in the UI) instead of discarding it."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


class WorkerSignals(QtCore.QObject):
    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, object)


class Worker(QtCore.QRunnable):
    def __init__(self, name, task):
        super().__init__()
        self.name = name
        self.task = task
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.succeeded.emit(self.name, self.task())
        except Exception as error:
            self.signals.failed.emit(self.name, error)


class Card(QtWidgets.QFrame):
    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 16)
        self.layout.setSpacing(10)
        if title:
            label = QtWidgets.QLabel(title)
            label.setObjectName("cardTitle")
            self.layout.addWidget(label)


class MetricCard(Card):
    def __init__(self, label, parent=None):
        super().__init__(parent=parent)
        self.caption = QtWidgets.QLabel(label)
        self.caption.setObjectName("muted")
        self.value = QtWidgets.QLabel("--")
        self.value.setObjectName("metricValue")
        self.detail = QtWidgets.QLabel("")
        self.detail.setObjectName("muted")
        self.layout.addWidget(self.caption)
        self.layout.addWidget(self.value)
        self.layout.addWidget(self.detail)

    def update_value(self, value, detail="", color=None):
        self.value.setText(str(value))
        self.detail.setText(str(detail))
        self.value.setStyleSheet(f"color: {color};" if color else "")


class BasePage(QtWidgets.QWidget):
    def __init__(self, window, title):
        super().__init__()
        self.window = window
        self.root = QtWidgets.QVBoxLayout(self)
        self.root.setContentsMargins(28, 20, 28, 24)
        self.root.setSpacing(14)
        header = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("pageTitle")
        header.addWidget(title_label)
        header.addStretch()
        self.header_actions = header
        self.root.addLayout(header)

    def refresh(self):
        pass

    def action_button(self, text, callback, primary=False):
        button = QtWidgets.QPushButton(text)
        if primary:
            button.setProperty("primary", True)
        button.clicked.connect(callback)
        self.header_actions.addWidget(button)
        return button

    def add_metric_row(self, labels):
        """Adds a QHBoxLayout of one MetricCard per label to the page and returns the
        cards in order, so callers keep a single call instead of hand-building the same
        row layout on every page that shows a metric strip."""
        row = QtWidgets.QHBoxLayout()
        cards = [MetricCard(label) for label in labels]
        for card in cards:
            row.addWidget(card, 1)
        self.root.addLayout(row)
        return cards
