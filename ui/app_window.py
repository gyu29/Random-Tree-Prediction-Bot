"""Main window shell, navigation, and the glue between background tasks and pages."""
import json
import os
import sys
from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except ImportError as error:
    raise RuntimeError(
        "The Qt interface requires PySide6 and PyQtGraph. Install them with: pip install PySide6 pyqtgraph"
    ) from error

from app import model_registry
from app.config import DEFAULT_DECISION_THRESHOLD, PROJECT_ROOT, STOCKS_FILE_PATH, UI_CONFIG_PATH, write_env_value
from app.data_loader import list_categories
from ui.pages.analyze import AnalyzePage
from ui.pages.backtest import BacktestPage
from ui.pages.dashboard import DashboardPage
from ui.pages.docs import DocsPage
from ui.pages.settings import SettingsPage
from ui.pages.signal_page import MonitorPage, SignalPage
from ui.pages.train import TrainPage
from ui.state import AppState
from ui.tasks import TaskController
from ui.widgets import Worker

PAGE_NAMES = ["Home", "Analyze", "Monitor", "Backtest", "Screener", "Train model", "Settings", "Docs"]

DEFAULT_KR_SYMBOL = "005930"  # Samsung Electronics
DEFAULT_US_SYMBOL = "AAPL"

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

# Same dark palette as docs/index.html's dark theme (styles.css), so the desktop terminal
# and the promotional site read as the same product instead of two different designs.
COLORS = {
    "bg": "#0A0E0B", "bg_top": "#10140F", "bg_bottom": "#0A0E0B",
    "panel": "#141A16", "panel_alt": "#1C231D", "border": "#263025", "shadow": "#000000",
    "text": "#EAEDE4", "muted": "#9BA396", "subtle": "#20281F",
    "teal": "#39CDBE", "teal_soft": "#14302C", "green": "#4FBE82",
    "amber": "#E8A23D", "amber_soft": "#2E2415", "coral": "#E2695C",
    "red": "#E2695C", "crimson": "#0E7A70", "blue": "#5B8FD6",
}


class TradingTerminalWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.colors = COLORS
        self.project_root = PROJECT_ROOT
        self.config_path = UI_CONFIG_PATH

        self.state = AppState()
        self.tasks = TaskController(self.state)
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.workers = set()
        self.busy_tasks = set()

        self.monitoring = False
        self.monitor_interval_seconds = 300
        self.training_running = False
        self.training_complete = False
        self.training_log = []

        self.config_data = self.load_config()
        self.state.set_watchlist(self.config_data.get("watchlist") or self.load_watchlist())
        watchlist = self.state.get_watchlist()
        self.market_mode = self.config_data.get("market_mode") or (
            "US" if any(any(char.isalpha() for char in symbol) for symbol in watchlist) else "KR"
        )
        self.last_symbol = self.config_data.get("last_symbol") or (
            watchlist[0] if watchlist else (DEFAULT_KR_SYMBOL if self.market_mode == "KR" else DEFAULT_US_SYMBOL)
        )
        self.alerts = self.config_data.get("alerts") or [
            {"kind": "system", "text": "Qt desktop terminal initialized.", "time": "now"}
        ]
        self.data_freshness = self.display_freshness([])
        self.dashboard_data = self.empty_dashboard_data()
        self.pages = {}
        self.nav_buttons = {}

        self.setWindowTitle("Random Tree Bot")
        self.resize(1360, 860)
        self.setMinimumSize(1100, 700)
        self.build_shell()
        self.apply_theme()
        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.timeout.connect(self.refresh_dashboard)
        self.show_screen("Home")
        if self.market_mode == "KR":
            QtCore.QTimer.singleShot(350, self.refresh_dashboard)

    # -- shell / theme ------------------------------------------------------------
    def build_shell(self):
        central = QtWidgets.QWidget()
        shell = QtWidgets.QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(14, 20, 14, 16)
        brand = QtWidgets.QLabel("Random Tree Bot")
        brand.setObjectName("brand")
        subtitle = QtWidgets.QLabel("Swing trading research")
        subtitle.setObjectName("muted")
        side.addWidget(brand)
        side.addWidget(subtitle)
        side.addSpacing(18)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for name in PAGE_NAMES:
            button = QtWidgets.QPushButton(name)
            button.setCheckable(True)
            button.setProperty("nav", True)
            button.clicked.connect(lambda checked=False, page=name: self.show_screen(page))
            group.addButton(button)
            side.addWidget(button)
            self.nav_buttons[name] = button
            if name == "Train model":
                side.addSpacing(12)
        side.addStretch()
        self.market_button = QtWidgets.QPushButton()
        self.market_button.clicked.connect(self.toggle_market)
        side.addWidget(self.market_button)
        self.model_status = QtWidgets.QLabel()
        self.model_status.setObjectName("muted")
        side.addWidget(self.model_status)

        self.stack = QtWidgets.QStackedWidget()
        page_types = {
            "Home": DashboardPage,
            "Analyze": AnalyzePage,
            "Monitor": MonitorPage,
            "Backtest": BacktestPage,
            "Screener": lambda window: SignalPage(window, "Screener", "Run screener"),
            "Train model": TrainPage,
            "Settings": SettingsPage,
            "Docs": DocsPage,
        }
        for name, page_type in page_types.items():
            page = page_type(self)
            self.pages[name] = page
            self.stack.addWidget(page)
        shell.addWidget(sidebar)
        shell.addWidget(self.stack, 1)
        self.setCentralWidget(central)

    def apply_theme(self):
        pg.setConfigOptions(antialias=True, background=self.colors["panel"], foreground=self.colors["muted"])
        mono = '"IBM Plex Mono", Menlo, Consolas, monospace'
        serif = '"Spectral", Georgia, serif'
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {self.colors["bg"]};
                color: {self.colors["text"]};
                font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
                font-size: 13px;
            }}
            QFrame#sidebar {{
                background: {self.colors["panel"]};
                border-right: 1px solid {self.colors["border"]};
            }}
            QFrame#card {{
                background: {self.colors["panel"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 12px;
            }}
            QLabel#brand {{ font-family: {mono}; font-size: 16px; font-weight: 500; color: {self.colors["text"]}; }}
            QLabel#pageTitle {{ font-family: {serif}; font-size: 24px; font-weight: 600; }}
            QLabel#cardTitle {{ font-size: 14px; font-weight: 600; }}
            QLabel#metricValue {{ font-family: {mono}; font-size: 22px; font-weight: 500; }}
            QLabel#muted {{ color: {self.colors["muted"]}; }}
            QPushButton {{
                background: {self.colors["panel_alt"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QPushButton:hover {{ background: {self.colors["subtle"]}; }}
            QPushButton[primary="true"] {{
                color: {self.colors["bg"]};
                background: {self.colors["teal"]};
                border-color: {self.colors["teal"]};
                font-weight: 600;
            }}
            QPushButton[nav="true"] {{
                text-align: left;
                border: none;
                border-left: 3px solid transparent;
                padding: 10px 12px;
                color: {self.colors["muted"]};
            }}
            QPushButton[nav="true"]:checked {{
                color: {self.colors["teal"]};
                background: {self.colors["teal_soft"]};
                border-left: 3px solid {self.colors["teal"]};
                font-weight: 600;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget, QTableWidget {{
                background: {self.colors["panel"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 7px;
                padding: 6px;
                selection-background-color: {self.colors["teal"]};
                selection-color: {self.colors["bg"]};
            }}
            QTableWidget {{ font-family: {mono}; gridline-color: {self.colors["border"]}; }}
            QHeaderView::section {{
                background: {self.colors["panel_alt"]};
                color: {self.colors["muted"]};
                border: none;
                border-bottom: 1px solid {self.colors["border"]};
                padding: 8px;
            }}
            QProgressBar {{
                border: 1px solid {self.colors["border"]};
                border-radius: 7px;
                text-align: center;
                background: {self.colors["subtle"]};
                min-height: 24px;
            }}
            QProgressBar::chunk {{ background: {self.colors["teal"]}; border-radius: 6px; }}
        """)

    def show_screen(self, name):
        page = self.pages[name]
        self.stack.setCurrentWidget(page)
        self.nav_buttons[name].setChecked(True)
        page.refresh()
        self.refresh_sidebar()

    def refresh_sidebar(self, trained_categories=None):
        self.market_button.setText(self.market_label())
        trained = model_registry.list_trained_categories() if trained_categories is None else trained_categories
        total = len(list_categories())
        self.model_status.setText(f"Models {len(trained)}/{total} trained")

    def toggle_market(self):
        self.market_mode = "US" if self.market_mode == "KR" else "KR"
        self.last_symbol = DEFAULT_US_SYMBOL if self.market_mode == "US" else DEFAULT_KR_SYMBOL
        self.state.invalidate_detector()
        self.state.clear_market_cache()
        self.save_config()
        self.pages["Analyze"].market.setCurrentText(self.market_mode)
        self.pages["Analyze"].symbol.setText(self.last_symbol)
        self.refresh_sidebar()

    def market_label(self):
        return "KR market" if self.market_mode == "KR" else "US market"

    def current_page_name(self):
        current = self.stack.currentWidget()
        return next(name for name, page in self.pages.items() if page is current)

    # -- table helpers --------------------------------------------------------------
    def make_table(self, headers):
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        return table

    def fill_table(self, table, rows):
        table.setUpdatesEnabled(False)
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column > 1:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(row_index, column, item)
        table.setUpdatesEnabled(True)

    def fill_signal_table(self, table, signals):
        rows = []
        for signal in signals:
            available = signal.get("status") != "unavailable"
            probability_text = self.pct(signal.get("probability", 0)) if available else "--"
            if available and signal.get("source") == "heuristic_no_model":
                probability_text += " (heuristic)"
            rows.append([
                signal.get("symbol", ""),
                signal.get("company", signal.get("symbol", "")),
                signal.get("category", "--"),
                probability_text,
                signal.get("confidence", "--"),
            ])
        self.fill_table(table, rows)

    def style_plot(self, plot):
        plot.setBackground(self.colors["panel"])
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getPlotItem().hideButtons()

    # -- background task orchestration -----------------------------------------------
    def run_background(self, name, task, success):
        if name in self.busy_tasks:
            return
        self.busy_tasks.add(name)
        worker = Worker(name, task)
        self.workers.add(worker)

        def completed(task_name, result):
            self.busy_tasks.discard(task_name)
            self.workers.discard(worker)
            success(result)

        def failed(task_name, error):
            self.busy_tasks.discard(task_name)
            self.workers.discard(worker)
            if task_name == "training":
                self.training_running = False
                self.training_log.append(f"[{self.now_time()}] ERROR: {error}")
                self.pages["Train model"].refresh()
            self.add_alert("warning", f"{task_name.title()} failed: {error}")
            QtWidgets.QMessageBox.critical(self, f"{task_name.title()} failed", str(error))
            self.pages[self.current_page_name()].refresh()

        worker.signals.succeeded.connect(completed)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def refresh_dashboard(self):
        watchlist_entries = self.state.get_watchlist_entries()
        mode = self.market_mode
        self.run_background(
            "dashboard", lambda: self.tasks.compute_dashboard_data(watchlist_entries, mode), self.dashboard_ready
        )

    def dashboard_ready(self, data):
        self.data_freshness = self.display_freshness(data.pop("displayed_dates", []))
        trained = model_registry.list_trained_categories()
        self.dashboard_data = {
            **data,
            "model_ready": len(trained) > 0,
            "artifact_count": len(trained),
            "total_categories": len(list_categories()),
            "data_freshness": self.data_freshness,
        }
        for name in ["Home", "Monitor", "Screener"]:
            self.pages[name].refresh()
        self.refresh_sidebar(trained_categories=trained)

    def start_analysis(self, symbol, mode, category=None):
        if mode != self.market_mode:
            self.state.invalidate_detector()
            self.state.clear_market_cache()
        self.market_mode = mode
        self.last_symbol = symbol
        self.add_alert("system", f"Started analysis for {symbol}.")
        self.run_background("analysis", lambda: self.tasks.analysis_task(symbol, mode, category), self.analysis_ready)

    def analysis_ready(self, payload):
        self.state.set_last_analysis(payload)
        result = payload["result"]
        self.add_alert(
            "signal" if result.get("swing_probability", 0) >= DEFAULT_DECISION_THRESHOLD else "system",
            f"{result['symbol']} probability {self.pct(result.get('swing_probability', 0))}.",
        )
        self.save_config()
        self.pages["Analyze"].refresh()

    def start_backtest(self, category, symbol, file_path, days, threshold, df=None):
        self.run_background(
            "backtest",
            lambda: self.tasks.backtest_task(
                category, symbol, self.market_mode, days, threshold, file_path=file_path, df=df
            ),
            self.backtest_ready,
        )

    def backtest_ready(self, result):
        self.state.set_last_backtest(result)
        self.last_symbol = result.get("symbol", self.last_symbol)
        symbol = result.get("symbol", self.last_symbol)
        if result.get("no_trades_reason"):
            self.add_alert(
                "system",
                f"Backtest for {symbol} triggered no trades (peak probability "
                f"{result.get('peak_probability', 0):.0%}). See Backtest page for details.",
            )
        else:
            self.add_alert("system", f"Backtest completed for {symbol}.")
        self.pages["Backtest"].refresh()

    def start_training(self, category, analyze_after=False):
        if self.training_running:
            return
        page = self.pages["Train model"]
        parameters = {
            "swing_threshold": page.swing_threshold.value() / 100,
            "swing_window": page.swing_window.value(),
            "rf_estimators": page.rf_estimators.value(),
            "learning_rate": page.learning_rate.value(),
            "max_depth": page.max_depth.value(),
        }
        self.training_running = True
        self.training_complete = False
        self.training_log.append(f"[{self.now_time()}] Training job started for category '{category}'.")
        self.pages["Train model"].refresh()
        self.run_background(
            "training",
            lambda: self.tasks.training_task(category, parameters),
            lambda result: self.training_ready(result, analyze_after),
        )

    def training_ready(self, result, analyze_after=False):
        self.training_running = False
        self.training_complete = True
        for line in result["log"].splitlines()[-80:]:
            self.training_log.append(f"[{self.now_time()}] {line}")
        # PR-AUC against the base rate, not accuracy: swing labels are 1-8% of rows, so
        # an accuracy in the high 90s is what "never predict a swing" scores.
        stats = result["training_stats"]
        base_rate = stats.get("validation_base_rate", 0.0)
        self.training_log.append(
            f"[{self.now_time()}] Training completed for '{result['category']}' "
            f"with validation PR-AUC {stats.get('validation_pr_auc', float('nan')):.4f} "
            f"against a {base_rate:.2%} base rate "
            f"(accuracy {result['validation_score']:.4f}, "
            f"{stats.get('validation_majority_accuracy', 1 - base_rate):.4f} for always-negative)."
        )
        self.state.invalidate_detector(result["category"])
        self.add_alert("system", f"Model retraining completed for '{result['category']}'.")
        self.pages["Train model"].refresh()
        self.refresh_sidebar()
        if analyze_after:
            self.show_screen("Analyze")

    # -- settings / config ------------------------------------------------------------
    def save_settings(self, krx_key, alpha_key, market, interval):
        write_env_value("KRX_SERVICE_KEY", krx_key)
        write_env_value("ALPHA_VANTAGE_API_KEY", alpha_key)
        self.market_mode = market
        self.monitor_interval_seconds = interval
        self.state.invalidate_detector()
        self.state.clear_market_cache()
        self.save_config()
        self.add_alert("system", "Settings saved.")
        self.refresh_sidebar()
        QtWidgets.QMessageBox.information(self, "Settings", "Settings saved.")

    def export_backtest(self):
        result = self.state.get_last_backtest()
        if not result:
            return
        path = os.path.join(self.project_root, f"{result.get('symbol', 'backtest')}_backtest.json")
        with open(path, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2, default=str)
        self.add_alert("system", f"Backtest exported to {os.path.basename(path)}.")

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as config:
                return json.load(config)
        except (OSError, ValueError):
            return {}

    def save_config(self):
        payload = {
            "market_mode": self.market_mode,
            "last_symbol": self.last_symbol,
            "watchlist": self.state.get_watchlist_entries(),
            "alerts": self.alerts[:12],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.config_path, "w", encoding="utf-8") as config:
            json.dump(payload, config, indent=2)

    def load_watchlist(self):
        try:
            with open(STOCKS_FILE_PATH, "r", encoding="utf-8") as stocks:
                values = json.load(stocks)
            if isinstance(values, list):
                return [str(value).upper() for value in values[:16]]
        except (OSError, ValueError):
            pass
        return [DEFAULT_KR_SYMBOL, "000660", "035420"]

    def empty_dashboard_data(self):
        trained = model_registry.list_trained_categories()
        return {
            "model_ready": len(trained) > 0,
            "artifact_count": len(trained),
            "total_categories": len(list_categories()),
            "signals": [],
            "watchlist": [],
            "avg_probability": 0,
            "signals_today": 0,
            "data_freshness": self.data_freshness,
        }

    def display_freshness(self, dates):
        clean = [value for value in dates if value is not None]
        if not clean:
            return {"label": "No market data", "detail": "No visible market rows loaded", "color": self.colors["red"]}
        oldest = min(clean)
        latest = max(clean)
        age = max(0, (datetime.now().date() - oldest).days)
        color = self.colors["green"] if age <= 3 else self.colors["amber"] if age <= 14 else self.colors["red"]
        label = f"As of {latest.isoformat()}" if oldest == latest else f"{oldest.isoformat()} to {latest.isoformat()}"
        return {"label": label, "detail": f"Oldest visible row {age}d", "color": color}

    def add_alert(self, kind, text):
        self.alerts.insert(0, {"kind": kind, "text": text, "time": self.now_time()})
        self.alerts = self.alerts[:12]
        if "Home" in self.pages:
            self.pages["Home"].refresh()

    @staticmethod
    def now_time():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def pct(value, signed=False):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0
        prefix = "+" if signed and number > 0 else ""
        return f"{prefix}{number * 100:.1f}%"

    @staticmethod
    def money(value, currency):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0
        return f"${number:,.2f}" if currency == "USD" else f"KRW {number:,.0f}"


def load_fonts():
    """Registers the same typefaces docs/index.html embeds (IBM Plex Sans/Mono,
    Spectral) so the desktop terminal and the promotional site match instead of one
    of them silently falling back to a generic system font. Missing/unreadable files
    are skipped rather than raised -- a font that fails to load just means apply_theme's
    "IBM Plex Sans"/"IBM Plex Mono"/"Spectral" QSS references fall back to the next
    family in their chain, not a crash."""
    if not os.path.isdir(FONTS_DIR):
        return
    for filename in os.listdir(FONTS_DIR):
        if filename.lower().endswith((".ttf", ".otf")):
            QtGui.QFontDatabase.addApplicationFont(os.path.join(FONTS_DIR, filename))


def launch_qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Random Tree Bot")
    app.setStyle("Fusion")
    load_fonts()
    window = TradingTerminalWindow()
    window.show()
    return app.exec()
