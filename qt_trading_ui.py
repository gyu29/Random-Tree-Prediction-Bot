import contextlib
import glob
import io
import json
import math
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as error:
    raise RuntimeError(
        "The Qt interface requires PySide6 and PyQtGraph. "
        "Install them with: pip install PySide6 pyqtgraph"
    ) from error


PAGE_NAMES = [
    "Home",
    "Analyze",
    "Monitor",
    "Backtest",
    "Screener",
    "Train model",
    "Settings",
    "Docs",
]


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


class DashboardPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Dashboard")
        self.action_button("Refresh", window.refresh_dashboard, primary=True)

        metrics = QtWidgets.QHBoxLayout()
        self.metrics = [
            MetricCard("Model status"),
            MetricCard("Signals today"),
            MetricCard("Watchlist size"),
            MetricCard("Average swing probability"),
        ]
        for card in self.metrics:
            metrics.addWidget(card, 1)
        self.root.addLayout(metrics)

        split = QtWidgets.QHBoxLayout()
        signals_card = Card("Top swing signals")
        self.signals = window.make_table(["Symbol", "Company", "Probability", "Confidence"])
        signals_card.layout.addWidget(self.signals)
        split.addWidget(signals_card, 3)

        watch_card = Card("Watchlist")
        self.watchlist = window.make_table(["Symbol", "Price", "Change"])
        watch_card.layout.addWidget(self.watchlist)
        split.addWidget(watch_card, 2)
        self.root.addLayout(split, 1)

        alerts_card = Card("Recent alerts")
        self.alerts = QtWidgets.QListWidget()
        self.alerts.setMaximumHeight(150)
        alerts_card.layout.addWidget(self.alerts)
        self.root.addWidget(alerts_card)

    def refresh(self):
        data = self.window.dashboard_data
        ready = data.get("model_ready", False)
        self.metrics[0].update_value(
            "Ready" if ready else "Missing",
            data.get("ensemble", ""),
            self.window.colors["teal"] if ready else self.window.colors["amber"],
        )
        self.metrics[1].update_value(data.get("signals_today", 0), "active setups")
        freshness = data.get("data_freshness", {})
        self.metrics[2].update_value(
            len(self.window.watchlist),
            freshness.get("label", "No market data"),
            freshness.get("color"),
        )
        self.metrics[3].update_value(
            self.window.pct(data.get("avg_probability", 0)),
            "available signals",
            self.window.colors["teal"],
        )
        self.window.fill_signal_table(self.signals, data.get("signals", [])[:10])
        rows = []
        for item in data.get("watchlist", [])[:10]:
            rows.append([
                item.get("symbol", ""),
                self.window.money(item.get("price", 0), item.get("currency", "USD")),
                self.window.pct(item.get("change", 0), signed=True),
            ])
        self.window.fill_table(self.watchlist, rows)
        self.alerts.clear()
        for alert in self.window.alerts[:6]:
            self.alerts.addItem(f"{alert.get('time', 'now')}  {alert.get('text', '')}")


class AnalyzePage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Symbol analysis")
        controls = QtWidgets.QHBoxLayout()
        self.symbol = QtWidgets.QLineEdit(window.last_symbol)
        self.symbol.setPlaceholderText("Ticker or Korean market code")
        self.market = QtWidgets.QComboBox()
        self.market.addItems(["KR", "US"])
        self.market.setCurrentText(window.market_mode)
        run = QtWidgets.QPushButton("Analyze")
        run.setProperty("primary", True)
        run.clicked.connect(self.run_analysis)
        controls.addWidget(self.symbol, 1)
        controls.addWidget(self.market)
        controls.addWidget(run)
        self.root.addLayout(controls)

        metrics = QtWidgets.QHBoxLayout()
        self.metrics = [
            MetricCard("Symbol"),
            MetricCard("Current price"),
            MetricCard("Swing probability"),
            MetricCard("Volume"),
        ]
        for card in self.metrics:
            metrics.addWidget(card, 1)
        self.root.addLayout(metrics)

        charts = QtWidgets.QHBoxLayout()
        price_card = Card("Recent price")
        self.price_plot = pg.PlotWidget()
        self.window.style_plot(self.price_plot)
        price_card.layout.addWidget(self.price_plot)
        charts.addWidget(price_card, 3)

        levels_card = Card("Trade levels")
        self.probability = QtWidgets.QProgressBar()
        self.probability.setRange(0, 100)
        self.probability.setFormat("Swing probability %p%")
        levels_card.layout.addWidget(self.probability)
        self.levels = QtWidgets.QFormLayout()
        self.level_values = {}
        for name in ["Confidence", "Entry", "Take-profit", "Stop-loss", "Region"]:
            value = QtWidgets.QLabel("--")
            self.level_values[name] = value
            self.levels.addRow(name, value)
        levels_card.layout.addLayout(self.levels)
        levels_card.layout.addStretch()
        charts.addWidget(levels_card, 2)
        self.root.addLayout(charts, 1)

        table_card = Card("Recent OHLCV")
        self.ohlcv = window.make_table(["Date", "Open", "High", "Low", "Close", "Volume"])
        self.ohlcv.setMaximumHeight(230)
        table_card.layout.addWidget(self.ohlcv)
        self.root.addWidget(table_card)

    def run_analysis(self):
        symbol = self.symbol.text().strip().upper()
        mode = self.market.currentText()
        if symbol:
            self.window.start_analysis(symbol, mode)

    def refresh(self):
        payload = self.window.last_analysis
        if not payload:
            self.metrics[0].update_value(self.window.last_symbol, "Awaiting analysis")
            return
        result = payload["result"]
        currency = result.get("currency", "USD")
        self.symbol.setText(result.get("symbol", self.window.last_symbol))
        self.market.setCurrentText(self.window.market_mode)
        self.metrics[0].update_value(result.get("symbol", ""), payload.get("company", ""))
        self.metrics[1].update_value(
            self.window.money(result.get("current_price", 0), currency),
            self.window.pct(result.get("price_change_1d", 0), signed=True),
        )
        self.metrics[2].update_value(
            self.window.pct(result.get("swing_probability", 0)),
            result.get("confidence_level", ""),
            self.window.colors["teal"],
        )
        self.metrics[3].update_value(f"{int(result.get('current_volume', 0)):,}", "latest session")
        probability = float(result.get("swing_probability", 0))
        self.probability.setValue(round(probability * 100))
        values = {
            "Confidence": result.get("confidence_level", "--"),
            "Entry": self.window.money(result.get("current_price", 0), currency),
            "Take-profit": self.window.money(result.get("take_profit", 0), currency),
            "Stop-loss": self.window.money(result.get("stop_loss", 0), currency),
            "Region": self.window.market_label(),
        }
        for name, value in values.items():
            self.level_values[name].setText(str(value))

        df = payload.get("df")
        self.price_plot.clear()
        if df is not None and not df.empty:
            close = df.tail(120)["close"].astype(float).to_numpy()
            self.price_plot.plot(close, pen=pg.mkPen(self.window.colors["teal"], width=2))
            rows = []
            for index, row in df.tail(8).iloc[::-1].iterrows():
                rows.append([
                    pd.to_datetime(index).strftime("%Y-%m-%d"),
                    f"{float(row.get('open', 0)):,.2f}",
                    f"{float(row.get('high', 0)):,.2f}",
                    f"{float(row.get('low', 0)):,.2f}",
                    f"{float(row.get('close', 0)):,.2f}",
                    f"{int(row.get('volume', 0)):,}",
                ])
            self.window.fill_table(self.ohlcv, rows)


class SignalPage(BasePage):
    def __init__(self, window, title, button_text):
        super().__init__(window, title)
        self.action_button(button_text, window.refresh_dashboard, primary=True)
        card = Card("Live watchlist signals")
        self.status = QtWidgets.QLabel("Idle")
        self.status.setObjectName("muted")
        card.layout.addWidget(self.status)
        self.table = window.make_table(["Symbol", "Company", "Probability", "Confidence"])
        card.layout.addWidget(self.table)
        self.root.addWidget(card, 1)

    def refresh(self):
        self.status.setText(
            f"{len(self.window.watchlist)} symbols, {self.window.dashboard_data.get('data_freshness', {}).get('label', 'not refreshed')}"
        )
        self.window.fill_signal_table(self.table, self.window.dashboard_data.get("signals", []))


class MonitorPage(SignalPage):
    def __init__(self, window):
        super().__init__(window, "Monitor", "Scan now")
        self.toggle = self.action_button("Start monitoring", self.toggle_monitoring)

    def toggle_monitoring(self):
        self.window.monitoring = not self.window.monitoring
        if self.window.monitoring:
            self.window.monitor_timer.start(self.window.monitor_interval_seconds * 1000)
            self.window.refresh_dashboard()
        else:
            self.window.monitor_timer.stop()
        self.refresh()

    def refresh(self):
        super().refresh()
        self.toggle.setText("Stop monitoring" if self.window.monitoring else "Start monitoring")


class BacktestPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Backtest")
        self.action_button("Export", window.export_backtest)
        controls = QtWidgets.QHBoxLayout()
        self.symbol = QtWidgets.QLineEdit(window.last_symbol)
        self.window_select = QtWidgets.QComboBox()
        self.window_select.addItems(["90 days", "180 days", "1 year", "3 years"])
        self.window_select.setCurrentText("180 days")
        self.threshold = QtWidgets.QComboBox()
        self.threshold.addItems(["60%", "70%", "80%"])
        self.threshold.setCurrentText("70%")
        run = QtWidgets.QPushButton("Run backtest")
        run.setProperty("primary", True)
        run.clicked.connect(self.run_backtest)
        controls.addWidget(self.symbol, 1)
        controls.addWidget(self.window_select)
        controls.addWidget(self.threshold)
        controls.addWidget(run)
        self.root.addLayout(controls)

        metrics = QtWidgets.QHBoxLayout()
        self.metrics = [
            MetricCard("Total return"),
            MetricCard("Win rate"),
            MetricCard("Sharpe ratio"),
            MetricCard("Maximum drawdown"),
        ]
        for card in self.metrics:
            metrics.addWidget(card, 1)
        self.root.addLayout(metrics)

        chart = Card("Equity curve")
        self.equity_plot = pg.PlotWidget()
        self.window.style_plot(self.equity_plot)
        self.equity_plot.addLegend()
        chart.layout.addWidget(self.equity_plot)
        self.root.addWidget(chart, 1)

        trades = Card("Recent trades")
        self.trade_table = window.make_table(["Exit date", "Symbol", "Entry", "Exit", "P&L", "Reason"])
        self.trade_table.setMaximumHeight(220)
        trades.layout.addWidget(self.trade_table)
        self.root.addWidget(trades)

    def run_backtest(self):
        symbol = self.symbol.text().strip().upper()
        days = {"90 days": 90, "180 days": 180, "1 year": 365, "3 years": 1095}[self.window_select.currentText()]
        threshold = float(self.threshold.currentText().replace("%", "")) / 100
        if symbol:
            self.window.start_backtest(symbol, days, threshold)

    def refresh(self):
        result = self.window.last_backtest
        if not result:
            return
        self.symbol.setText(result.get("symbol", self.window.last_symbol))
        self.metrics[0].update_value(
            self.window.pct(result.get("total_return", 0), signed=True),
            f"{result.get('num_trades', 0)} trades",
            self.window.colors["teal"] if result.get("total_return", 0) >= 0 else self.window.colors["red"],
        )
        self.metrics[1].update_value(self.window.pct(result.get("win_rate", 0)), "profitable trades")
        self.metrics[2].update_value(f"{result.get('sharpe', 0):.2f}", "risk adjusted")
        self.metrics[3].update_value(
            self.window.pct(result.get("max_drawdown", 0)),
            "strategy trough",
            self.window.colors["red"],
        )
        self.equity_plot.clear()
        equity = result.get("equity_curve", [])
        baseline = result.get("buy_hold_curve", [])
        if equity:
            self.equity_plot.plot(equity, name="Strategy", pen=pg.mkPen(self.window.colors["teal"], width=3))
        if baseline:
            self.equity_plot.plot(
                baseline,
                name="Buy and hold",
                pen=pg.mkPen(self.window.colors["muted"], width=2, style=QtCore.Qt.DashLine),
            )
        rows = []
        for trade in result.get("trades", [])[-8:][::-1]:
            rows.append([
                pd.to_datetime(trade.get("exit_date")).strftime("%Y-%m-%d"),
                trade.get("symbol", self.window.last_symbol),
                f"{float(trade.get('entry_price', 0)):,.2f}",
                f"{float(trade.get('exit_price', 0)):,.2f}",
                self.window.pct(trade.get("profit_pct", 0), signed=True),
                trade.get("exit_reason", ""),
            ])
        self.window.fill_table(self.trade_table, rows)


class TrainPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Train model")
        body = QtWidgets.QHBoxLayout()
        settings = Card("Training settings")
        form = QtWidgets.QFormLayout()
        self.data_path = QtWidgets.QLineEdit(os.path.join(window.project_root, "historical_data"))
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.data_path, 1)
        path_row.addWidget(browse)
        form.addRow("Data directory", path_row)
        self.rf_estimators = QtWidgets.QSpinBox()
        self.rf_estimators.setRange(50, 500)
        self.rf_estimators.setValue(250)
        self.learning_rate = QtWidgets.QDoubleSpinBox()
        self.learning_rate.setRange(0.01, 0.20)
        self.learning_rate.setSingleStep(0.01)
        self.learning_rate.setValue(0.05)
        self.max_depth = QtWidgets.QSpinBox()
        self.max_depth.setRange(3, 10)
        self.max_depth.setValue(6)
        self.swing_window = QtWidgets.QSpinBox()
        self.swing_window.setRange(3, 20)
        self.swing_window.setValue(10)
        self.swing_threshold = QtWidgets.QDoubleSpinBox()
        self.swing_threshold.setRange(1, 15)
        self.swing_threshold.setSuffix("%")
        self.swing_threshold.setValue(5)
        form.addRow("RF estimators", self.rf_estimators)
        form.addRow("XGBoost learning rate", self.learning_rate)
        form.addRow("XGBoost max depth", self.max_depth)
        form.addRow("Swing window", self.swing_window)
        form.addRow("Swing threshold", self.swing_threshold)
        settings.layout.addLayout(form)
        train = QtWidgets.QPushButton("Train model")
        train.setProperty("primary", True)
        train.clicked.connect(lambda: window.start_training(False))
        settings.layout.addWidget(train)
        body.addWidget(settings, 2)

        artifacts = Card("Model artifacts")
        self.artifact_list = QtWidgets.QListWidget()
        artifacts.layout.addWidget(self.artifact_list)
        body.addWidget(artifacts, 1)
        self.root.addLayout(body)

        log_card = Card("Training log")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        log_card.layout.addWidget(self.progress)
        log_card.layout.addWidget(self.log)
        self.root.addWidget(log_card, 1)

    def browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Historical data", self.data_path.text())
        if path:
            self.data_path.setText(path)

    def refresh(self):
        self.artifact_list.clear()
        for filename, description in self.window.model_artifacts:
            status = "Present" if os.path.exists(os.path.join(self.window.project_root, filename)) else "Missing"
            self.artifact_list.addItem(f"{status:7}  {filename}  -  {description}")
        self.log.setPlainText("\n".join(self.window.training_log[-100:]))
        self.progress.setValue(100 if self.window.training_complete else 20 if self.window.training_running else 0)


class SettingsPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Settings")
        self.action_button("Save", self.save, primary=True)
        card = Card("Application settings")
        form = QtWidgets.QFormLayout()
        self.krx_key = QtWidgets.QLineEdit()
        self.krx_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.alpha_key = QtWidgets.QLineEdit()
        self.alpha_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.market = QtWidgets.QComboBox()
        self.market.addItems(["KR", "US"])
        self.monitor_interval = QtWidgets.QSpinBox()
        self.monitor_interval.setRange(30, 86400)
        self.monitor_interval.setValue(window.monitor_interval_seconds)
        form.addRow("KRX service key", self.krx_key)
        form.addRow("Alpha Vantage key", self.alpha_key)
        form.addRow("Default market", self.market)
        form.addRow("Monitor interval (seconds)", self.monitor_interval)
        card.layout.addLayout(form)
        self.root.addWidget(card)

        watch = Card("Watchlist")
        row = QtWidgets.QHBoxLayout()
        self.watchlist = QtWidgets.QListWidget()
        controls = QtWidgets.QVBoxLayout()
        add = QtWidgets.QPushButton("Add symbol")
        remove = QtWidgets.QPushButton("Remove selected")
        add.clicked.connect(self.add_symbol)
        remove.clicked.connect(self.remove_symbol)
        controls.addWidget(add)
        controls.addWidget(remove)
        controls.addStretch()
        row.addWidget(self.watchlist, 1)
        row.addLayout(controls)
        watch.layout.addLayout(row)
        self.root.addWidget(watch, 1)

    def refresh(self):
        self.krx_key.setText(self.window.backend.SecretManager.get_optional_secret("KRX_SERVICE_KEY") or "")
        self.alpha_key.setText(self.window.backend.SecretManager.get_optional_secret("ALPHA_VANTAGE_API_KEY") or "")
        self.market.setCurrentText(self.window.market_mode)
        self.monitor_interval.setValue(self.window.monitor_interval_seconds)
        self.watchlist.clear()
        self.watchlist.addItems(self.window.watchlist)

    def save(self):
        self.window.save_settings(
            self.krx_key.text().strip(),
            self.alpha_key.text().strip(),
            self.market.currentText(),
            self.monitor_interval.value(),
        )

    def add_symbol(self):
        symbol, accepted = QtWidgets.QInputDialog.getText(self, "Add symbol", "Symbol")
        symbol = symbol.strip().upper()
        if accepted and symbol and symbol not in self.window.watchlist:
            self.window.watchlist.append(symbol)
            self.window.save_config()
            self.refresh()

    def remove_symbol(self):
        row = self.watchlist.currentRow()
        if row >= 0:
            self.window.watchlist.pop(row)
            self.window.save_config()
            self.refresh()


class DocsPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Docs")
        self.action_button("Open README", self.open_readme)
        card = Card("README")
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        card.layout.addWidget(self.text)
        self.root.addWidget(card, 1)

    def refresh(self):
        try:
            with open(os.path.join(self.window.project_root, "README.md"), "r", encoding="utf-8") as readme:
                self.text.setPlainText(readme.read())
        except OSError as error:
            self.text.setPlainText(str(error))

    def open_readme(self):
        webbrowser.open(os.path.join(self.window.project_root, "README.md"))


class TradingTerminalWindow(QtWidgets.QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.colors = backend.COLORS
        self.project_root = backend.PROJECT_ROOT
        self.model_artifacts = backend.MODEL_ARTIFACTS
        self.config_path = backend.UI_CONFIG_PATH
        self.system = backend.SwingTradingSystem(
            api_key=backend.SecretManager.get_optional_secret("ALPHA_VANTAGE_API_KEY")
        )
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.workers = set()
        self.market_history_cache = {}
        self.market_history_lock = threading.Lock()
        self.busy_tasks = set()
        self.monitoring = False
        self.monitor_interval_seconds = 300
        self.training_running = False
        self.training_complete = False
        self.training_log = []
        self.last_analysis = None
        self.last_backtest = None
        self.config_data = self.load_config()
        self.watchlist = self.config_data.get("watchlist") or self.load_watchlist()
        self.market_mode = self.config_data.get("market_mode") or (
            "US" if any(any(char.isalpha() for char in symbol) for symbol in self.watchlist) else "KR"
        )
        self.last_symbol = self.config_data.get("last_symbol") or (
            self.watchlist[0] if self.watchlist else ("005930" if self.market_mode == "KR" else "AAPL")
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
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {self.colors["bg"]};
                color: {self.colors["text"]};
                font-family: "Segoe UI";
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
            QLabel#brand {{ font-size: 17px; font-weight: 600; color: {self.colors["teal"]}; }}
            QLabel#pageTitle {{ font-size: 23px; font-weight: 600; }}
            QLabel#cardTitle {{ font-size: 14px; font-weight: 600; }}
            QLabel#metricValue {{ font-size: 22px; font-weight: 600; }}
            QLabel#muted {{ color: {self.colors["muted"]}; }}
            QPushButton {{
                background: {self.colors["panel_alt"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QPushButton:hover {{ background: {self.colors["subtle"]}; }}
            QPushButton[primary="true"] {{
                color: white;
                background: {self.colors["teal"]};
                border-color: {self.colors["teal"]};
            }}
            QPushButton[nav="true"] {{
                text-align: left;
                border: none;
                padding: 10px 12px;
                color: {self.colors["muted"]};
            }}
            QPushButton[nav="true"]:checked {{
                color: {self.colors["teal"]};
                background: {self.colors["teal_soft"]};
                font-weight: 600;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget, QTableWidget {{
                background: {self.colors["panel"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 7px;
                padding: 6px;
                selection-background-color: {self.colors["teal"]};
            }}
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

    def refresh_sidebar(self):
        self.market_button.setText(self.market_label())
        status = self.artifact_status()
        self.model_status.setText(
            f"Model {'ready' if status['ready'] else 'missing'}  |  {status['present_count']}/{len(self.model_artifacts)}"
        )

    def toggle_market(self):
        self.market_mode = "US" if self.market_mode == "KR" else "KR"
        self.last_symbol = "AAPL" if self.market_mode == "US" else "005930"
        self.system.detector = None
        self.clear_market_cache()
        self.save_config()
        self.pages["Analyze"].market.setCurrentText(self.market_mode)
        self.pages["Analyze"].symbol.setText(self.last_symbol)
        self.refresh_sidebar()

    def market_label(self):
        return "KR market" if self.market_mode == "KR" else "US market"

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
            rows.append([
                signal.get("symbol", ""),
                signal.get("company", signal.get("symbol", "")),
                self.pct(signal.get("probability", 0)) if available else "--",
                signal.get("confidence", "--"),
            ])
        self.fill_table(table, rows)

    def style_plot(self, plot):
        plot.setBackground(self.colors["panel"])
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.getPlotItem().hideButtons()

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

    def current_page_name(self):
        current = self.stack.currentWidget()
        return next(name for name, page in self.pages.items() if page is current)

    def refresh_dashboard(self):
        self.run_background("dashboard", self.compute_dashboard_data, self.dashboard_ready)

    def dashboard_ready(self, data):
        self.dashboard_data = data
        for name in ["Home", "Monitor", "Screener"]:
            self.pages[name].refresh()
        self.refresh_sidebar()

    def start_analysis(self, symbol, mode):
        if mode != self.market_mode:
            self.system.detector = None
            self.clear_market_cache()
        self.market_mode = mode
        self.last_symbol = symbol
        self.add_alert("system", f"Started analysis for {symbol}.")
        self.run_background("analysis", lambda: self.analysis_task(symbol, mode), self.analysis_ready)

    def analysis_ready(self, payload):
        self.last_analysis = payload
        result = payload["result"]
        self.add_alert(
            "signal" if result.get("swing_probability", 0) >= 0.65 else "system",
            f"{result['symbol']} probability {self.pct(result.get('swing_probability', 0))}.",
        )
        self.save_config()
        self.pages["Analyze"].refresh()

    def start_backtest(self, symbol, days, threshold):
        self.last_symbol = symbol
        self.run_background(
            "backtest",
            lambda: self.live_backtest(symbol, days, threshold),
            self.backtest_ready,
        )

    def backtest_ready(self, result):
        self.last_backtest = result
        self.add_alert("system", f"Backtest completed for {result.get('symbol', self.last_symbol)}.")
        self.pages["Backtest"].refresh()

    def start_training(self, analyze_after=False):
        if self.training_running:
            return
        page = self.pages["Train model"]
        parameters = {
            "data_path": page.data_path.text(),
            "swing_threshold": page.swing_threshold.value() / 100,
            "swing_window": page.swing_window.value(),
            "rf_estimators": page.rf_estimators.value(),
            "learning_rate": page.learning_rate.value(),
            "max_depth": page.max_depth.value(),
        }
        self.training_running = True
        self.training_complete = False
        self.training_log.append(f"[{self.now_time()}] Training job started.")
        self.pages["Train model"].refresh()
        self.run_background(
            "training",
            lambda: self.training_task(parameters),
            lambda result: self.training_ready(result, analyze_after),
        )

    def training_ready(self, result, analyze_after=False):
        self.training_running = False
        self.training_complete = True
        for line in result["log"].splitlines()[-80:]:
            self.training_log.append(f"[{self.now_time()}] {line}")
        self.training_log.append(f"[{self.now_time()}] Training completed with score {result['score']:.4f}.")
        self.system.detector = None
        self.add_alert("system", f"Model retraining completed with score {result['score']:.3f}.")
        self.pages["Train model"].refresh()
        self.refresh_sidebar()
        if analyze_after:
            self.show_screen("Analyze")

    def compute_dashboard_data(self):
        signals = []
        watch_items = []
        displayed_dates = []
        detector = None
        for symbol in self.watchlist:
            try:
                if detector is None:
                    detector = self.ensure_detector(self.market_mode)
                with contextlib.redirect_stdout(io.StringIO()):
                    df = self.fetch_market_history(symbol, self.market_mode)
                if df is None or len(df) < 2:
                    raise ValueError("No live data")
                as_of = self.date_from_index(df.index[-1])
                if as_of:
                    displayed_dates.append(as_of)
                price = float(df.iloc[-1]["close"])
                previous = float(df.iloc[-2]["close"])
                change = (price - previous) / previous if previous else 0
                watch_items.append({
                    "symbol": symbol,
                    "price": price,
                    "change": change,
                    "currency": "KRW" if self.market_mode == "KR" else "USD",
                })
                result = None
                if len(df) >= 100:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = detector.detect_swing_opportunity(df, symbol)
                result = result or self.fallback_analysis(symbol, df, self.market_mode)
                signals.append({
                    "symbol": symbol,
                    "company": self.company_name(symbol),
                    "probability": float(result.get("swing_probability", 0)),
                    "confidence": result.get("confidence_level", "Low"),
                })
            except Exception as error:
                signals.append({
                    "symbol": symbol,
                    "company": self.company_name(symbol),
                    "probability": 0,
                    "confidence": "Unavailable",
                    "status": "unavailable",
                    "message": str(error),
                })
        signals.sort(key=lambda item: (item.get("status") == "unavailable", -item.get("probability", 0)))
        available = [item["probability"] for item in signals if item.get("status") != "unavailable"]
        freshness = self.display_freshness(displayed_dates)
        self.data_freshness = freshness
        status = self.artifact_status()
        return {
            "model_ready": status["ready"],
            "ensemble": self.ensemble_label(),
            "signals": signals,
            "watchlist": watch_items,
            "avg_probability": float(np.mean(available)) if available else 0,
            "signals_today": len([value for value in available if value >= 0.65]),
            "artifact_count": status["present_count"],
            "data_freshness": freshness,
        }

    def analysis_task(self, symbol, mode):
        result = None
        df = None
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                detector = self.ensure_detector(mode)
                df = self.fetch_market_history(symbol, mode)
                if df is not None and len(df) >= 50:
                    result = detector.detect_swing_opportunity(df, symbol)
                    if result:
                        result["currency"] = "KRW" if mode == "KR" else "USD"
        except Exception as error:
            log.write(f"Provider analysis failed: {error}\n")
            if df is None:
                raise
        if result is None and df is not None:
            result = self.fallback_analysis(symbol, df, mode)
        if result is None:
            raise ValueError(f"Could not analyze {symbol}. Check API keys and symbol format.")
        return {"result": result, "df": df, "company": self.company_name(symbol), "log": log.getvalue()}

    def live_backtest(self, symbol, days_back, threshold):
        df = self.fetch_market_history(
            symbol,
            self.market_mode,
            outputsize="full" if self.market_mode == "US" else "compact",
        )
        if df is None or len(df) < 70:
            raise ValueError(f"No usable provider history for {symbol}.")
        df_recent = df.tail(days_back).copy()
        with contextlib.redirect_stdout(io.StringIO()):
            detector = self.ensure_detector(self.market_mode)
            features = self.backend.TechnicalIndicators.create_all_indicators(df_recent)
        features = features.ffill().bfill().replace([np.inf, -np.inf], 0)
        for feature in detector.feature_columns:
            if feature not in features.columns:
                features[feature] = 0
        trades = []
        position = None
        equity = [1.0]
        for index in range(50, len(features) - 10):
            row = features.iloc[index]
            date = features.index[index]
            price = float(row["close"])
            try:
                sample = features.iloc[index:index + 1][detector.feature_columns]
                probability = float(detector.model.predict_proba(detector.scaler.transform(sample))[0][1])
            except Exception:
                previous = float(features.iloc[max(0, index - 5)]["close"])
                momentum = (price - previous) / previous if previous else 0
                probability = self.fallback_probability(
                    momentum,
                    float(row.get("rsi_14", 50)),
                    float(row.get("volume_ratio", 1)),
                )
            if position is None and probability >= threshold:
                stop, take = detector._calculate_stop_take_profit(
                    price,
                    float(row.get("atr_14", 0)),
                    getattr(detector, "swing_threshold", 0.15),
                )
                position = {
                    "entry_date": date,
                    "entry_price": price,
                    "entry_probability": probability,
                    "stop_loss": stop,
                    "take_profit": take,
                }
            elif position is not None:
                days_held = (date - position["entry_date"]).days
                profit = (price - position["entry_price"]) / position["entry_price"]
                reason = None
                if price <= position["stop_loss"]:
                    reason = "Stop-loss"
                elif price >= position["take_profit"]:
                    reason = "Take-profit"
                elif days_held >= 10:
                    reason = "Max time"
                if reason:
                    trades.append({
                        "entry_date": position["entry_date"],
                        "exit_date": date,
                        "symbol": symbol,
                        "entry_price": position["entry_price"],
                        "exit_price": price,
                        "days_held": days_held,
                        "profit_pct": profit,
                        "entry_probability": position["entry_probability"],
                        "exit_reason": reason,
                    })
                    equity.append(equity[-1] * (1 + profit))
                    position = None
        profits = [trade["profit_pct"] for trade in trades]
        returns = np.asarray(profits or [0.0])
        first = float(df_recent.iloc[0]["close"])
        return {
            "symbol": symbol,
            "trades": trades,
            "win_rate": len([profit for profit in profits if profit > 0]) / len(profits) if profits else 0,
            "total_return": equity[-1] - 1,
            "num_trades": len(trades),
            "sharpe": float(np.mean(returns) / np.std(returns) * math.sqrt(12)) if np.std(returns) else 0,
            "max_drawdown": self.max_drawdown(equity),
            "equity_curve": equity,
            "buy_hold_curve": [float(price) / first for price in df_recent["close"]] if first else [],
        }

    def training_task(self, parameters):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            trainer = self.backend.SwingTradeTrainer(
                swing_threshold=parameters["swing_threshold"],
                lookforward_periods=parameters["swing_window"],
            )
            trainer.random_forest_model.set_params(n_estimators=parameters["rf_estimators"])
            trainer.xgboost_model.set_params(
                learning_rate=parameters["learning_rate"],
                max_depth=parameters["max_depth"],
            )
            score = trainer.train(parameters["data_path"])
            trainer.save_model()
        return {"score": score, "log": log.getvalue(), "stats": trainer.training_stats}

    def ensure_detector(self, mode):
        mode = self.backend.SecurityValidator.validate_choice(mode, "stock_mode", {"US", "KR"})
        if mode == "US":
            self.system.api_key = self.backend.SecretManager.get_optional_secret("ALPHA_VANTAGE_API_KEY")
            if not self.system.api_key:
                raise ValueError("ALPHA_VANTAGE_API_KEY is required for live US market data.")
        if self.system.detector is None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.system.initialize_detector(stock_mode=mode)
        return self.system.detector

    def fetch_market_history(self, symbol, mode, outputsize="compact"):
        normalized = self.backend.SecurityValidator.validate_symbol(
            symbol, mode, self.system.security_config
        )
        key = (mode, normalized, outputsize)
        cached = self.get_cached_history(key)
        if cached is not None:
            return cached
        detector = self.ensure_detector(mode)
        if mode == "US":
            df = detector.fetch_alpha_vantage_data(
                normalized, function="TIME_SERIES_DAILY", outputsize=outputsize
            )
        else:
            df = detector.fetch_korean_stock(normalized)
        if df is None or df.empty:
            return None
        df = df.sort_index()
        with self.market_history_lock:
            self.market_history_cache[key] = (time.time(), df.copy())
        return df.copy()

    def get_cached_history(self, key):
        with self.market_history_lock:
            cached = self.market_history_cache.get(key)
            if not cached:
                return None
            cached_at, df = cached
            ttl = (
                self.backend.ALPHA_VANTAGE_CACHE_TTL_SECONDS
                if key[0] == "US"
                else self.backend.LIVE_MARKET_CACHE_TTL_SECONDS
            )
            if time.time() - cached_at >= ttl:
                self.market_history_cache.pop(key, None)
                return None
            return df.copy()

    def clear_market_cache(self):
        with self.market_history_lock:
            self.market_history_cache.clear()

    def fallback_analysis(self, symbol, df, mode):
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        price = float(latest.get("close", 0))
        previous_price = float(previous.get("close", price))
        change = (price - previous_price) / previous_price if previous_price else 0
        recent = df.tail(15)
        start = float(recent.iloc[0]["close"])
        momentum = (float(recent.iloc[-1]["close"]) - start) / start if start else 0
        average_volume = float(df.tail(30)["volume"].mean()) if "volume" in df else 1
        current_volume = float(latest.get("volume", 0))
        volume_ratio = current_volume / average_volume if average_volume else 1
        gains = df["close"].diff().clip(lower=0).tail(14).mean()
        losses = (-df["close"].diff().clip(upper=0)).tail(14).mean()
        rsi = 100 - (100 / (1 + gains / losses)) if losses else 50
        probability = self.fallback_probability(momentum, rsi, volume_ratio)
        swing = 0.08 if mode == "US" else 0.05
        return {
            "symbol": symbol,
            "current_price": price,
            "current_volume": current_volume,
            "price_change_1d": change,
            "swing_probability": probability,
            "confidence_level": "High" if probability >= 0.78 else "Medium" if probability >= 0.6 else "Low",
            "stop_loss": price * (1 - swing * 0.5),
            "take_profit": price * (1 + swing),
            "currency": "KRW" if mode == "KR" else "USD",
        }

    @staticmethod
    def fallback_probability(momentum, rsi, volume_ratio):
        momentum_score = max(-0.2, min(0.2, momentum)) * 1.7
        rsi_score = max(-0.1, min(0.16, (50 - abs(float(rsi) - 55)) / 300))
        volume_score = max(-0.1, min(0.15, (float(volume_ratio) - 1) * 0.08))
        return max(0.05, min(0.95, 0.48 + momentum_score + rsi_score + volume_score))

    def save_settings(self, krx_key, alpha_key, market, interval):
        lines = [
            "# data.go.kr service key used for Korean market lookups",
            f"KRX_SERVICE_KEY={krx_key}",
            "",
            "# Optional: only needed for US market lookups",
            f"ALPHA_VANTAGE_API_KEY={alpha_key}",
        ]
        with open(self.backend.ENV_FILE_PATH, "w", encoding="utf-8") as env_file:
            env_file.write("\n".join(lines) + "\n")
        os.environ["KRX_SERVICE_KEY"] = krx_key
        os.environ["ALPHA_VANTAGE_API_KEY"] = alpha_key
        self.market_mode = market
        self.monitor_interval_seconds = interval
        self.system.api_key = alpha_key or None
        self.system.detector = None
        self.clear_market_cache()
        self.save_config()
        self.add_alert("system", "Settings saved.")
        self.refresh_sidebar()
        QtWidgets.QMessageBox.information(self, "Settings", "Settings saved.")

    def export_backtest(self):
        if not self.last_backtest:
            return
        path = os.path.join(
            self.project_root,
            f"{self.last_backtest.get('symbol', 'backtest')}_backtest.json",
        )
        with open(path, "w", encoding="utf-8") as output:
            json.dump(self.last_backtest, output, indent=2, default=str)
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
            "watchlist": self.watchlist,
            "alerts": self.alerts[:12],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.config_path, "w", encoding="utf-8") as config:
            json.dump(payload, config, indent=2)

    def load_watchlist(self):
        try:
            with open(self.backend.STOCKS_FILE_PATH, "r", encoding="utf-8") as stocks:
                values = json.load(stocks)
            if isinstance(values, list):
                return [str(value).upper() for value in values[:16]]
        except (OSError, ValueError):
            pass
        return ["005930", "000660", "035420"]

    def empty_dashboard_data(self):
        status = self.artifact_status()
        return {
            "model_ready": status["ready"],
            "ensemble": self.ensemble_label(),
            "signals": [],
            "watchlist": [],
            "avg_probability": 0,
            "signals_today": 0,
            "artifact_count": status["present_count"],
            "data_freshness": self.data_freshness,
        }

    def artifact_status(self):
        present = [
            os.path.exists(os.path.join(self.project_root, filename))
            for filename, _description in self.model_artifacts
        ]
        return {"ready": all(present), "present_count": sum(present)}

    def ensemble_label(self):
        try:
            stats = self.backend.joblib.load(self.backend.resource_path("training_stats.pkl"))
            return stats.get("model_type", "hybrid_random_forest_xgboost").replace("_", " ").title()
        except Exception:
            return "RF + XGBoost ensemble"

    def display_freshness(self, dates):
        clean = [value for value in dates if value is not None]
        if not clean:
            return {
                "label": "No market data",
                "detail": "No visible market rows loaded",
                "color": self.colors["red"],
            }
        oldest = min(clean)
        latest = max(clean)
        age = max(0, (datetime.now().date() - oldest).days)
        color = self.colors["green"] if age <= 3 else self.colors["amber"] if age <= 14 else self.colors["red"]
        label = f"As of {latest.isoformat()}" if oldest == latest else f"{oldest.isoformat()} to {latest.isoformat()}"
        return {"label": label, "detail": f"Oldest visible row {age}d", "color": color}

    @staticmethod
    def date_from_index(value):
        try:
            timestamp = pd.to_datetime(value, errors="coerce", utc=True)
            return None if pd.isna(timestamp) else timestamp.date()
        except Exception:
            return None

    @staticmethod
    def company_name(symbol):
        names = {
            "005930": "Samsung Electronics",
            "000660": "SK hynix",
            "035420": "NAVER",
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "NVDA": "NVIDIA",
            "GOOGL": "Alphabet",
            "GOOG": "Alphabet",
            "AMZN": "Amazon",
            "META": "Meta Platforms",
            "TSLA": "Tesla",
            "JPM": "JPMorgan Chase",
        }
        return names.get(symbol.upper(), symbol.upper())

    @staticmethod
    def max_drawdown(equity):
        peak = equity[0] if equity else 1
        maximum = 0
        for value in equity:
            peak = max(peak, value)
            if peak:
                maximum = max(maximum, (peak - value) / peak)
        return maximum

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


def launch_qt_app(backend):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Random Tree Bot")
    app.setStyle("Fusion")
    window = TradingTerminalWindow(backend)
    window.show()
    return app.exec()
