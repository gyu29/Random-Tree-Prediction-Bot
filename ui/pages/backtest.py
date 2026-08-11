import os

import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from app import model_registry
from app.config import DEFAULT_DECISION_THRESHOLD
from app.data_loader import DataProcessor, category_for_symbol, list_categories
from ui.widgets import BasePage, Card, silence_stdout


class BacktestPage(BasePage):
    # (label, days) pairs always available as long as the file has enough history for them.
    FIXED_TIME_CANDIDATES = [("90 days", 90), ("180 days", 180), ("1 year", 365), ("3 years", 1095)]
    # Shown before any file is picked, or if the file's date range can't be read.
    DEFAULT_TIME_OPTIONS = ["90 days", "180 days", "1 year", "3 years", "5 years", "10 years"]

    def __init__(self, window):
        super().__init__(window, "Backtest")
        self.action_button("Export", window.export_backtest)
        self.file_path = None
        self._cached_df = None
        self._cached_df_path = None
        self._cached_df_mtime = None

        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(QtWidgets.QLabel("Data file:"))
        browse = QtWidgets.QPushButton("Choose CSV...")
        browse.clicked.connect(self.browse_file)
        self.file_label = QtWidgets.QLabel("No file selected")
        source_row.addWidget(browse)
        source_row.addWidget(self.file_label, 1)
        self.root.addLayout(source_row)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Model:"))
        self.category = QtWidgets.QComboBox()
        self.category.addItems(list_categories())
        self.category.currentTextChanged.connect(self.sync_threshold_to_model)
        self.window_select = QtWidgets.QComboBox()
        self.window_select.addItems(self.DEFAULT_TIME_OPTIONS)
        self.window_select.setCurrentText("180 days")
        self.threshold = QtWidgets.QComboBox()
        self.threshold.addItems(["5%", "10%", "20%", "30%", "40%", "50%", "60%", "65%", "70%", "80%"])
        run = QtWidgets.QPushButton("Run backtest")
        run.setProperty("primary", True)
        run.clicked.connect(self.run_backtest)
        controls.addWidget(self.category)
        controls.addWidget(self.window_select)
        controls.addWidget(self.threshold)
        controls.addWidget(run)
        controls.addStretch(1)
        self.root.addLayout(controls)

        self.notice = QtWidgets.QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet(f"color: {window.colors['amber']};")
        self.notice.setVisible(False)
        self.root.addWidget(self.notice)

        self.metrics = self.add_metric_row(["Total return", "Win rate", "Sharpe ratio", "Maximum drawdown"])

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

        self.sync_threshold_to_model(self.category.currentText())

    def sync_threshold_to_model(self, category):
        """Snaps the threshold dropdown to the selected model's actual trained
        decision_threshold, instead of a hardcoded guess -- so running a backtest without
        touching this control tests what that model actually does by default. Every
        category happens to share the same 65% today (it isn't a Train-page-adjustable
        parameter), but this stays correct if that ever changes per category."""
        try:
            threshold = model_registry.load(category).training_stats["decision_threshold"]
        except Exception:
            threshold = DEFAULT_DECISION_THRESHOLD
        label = f"{threshold * 100:.0f}%"
        if self.threshold.findText(label) < 0:
            self.threshold.addItem(label)
        self.threshold.setCurrentText(label)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose historical data", "", "CSV files (*.csv)"
        )
        if path:
            self.file_path = path
            symbol = DataProcessor.infer_symbol_from_path(path).upper()
            self.category.setCurrentText(category_for_symbol(symbol))
            self.file_label.setText(f"{symbol} — {os.path.basename(path)}")
            self.update_time_window_options(path)

    def _load_df_cached(self, file_path):
        """Parses file_path at most once per (path, mtime) pair, so choosing a file and
        then running several backtests against it (varying threshold/window) doesn't
        re-read and re-validate the same CSV from disk every time. Invalidated by mtime,
        not just path, so an external edit to the file between runs is still picked up."""
        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = None
        if self._cached_df_path != file_path or self._cached_df_mtime != mtime:
            with silence_stdout():
                self._cached_df = DataProcessor.load_and_validate_data(file_path)
            self._cached_df_path = file_path
            self._cached_df_mtime = mtime
        return self._cached_df

    def compute_time_window_options(self, file_path):
        """Only offer windows the file can actually support, plus 5-year steps up to its full span."""
        try:
            df = self._load_df_cached(file_path)
            span_days = (df.index.max() - df.index.min()).days
        except Exception:
            return list(self.DEFAULT_TIME_OPTIONS)
        options = [label for label, days in self.FIXED_TIME_CANDIDATES if days <= span_days]
        max_5yr_step = int((span_days / 365.25) // 5) * 5
        options.extend(f"{years} years" for years in range(5, max_5yr_step + 1, 5))
        return options or ["90 days"]

    def update_time_window_options(self, file_path):
        previous = self.window_select.currentText()
        options = self.compute_time_window_options(file_path)
        self.window_select.blockSignals(True)
        self.window_select.clear()
        self.window_select.addItems(options)
        if previous in options:
            self.window_select.setCurrentText(previous)
        elif "180 days" in options:
            self.window_select.setCurrentText("180 days")
        else:
            self.window_select.setCurrentText(options[-1])
        self.window_select.blockSignals(False)

    @staticmethod
    def window_label_to_days(label):
        fixed = dict(BacktestPage.FIXED_TIME_CANDIDATES)
        if label in fixed:
            return fixed[label]
        years = int(label.split()[0])
        return years * 365

    def run_backtest(self):
        if not self.file_path:
            QtWidgets.QMessageBox.warning(self, "Backtest", "Choose a CSV file first.")
            return
        symbol = DataProcessor.infer_symbol_from_path(self.file_path).upper()
        days = self.window_label_to_days(self.window_select.currentText())
        threshold = float(self.threshold.currentText().replace("%", "")) / 100
        try:
            cached_df = self._load_df_cached(self.file_path)
        except Exception:
            # Let backtest_task re-parse (and surface a proper error dialog) on the
            # worker thread instead, exactly as if no cache existed.
            cached_df = None
        self.window.start_backtest(self.category.currentText(), symbol, self.file_path, days, threshold, df=cached_df)

    def refresh(self):
        result = self.window.state.get_last_backtest()
        if not result:
            return
        symbol = result.get("symbol", self.window.last_symbol)
        if self.file_path:
            self.file_label.setText(f"{symbol} — {os.path.basename(self.file_path)}")
        no_trades_reason = result.get("no_trades_reason")
        self.notice.setText(no_trades_reason or "")
        self.notice.setVisible(bool(no_trades_reason))
        no_trades_color = (
            self.window.colors["amber"] if no_trades_reason
            else self.window.colors["teal"] if result.get("total_return", 0) >= 0
            else self.window.colors["red"]
        )
        self.metrics[0].update_value(
            self.window.pct(result.get("total_return", 0), signed=True),
            f"{result.get('num_trades', 0)} trades",
            no_trades_color,
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
