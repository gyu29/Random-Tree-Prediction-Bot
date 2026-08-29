import pandas as pd
import pyqtgraph as pg
from PySide6 import QtWidgets

from app.data_loader import category_for_symbol, list_categories
from ui.widgets import BasePage, Card


class AnalyzePage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Symbol analysis")
        controls = QtWidgets.QHBoxLayout()
        self.symbol = QtWidgets.QLineEdit(window.last_symbol)
        self.symbol.setPlaceholderText("Ticker or Korean market code")
        self.symbol.editingFinished.connect(self.auto_select_category)
        self.market = QtWidgets.QComboBox()
        self.market.addItems(["KR", "US"])
        self.market.setCurrentText(window.market_mode)
        self.category = QtWidgets.QComboBox()
        self.category.addItems(list_categories())
        self.category.setCurrentText(category_for_symbol(window.last_symbol))
        run = QtWidgets.QPushButton("Analyze")
        run.setProperty("primary", True)
        run.clicked.connect(self.run_analysis)
        controls.addWidget(self.symbol, 1)
        controls.addWidget(self.market)
        controls.addWidget(self.category)
        controls.addWidget(run)
        self.root.addLayout(controls)

        self.metrics = self.add_metric_row(["Symbol", "Current price", "Swing probability", "Volume"])

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
        self.source_notice = QtWidgets.QLabel("")
        self.source_notice.setWordWrap(True)
        self.source_notice.setStyleSheet(f"color: {window.colors['amber']};")
        self.source_notice.setVisible(False)
        levels_card.layout.addWidget(self.source_notice)
        self.levels = QtWidgets.QFormLayout()
        self.level_values = {}
        for name in ["Confidence", "Entry", "Take-profit", "Stop-loss", "Category", "Region"]:
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

    def auto_select_category(self):
        symbol = self.symbol.text().strip().upper()
        if symbol:
            self.category.setCurrentText(category_for_symbol(symbol))

    def run_analysis(self):
        symbol = self.symbol.text().strip().upper()
        mode = self.market.currentText()
        category = self.category.currentText()
        if symbol:
            self.window.start_analysis(symbol, mode, category)

    def refresh(self):
        payload = self.window.state.get_last_analysis()
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
            self.probability_color(result),
        )
        self.metrics[3].update_value(f"{int(result.get('current_volume', 0)):,}", "latest session")
        probability = float(result.get("swing_probability", 0))
        self.probability.setValue(round(probability * 100))

        # Two different reasons this number may not mean what it looks like: there was no
        # model to ask, or there was one and it failed its out-of-sample check. The second
        # is the more dangerous of the two, because the number does come from a model.
        is_heuristic = result.get("source") == "heuristic_no_model"
        validation_warning = result.get("validation_warning")
        self.source_notice.setVisible(bool(is_heuristic or validation_warning))
        if is_heuristic:
            self.source_notice.setText(
                f"⚠ Heuristic estimate, not model-based: {result.get('heuristic_reason', '')}"
            )
        elif validation_warning:
            self.source_notice.setText(f"⚠ Unvalidated model: {validation_warning}")

        # Entry/take-profit/stop-loss read as a trade plan. For a model that failed its
        # out-of-sample check they are arithmetic on a meaningless probability, so they
        # are blanked rather than rendered as levels somebody might act on.
        levels_meaningful = not validation_warning
        values = {
            "Confidence": result.get("confidence_level", "--") if levels_meaningful else "--",
            "Entry": self.window.money(result.get("current_price", 0), currency),
            "Take-profit": (self.window.money(result.get("take_profit", 0), currency)
                            if levels_meaningful else "--"),
            "Stop-loss": (self.window.money(result.get("stop_loss", 0), currency)
                          if levels_meaningful else "--"),
            "Category": result.get("category", "--"),
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

    def probability_color(self, result):
        """confidence_level is symmetric (it's high whenever the model is sure of its
        call, whether that call is yes or no), so it can't drive color on its own --
        an 8% swing probability the model is 91%-sure of is a confident NO, not
        something to paint the same teal as a confident YES."""
        if result.get("validation_warning"):
            return self.window.colors["muted"]
        if result.get("is_swing_opportunity"):
            return self.window.colors["teal"]
        if result.get("confidence_level") == "High":
            return self.window.colors["red"]
        return self.window.colors["muted"]
