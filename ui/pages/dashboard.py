from PySide6 import QtWidgets

from ui.widgets import BasePage, Card


class DashboardPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Dashboard")
        self.action_button("Refresh", window.refresh_dashboard, primary=True)

        self.metrics = self.add_metric_row(
            ["Models trained", "Signals today", "Watchlist size", "Average swing probability"]
        )

        split = QtWidgets.QHBoxLayout()
        signals_card = Card("Top swing signals")
        self.signals = window.make_table(["Symbol", "Company", "Category", "Probability", "Confidence"])
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
        trained = data.get("artifact_count", 0)
        total = data.get("total_categories", 8)
        self.metrics[0].update_value(
            f"{trained}/{total}",
            "Hybrid RF + XGBoost" if trained else "Train a category to begin",
            self.window.colors["teal"] if trained == total else self.window.colors["amber"],
        )
        self.metrics[1].update_value(data.get("signals_today", 0), "active setups")
        freshness = data.get("data_freshness", {})
        self.metrics[2].update_value(
            len(self.window.state.get_watchlist()),
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
