from PySide6 import QtWidgets

from ui.widgets import BasePage, Card


class SignalPage(BasePage):
    def __init__(self, window, title, button_text):
        super().__init__(window, title)
        self.action_button(button_text, window.refresh_dashboard, primary=True)
        card = Card("Live watchlist signals")
        self.status = QtWidgets.QLabel("Idle")
        self.status.setObjectName("muted")
        card.layout.addWidget(self.status)
        self.table = window.make_table(["Symbol", "Company", "Category", "Probability", "Confidence"])
        card.layout.addWidget(self.table)
        self.root.addWidget(card, 1)

    def refresh(self):
        watchlist = self.window.state.get_watchlist()
        self.status.setText(
            f"{len(watchlist)} symbols, {self.window.dashboard_data.get('data_freshness', {}).get('label', 'not refreshed')}"
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
