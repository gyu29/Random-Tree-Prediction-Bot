from PySide6 import QtWidgets

from app.data_loader import category_for_symbol, list_categories
from app.security import SecretManager
from ui.widgets import BasePage, Card

AUTO_DETECT = "(auto-detect)"


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
        hint = QtWidgets.QLabel(
            "Category controls which of the 8 trained models scores that symbol in "
            "Monitor/Screener/Home. \"(auto-detect)\" tracks the symbol's category "
            "automatically instead of freezing in a choice made now."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        watch.layout.addWidget(hint)
        row = QtWidgets.QHBoxLayout()
        self.watchlist = QtWidgets.QListWidget()
        controls = QtWidgets.QVBoxLayout()
        add = QtWidgets.QPushButton("Add symbol")
        set_category = QtWidgets.QPushButton("Set category")
        remove = QtWidgets.QPushButton("Remove selected")
        add.clicked.connect(self.add_symbol)
        set_category.clicked.connect(self.set_category)
        remove.clicked.connect(self.remove_symbol)
        controls.addWidget(add)
        controls.addWidget(set_category)
        controls.addWidget(remove)
        controls.addStretch()
        row.addWidget(self.watchlist, 1)
        row.addLayout(controls)
        watch.layout.addLayout(row)
        self.root.addWidget(watch, 1)

    def refresh(self):
        self.krx_key.setText(SecretManager.get_optional_secret("KRX_SERVICE_KEY") or "")
        self.alpha_key.setText(SecretManager.get_optional_secret("ALPHA_VANTAGE_API_KEY") or "")
        self.market.setCurrentText(self.window.market_mode)
        self.monitor_interval.setValue(self.window.monitor_interval_seconds)
        self.watchlist.clear()
        for entry in self.window.state.get_watchlist_entries():
            override = entry.get("category")
            label = override or f"{category_for_symbol(entry['symbol'])} (auto)"
            self.watchlist.addItem(f"{entry['symbol']} — {label}")

    def save(self):
        self.window.save_settings(
            self.krx_key.text().strip(),
            self.alpha_key.text().strip(),
            self.market.currentText(),
            self.monitor_interval.value(),
        )

    def prompt_category(self, symbol, current):
        options = [AUTO_DETECT] + list_categories()
        default_index = options.index(current) if current in options else 0
        choice, accepted = QtWidgets.QInputDialog.getItem(
            self, "Category", f"Category for {symbol}", options, default_index, editable=False
        )
        if not accepted:
            return None, False
        return (None if choice == AUTO_DETECT else choice), True

    def add_symbol(self):
        symbol, accepted = QtWidgets.QInputDialog.getText(self, "Add symbol", "Symbol")
        symbol = symbol.strip().upper()
        if not (accepted and symbol):
            return
        category, accepted = self.prompt_category(symbol, AUTO_DETECT)
        if accepted and self.window.state.append_watchlist(symbol, category):
            self.window.save_config()
            self.refresh()

    def set_category(self):
        row = self.watchlist.currentRow()
        entries = self.window.state.get_watchlist_entries()
        if not (0 <= row < len(entries)):
            return
        symbol = entries[row]["symbol"]
        current = entries[row].get("category") or AUTO_DETECT
        category, accepted = self.prompt_category(symbol, current)
        if accepted and self.window.state.set_watchlist_category_at(row, category):
            self.window.save_config()
            self.refresh()

    def remove_symbol(self):
        row = self.watchlist.currentRow()
        if row >= 0 and self.window.state.remove_watchlist_at(row):
            self.window.save_config()
            self.refresh()
