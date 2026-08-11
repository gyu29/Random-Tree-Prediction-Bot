from PySide6 import QtWidgets

from app import model_registry
from app.config import DEFAULT_LOOKFORWARD_PERIODS, DEFAULT_SWING_THRESHOLD
from app.data_loader import category_symbols_on_disk, category_train_dir, list_categories
from ui.widgets import BasePage, Card


class TrainPage(BasePage):
    def __init__(self, window):
        super().__init__(window, "Train model")
        body = QtWidgets.QHBoxLayout()
        settings = Card("Training settings")
        form = QtWidgets.QFormLayout()
        self.category = QtWidgets.QComboBox()
        self.category.addItems(list_categories())
        self.category.currentTextChanged.connect(self.update_symbol_preview)
        form.addRow("Factor category", self.category)
        self.symbol_preview = QtWidgets.QLabel("")
        self.symbol_preview.setObjectName("muted")
        self.symbol_preview.setWordWrap(True)
        form.addRow("Training symbols", self.symbol_preview)

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
        self.swing_window.setRange(20, 200)
        self.swing_window.setValue(DEFAULT_LOOKFORWARD_PERIODS)
        self.swing_threshold = QtWidgets.QDoubleSpinBox()
        self.swing_threshold.setRange(1, 50)
        self.swing_threshold.setSuffix("%")
        self.swing_threshold.setValue(DEFAULT_SWING_THRESHOLD * 100)
        form.addRow("RF estimators", self.rf_estimators)
        form.addRow("XGBoost learning rate", self.learning_rate)
        form.addRow("XGBoost max depth", self.max_depth)
        form.addRow("Swing window", self.swing_window)
        form.addRow("Swing threshold", self.swing_threshold)
        settings.layout.addLayout(form)
        train = QtWidgets.QPushButton("Train model")
        train.setProperty("primary", True)
        train.clicked.connect(lambda: window.start_training(self.category.currentText(), False))
        settings.layout.addWidget(train)
        body.addWidget(settings, 2)

        artifacts = Card("Trained categories")
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

        self.update_symbol_preview(self.category.currentText())

    def update_symbol_preview(self, category):
        symbols = category_symbols_on_disk(category, split="train")
        directory = category_train_dir(category)
        if symbols:
            self.symbol_preview.setText(f"{', '.join(symbols)}  ({directory})")
        else:
            self.symbol_preview.setText(
                f"No training CSVs found in {directory}. Run scripts/build_factor_datasets.py first."
            )

    def refresh(self):
        self.artifact_list.clear()
        for category in list_categories():
            trained = model_registry.is_trained(category)
            status = "Trained" if trained else "Not trained"
            self.artifact_list.addItem(f"{status:12}  {category}")
        self.log.setPlainText("\n".join(self.window.training_log[-100:]))
        self.progress.setValue(100 if self.window.training_complete else 20 if self.window.training_running else 0)
