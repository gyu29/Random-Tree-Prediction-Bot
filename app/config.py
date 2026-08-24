import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")
ENV_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, ".env.example")
STOCKS_FILE_PATH = os.path.join(PROJECT_ROOT, "stocks.json")
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "train")
VALIDATION_ROOT = os.path.join(PROJECT_ROOT, "validation")
TEST_ROOT = os.path.join(PROJECT_ROOT, "test")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
UI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "trading_ui_config.json")

ALPHA_VANTAGE_MIN_INTERVAL_SECONDS = 12.0
ALPHA_VANTAGE_CACHE_TTL_SECONDS = 12 * 60 * 60
LIVE_MARKET_CACHE_TTL_SECONDS = 15 * 60

DEFAULT_SWING_THRESHOLD = 0.15
DEFAULT_LOOKFORWARD_PERIODS = 10
DEFAULT_MIN_HOLD_PERIODS = 3
DEFAULT_DECISION_THRESHOLD = 0.65

DEFAULT_ENV_CONTENT = (
    "# data.go.kr service key used for Korean market lookups\n"
    "KRX_SERVICE_KEY=replace_with_your_data_go_kr_service_key\n"
    "\n"
    "# Optional: only needed for US market lookups\n"
    "# ALPHA_VANTAGE_API_KEY=replace_with_your_alpha_vantage_key\n"
    "\n"
    "# Auto-generated the first time a model is saved. Used to sign model\n"
    "# artifact manifests so tampered/corrupted .pkl files are detected before\n"
    "# they're loaded. Do not share this value or commit it.\n"
    "# MODEL_SIGNING_KEY=\n"
)

# Eight macro-factor categories used for both dataset construction
# (scripts/build_factor_datasets.py) and per-category model training/selection.
# Tickers are category-exclusive (no symbol repeated across categories).
FACTOR_CATEGORIES = {
    "market_beta": ["SPY", "VOO", "IVV", "VTI", "^GSPC", "^DJI", "DIA"],
    "growth_tech": ["QQQ", "XLK", "VGT", "ARKK", "AAPL", "MSFT", "NVDA"],
    "small_cap": ["IWM", "IJR", "VB", "^RUT", "VTWO", "SCHA", "IWO"],
    "international_emerging": ["EEM", "VWO", "EFA", "VEU", "FXI", "IEMG", "ACWX"],
    "credit_conditions": ["HYG", "JNK", "LQD", "BKLN", "EMB", "SJNK", "ANGL"],
    # BIL/GOVT (tradable T-bill/Treasury ETFs) rather than ^IRX/^TNX: those are yield
    # quotes, not prices. ^IRX in particular went briefly negative in 2020 (a real,
    # documented flight-to-safety event) -- computing a percentage return against an
    # entry "price" near or below zero produces meaningless blown-up numbers (a single
    # trade showing -3600% profit, a "compounded return" of -164 billion percent).
    # Nobody could trade the raw yield anyway; a real position here is a bond ETF.
    "rates_recession": ["TLT", "IEF", "SHY", "GOVT", "BIL", "TBT", "IEI"],
    # FXY (Japanese Yen, a traditional safe-haven currency) rather than ^VIX: the raw
    # CBOE Volatility Index isn't a tradable instrument (no ETF tracks the spot index
    # 1:1 -- VIX ETPs track VIX futures and behave very differently, with heavy
    # contango decay), so backtesting entries/exits against its raw level produced
    # nonsensical results (a single-digit-to-80s swing during the 2020 crash showed up
    # as an ~800x compounded "return" on a strategy nobody could have actually traded).
    "inflation_safe_haven": ["GLD", "SLV", "TIP", "IAU", "UUP", "FXY", "SCHP"],
    "energy_commodity": ["USO", "USOI", "XLE", "DBC", "XOP", "UNG", "CVX"],
}

# Reverse lookup: ticker -> category, for auto-selecting a model given a symbol.
SYMBOL_TO_CATEGORY = {
    symbol.lstrip("^"): category
    for category, symbols in FACTOR_CATEGORIES.items()
    for symbol in symbols
}

# Per-category overrides for swing_threshold and decision_threshold, calibrated against
# a real train/validation/test evaluation rather than guessed (see
# MODEL_CALIBRATION_FINDINGS.md for the full investigation and numbers). The uniform
# DEFAULT_SWING_THRESHOLD (0.15) turned out miscalibrated for lower-volatility
# categories -- e.g. credit_conditions had so few genuine 15%-in-10-days moves that its
# model never once reached a usable decision_threshold on the entire test set. A
# category absent from either dict just uses the global default -- energy_commodity
# needed neither override. scripts/train_all_categories.py applies both automatically,
# so a fresh clone (with freshly-fetched data) reproduces this calibration rather than
# silently retraining everything back to the uncalibrated defaults.
CALIBRATED_SWING_THRESHOLDS = {
    "credit_conditions": 0.05,
    "rates_recession": 0.05,
    "inflation_safe_haven": 0.07,
    "small_cap": 0.08,
    "international_emerging": 0.09,
    "market_beta": 0.08,
}
CALIBRATED_DECISION_THRESHOLDS = {
    "credit_conditions": 0.10,
    "rates_recession": 0.45,
    "inflation_safe_haven": 0.40,
    "small_cap": 0.40,
    "international_emerging": 0.10,
    "market_beta": 0.10,
    "growth_tech": 0.70,
}

KRX_MARKET_ENDPOINTS = {
    "stock": {
        "label": "stock",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo",
        "verified": True,
    },
    "security": {
        "label": "beneficiary security",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getSecuritiesPriceInfo",
        "verified": True,
    },
    "warrant_certificate": {
        "label": "preemptive right certificate",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getPreemptiveRightCertificatePriceInfo",
        "verified": True,
    },
    "warrant_security": {
        "label": "preemptive right security",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getPreemptiveRightSecuritiesPriceInfo",
        "verified": True,
    },
    "etf": {
        "label": "ETF",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getETFPriceInfo",
        "verified": True,
    },
    "etn": {
        "label": "ETN",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getETNPriceInfo",
        "verified": True,
    },
    "elw": {
        "label": "ELW",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getELWPriceInfo",
        "verified": True,
    },
    "bond": {
        "label": "bond",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetBondSecuritiesInfoService/getBondPriceInfo",
        "verified": True,
    },
    "future": {
        "label": "future",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetDerivativesInfoService/getFuturePriceInfo",
        "verified": False,
    },
    "option": {
        "label": "option",
        "guide": "data.go.kr API guide",
        "url": "https://apis.data.go.kr/1160100/service/GetDerivativesInfoService/getOptionPriceInfo",
        "verified": False,
    },
}

KRX_ASSET_TYPE_ALIASES = {
    "stock": "stock",
    "stocks": "stock",
    "equity": "stock",
    "equities": "stock",
    "security": "security",
    "securities": "security",
    "beneficiary": "security",
    "fund": "security",
    "funds": "security",
    "etf": "etf",
    "etn": "etn",
    "elw": "elw",
    "bond": "bond",
    "bonds": "bond",
    "future": "future",
    "futures": "future",
    "option": "option",
    "options": "option",
    "warrant": "warrant_security",
    "warrants": "warrant_security",
    "certificate": "warrant_certificate",
}

KRX_DEFAULT_LOOKUP_ORDER = [
    "stock",
    "etf",
    "etn",
    "elw",
    "security",
    "warrant_security",
    "warrant_certificate",
    "bond",
]


def resource_path(relative_path):
    """Resolve a path that works both from source and inside a PyInstaller bundle."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = PROJECT_ROOT
    return os.path.join(base_path, relative_path)


def ensure_env_file_exists():
    if os.path.exists(ENV_FILE_PATH):
        return

    if os.path.exists(ENV_EXAMPLE_PATH):
        with open(ENV_EXAMPLE_PATH, "r", encoding="utf-8") as example_file:
            env_content = example_file.read()
    else:
        env_content = DEFAULT_ENV_CONTENT

    with open(ENV_FILE_PATH, "w", encoding="utf-8") as env_file:
        env_file.write(env_content)

    print("Created .env file. Update KRX_SERVICE_KEY before requesting Korean market data.")


def load_env_file():
    if not os.path.exists(ENV_FILE_PATH):
        return

    with open(ENV_FILE_PATH, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def read_env_file_value(target_key):
    """Read a single key directly from the project .env file."""
    if not os.path.exists(ENV_FILE_PATH):
        return None

    with open(ENV_FILE_PATH, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if key != target_key:
                continue
            return value.strip().strip('"').strip("'")

    return None


def write_env_value(target_key, value):
    """Set (or add) a single key in the project .env file, preserving other lines."""
    lines = []
    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, "r", encoding="utf-8") as env_file:
            lines = env_file.readlines()

    found = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key == target_key:
            lines[index] = f"{target_key}={value}\n"
            found = True
            break

    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{target_key}={value}\n")

    with open(ENV_FILE_PATH, "w", encoding="utf-8") as env_file:
        env_file.writelines(lines)

    os.environ[target_key] = value


ensure_env_file_exists()
load_env_file()
