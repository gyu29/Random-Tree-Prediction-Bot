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
# Cached ^VIX / SPY / ^TNX / ^IRX closes backing app/market_context.py. Regeneratable
# and gitignored, like train/ and models/; scripts/build_factor_datasets.py writes it.
MARKET_CONTEXT_PATH = os.path.join(PROJECT_ROOT, "market_context.csv")

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
#
# The universe is deliberately wide and deliberately made of ETFs.
#
# Wide, because the original seven-per-category list was mostly the same asset written
# down seven times: market_beta held SPY, VOO, IVV, VTI, ^GSPC, ^DJI and DIA at a median
# pairwise daily-return correlation of 0.98, which is one asset's worth of information,
# not seven. That is the leading explanation for market_beta's models ranking no better
# than chance -- there was almost nothing independent to learn from. Each category now
# spans distinct exposures: sectors within the US market, single countries within
# international, credit tiers within credit, maturities within rates, separate metals
# and commodities within their categories, and sector sub-indices within small cap.
#
# ETFs rather than individual stocks, because a list of single names chosen today is a
# list of companies that survived to today. Backtesting on it reports the returns of
# known winners and flatters every number in this project -- the exact failure the rest
# of the pipeline has been cleaned of. ETFs are not immune (funds do close, and a closed
# fund leaves the list the same way a delisted stock does) but the bias is far smaller.
#
# growth_tech is the exception and should be read with that in mind: it carries the
# individual mega-cap names the category was built around, and those are survivors by
# definition -- the semiconductor and software firms that failed over the same decades
# are not in the list and never will be. Its numbers are optimistic by an unknown
# margin. The other seven categories are ETF-only.
#
# SPY is absent on purpose: app/market_context.py uses it as the benchmark every
# relative feature is measured against, so a model scoring SPY would be handed
# excess_return columns that are identically zero and a beta of exactly one.
FACTOR_CATEGORIES = {
    # Broad US equity, spread across sectors and factor styles instead of seven
    # wrappers around the same index.
    "market_beta": [
        "^GSPC", "^DJI", "DIA", "RSP", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU",
        "XLB", "XLRE", "XLC", "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "OEF",
        "IWB", "IWV", "VONE", "SCHX", "SPYD",
    ],
    # Survivorship-biased by construction -- see the note above. The single names are
    # here because the category was defined around them, not because a clean version of
    # this list exists.
    "growth_tech": [
        "QQQ", "XLK", "VGT", "ARKK", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
        "AVGO", "AMD", "CRM", "ADBE", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM",
        "MU", "AMAT", "SOXX", "SMH", "IGV", "XSW", "SKYY", "HACK",
    ],
    # Small cap by sector (the PSC* family) and by style, not seven Russell 2000 clones.
    "small_cap": [
        "IWM", "IJR", "VB", "^RUT", "VTWO", "SCHA", "IWO", "IWN", "IJS", "IJT",
        "VBR", "VBK", "SLYV", "SLYG", "PSCT", "PSCH", "PSCF", "PSCD", "PSCE", "PSCI",
        "PSCM", "PSCC", "PSCU", "DES", "DGRS", "FNDA",
    ],
    # Single-country funds: genuinely different economies rather than seven overlapping
    # aggregates of the same ones.
    "international_emerging": [
        "EEM", "VWO", "EFA", "VEU", "FXI", "IEMG", "ACWX", "EWJ", "EWZ", "EWY",
        "EWT", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWW", "EZA", "INDA",
        "EIDO", "THD", "TUR", "EPOL", "EWL", "EWN", "EWD", "EWQ", "EWI", "EWP",
    ],
    # Credit quality tiers and maturities, which do not move together the way seven
    # high-yield funds do.
    "credit_conditions": [
        "HYG", "JNK", "LQD", "BKLN", "EMB", "SJNK", "ANGL", "VCIT", "VCSH", "VCLT",
        "IGSB", "IGIB", "SPIB", "SPLB", "USHY", "HYLB", "SHYG", "FALN", "PFF", "PGX",
        "CWB", "EMHY", "PCY", "IHY",
    ],
    # BIL/GOVT (tradable T-bill/Treasury ETFs) rather than ^IRX/^TNX: those are yield
    # quotes, not prices. ^IRX in particular went briefly negative in 2020 (a real,
    # documented flight-to-safety event) -- computing a percentage return against an
    # entry "price" near or below zero produces meaningless blown-up numbers (a single
    # trade showing -3600% profit, a "compounded return" of -164 billion percent).
    # Nobody could trade the raw yield anyway; a real position here is a bond ETF.
    # The list now spans the curve from 1-month bills to 25-year zeros.
    "rates_recession": [
        "TLT", "IEF", "SHY", "GOVT", "BIL", "TBT", "IEI", "SHV", "SPTL", "SPTS",
        "SPTI", "VGSH", "VGIT", "VGLT", "SCHO", "SCHR", "SCHQ", "EDV", "ZROZ", "TLH",
        "TBF", "STIP", "VTIP", "TMV",
    ],
    # FXY (Japanese Yen, a traditional safe-haven currency) rather than ^VIX: the raw
    # CBOE Volatility Index isn't a tradable instrument (no ETF tracks the spot index
    # 1:1 -- VIX ETPs track VIX futures and behave very differently, with heavy
    # contango decay), so backtesting entries/exits against its raw level produced
    # nonsensical results (a single-digit-to-80s swing during the 2020 crash showed up
    # as an ~800x compounded "return" on a strategy nobody could have actually traded).
    # Separate metals and separate currencies now, rather than three gold funds.
    "inflation_safe_haven": [
        "GLD", "SLV", "TIP", "IAU", "UUP", "FXY", "SCHP", "SGOL", "PPLT", "PALL",
        "GLTR", "DBP", "FXE", "FXB", "FXF", "FXA", "FXC", "UDN", "LTPZ", "SPIP",
        "IVOL", "CGW",
    ],
    # Energy sub-industries plus the metals and agricultural complexes, rather than
    # four ways to own crude.
    "energy_commodity": [
        "USO", "USOI", "XLE", "DBC", "XOP", "UNG", "CVX", "XES", "OIH", "IEO",
        "IEZ", "PXE", "PXJ", "AMLP", "MLPX", "FCG", "BNO", "UGA", "DBO", "DBA",
        "CORN", "WEAT", "SOYB", "DBB", "COPX", "PICK", "GDX", "GDXJ", "SIL",
    ],
}

# Reverse lookup: ticker -> category, for auto-selecting a model given a symbol.
SYMBOL_TO_CATEGORY = {
    symbol.lstrip("^"): category
    for category, symbols in FACTOR_CATEGORIES.items()
    for symbol in symbols
}

# Per-category overrides for swing_threshold and decision_threshold, calibrated against
# a real train/validation/test evaluation rather than guessed. A category absent from
# either dict just uses the global default -- energy_commodity needs neither override.
# scripts/train_all_categories.py applies both automatically, so a fresh clone (with
# freshly-fetched data) reproduces this calibration rather than silently retraining
# everything back to the uncalibrated defaults.
#
# swing_threshold decides what counts as a positive label, so changing one requires a
# retrain. The uniform DEFAULT_SWING_THRESHOLD (0.15) is miscalibrated for
# lower-volatility categories -- a 15% move inside ten days is a near-nonevent for
# investment-grade credit or Treasuries, so under 0.6% of their training rows ever
# qualified and those models learned to never predict a positive at all. Each override
# below puts its category's positive-label rate in the 2-8% band that the two
# categories needing no override (growth_tech 7.3%, energy_commodity 1.8%) sit in
# naturally. See docs/2026-08-24-calibration-investigation.md for the investigation that found this,
# bearing in mind it predates the current split, feature set and universe.
#
# decision_threshold only converts a predicted probability into a trade, so it can be
# changed on a trained model without retraining (model_registry.update_decision_threshold).
# These values come from scripts/select_thresholds.py, which sweeps candidates against
# validation/ only and requires a candidate to be an interior peak on a shrunk-return
# score before it will pick one. Five of the eight categories have no such peak and keep
# the value they were trained with; that script's module docstring explains the rule and
# its output records which categories are which.
CALIBRATED_SWING_THRESHOLDS = {
    # 3%, not the 5% the other low-volatility categories use: at 5% only 1.78% of
    # credit_conditions' rows were positive and its calibration slice held six of them,
    # below app.ensemble.MIN_POSITIVES_FOR_CALIBRATION, so the model could not be
    # calibrated and was gated for that reason alone. 3% gives a 5.28% positive rate and
    # 78 calibration positives. See app.labeling.SWING_THRESHOLD_FLOOR.
    "credit_conditions": 0.03,
    "rates_recession": 0.05,
    "inflation_safe_haven": 0.07,
    "small_cap": 0.08,
    "international_emerging": 0.09,
    "market_beta": 0.08,
}
CALIBRATED_DECISION_THRESHOLDS = {
    # Computed, not swept: the lowest entry probability from which every marginal-return
    # bin is non-negative (scripts/expected_value_thresholds.py). Calibrated probabilities,
    # so most are small -- a 4% chance of a swing is a high reading against a base rate
    # under 1%. Do not round these to two decimals; 0.0011 becomes 0.00, which would turn
    # the app into an unconditional trader.
    "international_emerging": 0.0161,
    "market_beta": 0.0415,
    "energy_commodity": 0.0429,
    "inflation_safe_haven": 0.1864,
    "credit_conditions": 0.0075,
    # No floor exists for these; all three are gated by CATEGORIES_FAILING_VALIDATION, so
    # the values are inert and only keep the Backtest page reproducible.
    "small_cap": 0.0011,
    "growth_tech": 0.0072,
    "rates_recession": 0.45,
}

# Categories whose model must not be presented as a trading signal.
#
# The test is scripts/expected_value_thresholds.py: with calibrated probabilities, bin
# every trade the model would open with no threshold at all by its entry probability and
# read the marginal return per bin. A category ships only if some probability floor
# exists whose bins are all non-negative and whose bottom bin is excluded -- only, that
# is, if the model's ranking separates profitable trades from unprofitable ones.
#
# Measured 2026-08-30, after removing a look-ahead in the backtest scoring path that had
# been back-filling roughly 10% of every decision window from its own future. Every
# earlier version of these figures was computed through that leak; correcting it moved
# energy_commodity from "inverted" to a usable floor and moved inflation_safe_haven's
# floor by a factor of fifty, so nothing measured before it is quoted here.
#
# Edge over the model-off null on test/, in standard errors, at the floor below:
#
#   ship  international_emerging  floor 1.61%   +2.67 SE   2298 trades
#   ship  credit_conditions       floor 0.75%   +2.16 SE   1094 trades
#   ship  inflation_safe_haven    floor 18.64%  +2.16 SE    419 trades
#   ship  energy_commodity        floor 4.29%   +1.68 SE    636 trades
#   ship  market_beta             floor 4.15%   +1.16 SE    265 trades
#
#   GATE  small_cap        inverted: the trades it rates highest lose money, and at its
#         old floor it is +0.33 SE against its null -- indistinguishable from ignoring it.
#   GATE  rates_recession  inverted.
#   GATE  growth_tech      every bin non-negative from the lowest upward, so no floor
#         excludes anything: a high reading does not indicate a better trade.
# credit_conditions was gated here until 2026-08-30 for being uncalibratable: at a 5%
# swing_threshold only six positive examples fell in its calibration slice. Lowering that
# threshold to 3% -- a real move for investment-grade credit, see
# app.labeling.SWING_THRESHOLD_FLOOR -- gave it 78, and it calibrates. Its marginal-return
# curve is now the cleanest of the eight, rising monotonically from +0.04% to +0.36% per
# trade across probability bands. The +2.11 SE it showed while gated was not an artifact;
# it was a real signal on a scale nobody could read.
#
# Purged walk-forward cross-validation (scripts/walk_forward_cv.py, 5 expanding-window
# folds), re-run on the fixed path. Folds in which the category beat its own null, out of
# the folds where enough trades fired to compare:
#
#   international_emerging  5/5   PR-AUC 0.283 +/- 0.129   ROC-AUC 0.859 +/- 0.015
#   inflation_safe_haven    4/5   PR-AUC 0.169 +/- 0.069   ROC-AUC 0.866 +/- 0.046
#   energy_commodity        3/5   PR-AUC 0.182 +/- 0.072   ROC-AUC 0.845 +/- 0.045
#   credit_conditions       3/5   PR-AUC 0.195 +/- 0.113   ROC-AUC 0.838 +/- 0.030
#   market_beta             2/4   PR-AUC 0.262 +/- 0.207   ROC-AUC 0.909 +/- 0.032
#
# market_beta is the one the folds do not corroborate: two of four is a tie, not a
# majority, and it also carries the weakest test-side edge of the four (+1.16 SE) and by
# far the widest PR-AUC spread across regimes -- 0.26 give or take 0.21 is a number that
# should never be quoted on its own. It ships because the stated criterion is the
# marginal-return curve and it has a floor, but of the four it is the one to drop first
# if the evidence has to be narrowed.
#
# credit_conditions carries a caveat its 3/5 hides, and it is the sharpest thing
# cross-validation has surfaced here. Its folds read +0.70, +2.04, +0.48, -3.32, -0.12
# standard errors. The -3.32 fold validates on the block ending 2020-04-08, which
# contains the March 2020 credit dislocation -- VIX peaked at 82.7 inside it against a
# 17.6 median across the whole series. The model works in ordinary conditions and fails,
# by the widest margin of any fold measured anywhere in this project, in precisely the
# event the category is named for. It ships because the curve is clean, a majority of
# folds beat the null, and it is +2.16 SE over that null on test -- but it should not be
# leaned on during a credit stress event, which is the only time anyone would reach for
# it. That is a modelling problem, not a threshold one.
#
# The pooled sweep independently reselected credit_conditions' 0.75% floor, the second
# category where the two methods converge.
#
# international_emerging is the opposite case and the strongest result in the project:
# it beat its null in every fold, and cross-validation independently selected 1.61% --
# the same threshold the marginal-return curve arrived at. The two methods had disagreed
# wildly everywhere before the look-ahead was removed; here they now agree exactly.
#
# They still disagree elsewhere, and the curve still decides. Pooled folds would put
# market_beta and inflation_safe_haven at 65%, because the pooled sweep asks only whether
# the average trade above a floor is profitable while the curve asks whether every
# probability band above it is -- an average can be carried by one good band. The curve
# also reads the deployed model, where the pooled sweep reads five fold models.
#
# Two gated categories look better under cross-validation than under the curve and are
# worth revisiting: small_cap beat its null in 4 of 5 folds and the pooled sweep found an
# interior peak at 20% scoring +1.96%, while its curve is inverted; growth_tech beat its
# null in 4 of 5. Neither changes the gate, because a category whose ranking cannot
# separate profitable trades from unprofitable ones has nothing to threshold, but both
# are the obvious places to look next.
#
# Re-measure after any retrain. The gate is deliberately not a refusal: Analyze still
# scores these categories, because investigating a model requires being able to run it.
# What stops is presenting the output as actionable -- no alerts, no place in the
# screener ranking, and a warning on every payload (app/detector.py, app/trading_system.py).
CATEGORIES_FAILING_VALIDATION = {
    "small_cap": (
        "This model's ranking is inverted -- the trades it rates highest are the ones "
        "that lose money. Its output is not a trading signal."
    ),
    "rates_recession": (
        "This model's ranking is inverted -- the trades it rates highest are the ones "
        "that lose money. Its output is not a trading signal."
    ),
    "growth_tech": (
        "This model's probability does not rank trades: a high reading does not indicate "
        "a better trade than a low one. Not a signal, whatever the number says."
    ),
}

# Optional per-category pins for scripts/build_factor_datasets.py's calendar split,
# as ("train_end", "validation_end") date strings: rows before train_end go to train,
# rows in [train_end, validation_end) to validation, the rest to test. The script
# derives these from the pooled trading calendar when a category is absent here, which
# is the normal path -- pin a category only when a specific run has to be reproducible,
# since yfinance returns a longer history every time it is called and derived cutoffs
# move with it. See that script's module docstring for why the cutoff is per category
# rather than per symbol.
CALENDAR_SPLIT_CUTOFFS = {}

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
