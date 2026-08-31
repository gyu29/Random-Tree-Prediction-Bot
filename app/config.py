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
#
# 36 funds trading under $5M a day were removed in 2026-08-30. Not on cost grounds -- a
# per-trade cost is the same for every trade and cancels out of the separation statistic,
# which is a difference of two means -- but because a position in a fund turning over
# $41,000 a day (PSCU) cannot be taken at size, so returns measured on it are not
# returns anybody could have had. They cost up to 44% of a category's trade count and
# almost none of its independent information: effective independent series barely moved
# for any category (small_cap 1.17 to 1.05, energy_commodity 2.13 to 1.92, the rest
# unchanged), which says they were duplicating the liquid names rather than adding to
# them. Cheap in the terms that decide significance, and it makes every figure realizable.
FACTOR_CATEGORIES = {
    # Broad US equity, spread across sectors and factor styles instead of seven
    # wrappers around the same index.
    "market_beta": [
        "^GSPC", "^DJI", "DIA", "RSP", "XLF", "XLV", "XLI", "XLY", "XLP", "XLU",
        "XLB", "XLRE", "XLC", "MTUM", "QUAL", "USMV", "VLUE", "SPLV", "OEF",
        "IWB", "IWV", "VONE", "SCHX", "SPYD",
    ],
    # Survivorship-biased by construction -- see the note above. The single names are
    # here because the category was defined around them, not because a clean version of
    # this list exists.
    "growth_tech": [
        "QQQ", "XLK", "VGT", "ARKK", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
        "AVGO", "AMD", "CRM", "ADBE", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM",
        "MU", "AMAT", "SOXX", "SMH", "IGV", "SKYY", "HACK",
    ],
    # Small cap by sector (the PSC* family) and by style, not seven Russell 2000 clones.
    "small_cap": [
        "IWM", "IJR", "VB", "^RUT", "VTWO", "SCHA", "IWO", "IWN", "IJS", "IJT",
        "VBR", "VBK", "SLYV", "SLYG", "FNDA",
    ],
    # Single-country funds: genuinely different economies rather than seven overlapping
    # aggregates of the same ones.
    "international_emerging": [
        "EEM", "VWO", "EFA", "VEU", "FXI", "IEMG", "ACWX", "EWJ", "EWZ", "EWY",
        "EWT", "EWG", "EWU", "EWC", "EWA", "EWH", "EWS", "EWW", "EZA", "INDA",
        "EIDO", "TUR", "EWL", "EWD", "EWQ", "EWI", "EWP",
    ],
    # Credit quality tiers and maturities, which do not move together the way seven
    # high-yield funds do.
    "credit_conditions": [
        "HYG", "JNK", "LQD", "BKLN", "EMB", "SJNK", "ANGL", "VCIT", "VCSH", "VCLT",
        "IGSB", "IGIB", "SPIB", "SPLB", "USHY", "HYLB", "SHYG", "FALN", "PFF", "PGX",
        "CWB", "PCY", ],
    # BIL/GOVT (tradable T-bill/Treasury ETFs) rather than ^IRX/^TNX: those are yield
    # quotes, not prices. ^IRX in particular went briefly negative in 2020 (a real,
    # documented flight-to-safety event) -- computing a percentage return against an
    # entry "price" near or below zero produces meaningless blown-up numbers (a single
    # trade showing -3600% profit, a "compounded return" of -164 billion percent).
    # Nobody could trade the raw yield anyway; a real position here is a bond ETF.
    # The list now spans the curve from 1-month bills to 25-year zeros.
    "rates_recession": [
        # One fund per distinct exposure, not one per issuer. Widening this category to
        # 24 funds made it *less* informative, not more: SPTL, VGLT, SCHQ and TLT are the
        # same long-Treasury trade four times over (pairwise correlation 0.997), and TBF,
        # TBT and TMV are one inverse trade at three leverages. Ten such duplicates were
        # removed. What is left spans bills, each segment of the curve, zero-coupon,
        # broad, TIPS and inverse.
        #
        # Judged on effective independent series -- n / (1 + (n-1) * mean correlation),
        # which accounts for the whole correlation structure rather than the median pair
        # this project had been quoting -- these 14 score 3.70, against 2.96 for the 24
        # they replace and 3.04 for the original 7, on 66k rows against the original's
        # 36k. Better than both predecessors on both axes.
        "BIL", "SHV", "SHY", "IEI", "IEF", "TLH", "TLT", "ZROZ", "EDV", "GOVT",
        "STIP", "VTIP", "TBT", "TMV",
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
        "FXE", "SPIP",
        "IVOL", ],
    # Energy sub-industries plus the metals and agricultural complexes, rather than
    # four ways to own crude.
    "energy_commodity": [
        "USO", "XLE", "DBC", "XOP", "UNG", "CVX", "XES", "OIH", "IEO",
        "AMLP", "MLPX", "FCG", "BNO", "DBO", "DBA",
        "COPX", "PICK", "GDX", "GDXJ", "SIL",
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
    # 6%, not 8%. Widening the universe changed what these labels describe, and both of
    # the values below were left over from the seven-ticker era.
    #
    # At 8% market_beta's positive rate is 2.90% -- inside the band, but only 31 examples
    # reach the calibration slice, barely over MIN_POSITIVES_FOR_CALIBRATION and few
    # enough that the model fell back to a sigmoid fit. 6% gives 6.53% and 221 positives,
    # enough for isotonic.
    "market_beta": 0.06,
    # 20%, not 15%. growth_tech's universe now holds individual semiconductor and
    # software names alongside the sector funds, and a 15% move inside ten days is
    # ordinary for them: 8.60% of rows qualified, above the 2-8% band the working
    # categories sit in, which makes the label describe a common event rather than a
    # swing. 20% brings it to 4.28%.
    "growth_tech": 0.20,
}
# Every category is gated, so none of these decides anything a user acts on. They are the
# last floors the search produced and are kept so the Backtest page stays reproducible.
CALIBRATED_DECISION_THRESHOLDS = {
    "small_cap": 0.0011,
    "growth_tech": 0.004055,
    "inflation_safe_haven": 0.006068,
    "credit_conditions": 0.012977,
    "international_emerging": 0.015140,
    "energy_commodity": 0.0429,
    "rates_recession": 0.3238,
    "market_beta": 0.371525,
}

# Categories whose model must not be presented as a trading signal.
#
# All eight, as of 2026-08-30. None of them survives a significance test that treats
# trades as the correlated observations they are.
#
# The test is scripts/expected_value_thresholds.py: with calibrated probabilities, find a
# probability floor above which trades demonstrably out-earn those below it. What changed
# is not the models but the standard error. Every earlier version of this file divided by
# a two-sample standard error computed as though each trade were an independent draw.
# They are not. Two or three fire on the same day across correlated symbols -- measured
# within-date correlation runs 0.09 to 0.46 -- and each is held up to lookforward_periods
# bars, overlapping every trade entered during them. Resampling blocks of consecutive
# entry dates instead widens the standard errors by 1.6x to 4.4x, and nothing is left:
#
#   category                separation    t     bar
#   growth_tech               +1.70%     2.09   2.17
#   inflation_safe_haven      +0.71%     1.49   2.00
#   rates_recession           +0.98%     1.09   3.71
#   international_emerging    +0.46%     1.00   2.33
#   energy_commodity          +1.17%     0.75   2.63
#   market_beta               +1.69%     0.72   3.08
#   credit_conditions            --       --     --    top band inverts
#   small_cap                    --       --     --    top band inverts
#
# The separations themselves barely moved. What moved is how much of each is attributable
# to having a few correlated bets rather than many independent ones, and market_beta is
# the clearest case: 286 trades over 116 entry dates, within-date correlation 0.46, so
# what looked like 2.93 standard errors of evidence is 0.66.
#
# Transaction costs were checked and are not the constraint. A per-trade cost is identical
# for every trade and so cancels out of the separation, which is a difference of two means
# -- verified numerically, +1.7114% before and after charging 0.50% a trade. What costs
# decide is whether the trades are worth taking, and the break-even cost per category runs
# 0.38% to 1.54% against realistic round trips of roughly 0.05% to 0.50%. The edges, such
# as they are, clear costs comfortably; they do not clear noise.
#
# A Corwin-Schultz spread estimator was built for this and discarded. It ranked NVDA, at
# $12.7bn of daily turnover, as the most expensive name in the universe and BIL as the
# cheapest, correlating +0.28 with dollar volume where it should correlate strongly
# negative. It was measuring volatility rather than spread, which is the one thing that
# estimator exists to avoid, so its numbers are not in this file.
#
# This supersedes every earlier reading in this file, including several that shipped
# categories. Those were not different models measured honestly; they were these models
# measured against a standard error that assumed away the dependence in the data.
#
# Nothing here says the models have no signal. It says the evidence for it is not
# separable from noise at the sample sizes available, which is a statement about how much
# independent data eight overlapping factor categories can yield, not a verdict on the
# modelling. The lever is more independent observations -- and adding more correlated ETFs
# is not that, since effective independent series per category already runs 1.2 to 3.7
# across 208 tickers.
#
# The gate is deliberately not a refusal: Analyze still scores every category, because
# investigating a model requires being able to run it. What stops is presenting the output
# as actionable -- no alerts, no screener ranking, and a warning on every payload.
_NO_ESTABLISHED_EDGE = (
    "No advantage over ignoring this model is measurable at the sample sizes available "
    "once correlated and overlapping trades stop being counted as independent evidence. "
    "Not a trading signal."
)
CATEGORIES_FAILING_VALIDATION = {
    "credit_conditions": _NO_ESTABLISHED_EDGE,
    "energy_commodity": _NO_ESTABLISHED_EDGE,
    "growth_tech": _NO_ESTABLISHED_EDGE,
    "inflation_safe_haven": _NO_ESTABLISHED_EDGE,
    "international_emerging": _NO_ESTABLISHED_EDGE,
    "market_beta": _NO_ESTABLISHED_EDGE,
    "rates_recession": _NO_ESTABLISHED_EDGE,
    "small_cap": (
        "This model's ranking is inverted at the top of its range -- the trades it rates "
        "highest are the ones that lose money. Its output is not a trading signal."
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
