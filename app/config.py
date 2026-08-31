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
CALIBRATED_DECISION_THRESHOLDS = {
    # Computed, not swept: the candidate floor whose above/below separation survives a
    # multiplicity-corrected significance test (scripts/expected_value_thresholds.py).
    # Calibrated probabilities, so some are small -- do not round to two decimals, since
    # 0.0020 becomes 0.00 and turns the app into an unconditional trader.
    "growth_tech": 0.0020,
    "energy_commodity": 0.0429,
    "market_beta": 0.4514,
    # No floor survives; these five are gated by CATEGORIES_FAILING_VALIDATION, so the
    # values are inert and only keep the Backtest page reproducible.
    "small_cap": 0.0011,
    "credit_conditions": 0.0075,
    "international_emerging": 0.0161,
    "inflation_safe_haven": 0.1864,
    "rates_recession": 0.3238,
}

# Categories whose model must not be presented as a trading signal.
#
# A category ships only if a probability floor exists whose bins all earn non-negative
# marginal returns AND above which trades demonstrably out-earn those below it
# (scripts/expected_value_thresholds.py). The floor is chosen by searching the candidate
# bin edges for the largest separation, so the significance bar is Bonferroni-corrected
# for that search -- 2.2 to 2.5 standard errors depending on how many candidates a
# category offers, rather than the 2.0 a single pre-chosen test would need.
#
# That rule is stricter than the one it replaces, in both directions. The old version
# required the bottom bin to be *losing* money, which threw away any model whose trades
# were all profitable but very unevenly so -- growth_tech earns +1.23% per trade above
# its floor against +0.61% below, a ranking that plainly works and that the old rule
# rejected outright. It also never made a floor prove it beat having no floor, which
# several categories turn out not to.
#
# Measured 2026-08-30. Separation is above-floor minus below-floor return in standard
# errors, on validation, with that category's corrected bar in brackets:
#
#   ship  market_beta       floor 45.14%   test +0.87%/trade over  308 trades
#   ship  growth_tech       floor 0.20%    test +1.23%/trade over 3018 trades
#   ship  energy_commodity  floor 4.29%    test +1.21%/trade over  638 trades
#
#   GATE  international_emerging  2.35 [2.39] -- misses by four hundredths of a standard
#         error, with five of five cross-validation folds beating its null behind it.
#         Read as not established rather than absent: Bonferroni assumes independent
#         tests and these candidate floors are nested, so the correction is conservative
#         here. First category to revisit.
#   GATE  inflation_safe_haven    2.15 [2.33]
#   GATE  credit_conditions       2.07 [2.45]
#   GATE  rates_recession         1.30 [2.24]
#   GATE  small_cap  no floor at any level: its top band inverts, marginal return falling
#         as predicted probability rises.
#
# Three categories that shipped under the old rule no longer do, including the one with
# the strongest independent evidence. That is the rule getting stricter, not the models
# getting worse. The honest reading is that most of these edges were never established --
# they were reported at a bar that did not account for having gone looking for them.
#
# small_cap is the one failing on shape rather than significance. On validation its curve
# is close to textbook, rising to +5.66% per trade on an 81.7% win rate in the 19-53%
# band, and then the top band collapses to -2.50%. A band rather than a floor was tried,
# since the shape invites it: on validation it beats taking every trade, +1.12% against
# +0.99%; on test it returns +0.47% against +0.48%, exactly nothing. Real in the window
# it was measured in and absent in the next, so no band is applied. Its labels and
# calibration are fine -- an 8% swing_threshold gives a 4.51% positive rate inside the
# target band, calibrating on 294 positives -- so there is no configuration fix here.
#
# Re-measure after any retrain. The gate is deliberately not a refusal: Analyze still
# scores these categories, because investigating a model requires being able to run it.
# What stops is presenting the output as actionable -- no alerts, no place in the
# screener ranking, and a warning on every payload (app/detector.py, app/trading_system.py).
CATEGORIES_FAILING_VALIDATION = {
    "international_emerging": (
        "This model's advantage over ignoring it is not established at the confidence "
        "this system requires. Not a trading signal."
    ),
    "inflation_safe_haven": (
        "This model's advantage over ignoring it is not established at the confidence "
        "this system requires. Not a trading signal."
    ),
    "credit_conditions": (
        "This model's advantage over ignoring it is not established at the confidence "
        "this system requires. Not a trading signal."
    ),
    "rates_recession": (
        "This model's advantage over ignoring it is not established at the confidence "
        "this system requires. Not a trading signal."
    ),
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
