"""Coverage for app/detector.py's walk_forward_backtest -- the loop that turns a
model's per-bar probabilities into simulated entries/exits/equity. Uses a fake
model + scaler so the probability at every bar is exactly controlled, which lets
each scenario (stop-loss, take-profit, max-time, no-trade) be driven deterministically
instead of relying on what a real trained ensemble happens to predict.

N/LOOKFORWARD/DECISION_START are shared across the trade-scenario tests so the
decision window ([DECISION_START, DECISION_END)) is the same fixed shape everywhere.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import (  # noqa: E402
    NoModelAvailableError,
    SwingTradeDetector,
    _max_drawdown,
    walk_forward_backtest,
)

N = 40
LOOKFORWARD = 10
DECISION_START = 5
DECISION_END = N - LOOKFORWARD  # 30
WINDOW_LEN = DECISION_END - DECISION_START  # 25


class _FakeModel:
    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        positive = self._probabilities[: len(X)]
        return np.column_stack([1.0 - positive, positive])


class _FakeScaler:
    def transform(self, X):
        return X.to_numpy()


class _FakeDetector:
    """Duck-types the subset of SwingTradeDetector that walk_forward_backtest uses.
    Reuses the real calculate_stop_take_profit (plain math on effective_swing_threshold)
    rather than re-implementing it, so these tests check the backtest loop, not a
    second copy of the sizing formula."""

    calculate_stop_take_profit = SwingTradeDetector.calculate_stop_take_profit

    def __init__(self, probabilities, feature_columns=("close",), decision_threshold=0.65,
                 lookforward_periods=LOOKFORWARD, effective_swing_threshold=0.15, is_ready=True):
        self.is_ready = is_ready
        self.category = "fake_category"
        self.model = _FakeModel(probabilities)
        self.scaler = _FakeScaler()
        self.feature_columns = list(feature_columns)
        self.decision_threshold = decision_threshold
        self.lookforward_periods = lookforward_periods
        self.effective_swing_threshold = effective_swing_threshold


def _make_ohlcv_df(closes, start="2020-01-01"):
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.004,
            "low": closes * 0.996,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000.0),
        },
        index=index,
    )


def test_max_drawdown_empty_curve():
    assert _max_drawdown([]) == 0.0


def test_max_drawdown_monotonic_increase_is_zero():
    assert _max_drawdown([1.0, 1.1, 1.3, 1.3]) == 0.0


def test_max_drawdown_drop_from_peak():
    assert _max_drawdown([1.0, 1.2, 0.9]) == pytest.approx((0.9 - 1.2) / 1.2)


def test_max_drawdown_tracks_worst_trough_not_final_value():
    assert _max_drawdown([1.0, 0.5, 1.5]) == pytest.approx(-0.5)


def test_raises_when_model_not_ready():
    detector = _FakeDetector(probabilities=[], is_ready=False)
    df = _make_ohlcv_df(np.full(N, 100.0))

    with pytest.raises(NoModelAvailableError):
        walk_forward_backtest(detector, df, decision_start=DECISION_START)


def test_raises_on_insufficient_rows():
    """60 rows clears every unconditional indicator window in create_all_indicators
    (atr_14/atr_21/rsi_21/macd's 26-period EMA etc. have no `len(df) >= period` guard,
    unlike the sma/ema/bollinger blocks -- shorter data crashes there with an unrelated
    IndexError before this function's own row-count check ever runs). A large
    lookforward_periods is what actually drives decision_end <= decision_start here."""
    df = _make_ohlcv_df(np.full(60, 100.0))
    detector = _FakeDetector(probabilities=np.zeros(1), lookforward_periods=58)

    with pytest.raises(ValueError, match="Only 60 usable rows"):
        walk_forward_backtest(detector, df, decision_start=5)


@pytest.mark.parametrize("probability, expected_trades", [(0.65, 1), (0.6499999, 0)])
def test_entry_threshold_is_inclusive(probability, expected_trades):
    df = _make_ohlcv_df(np.full(N, 100.0))
    probabilities = np.zeros(WINDOW_LEN)
    probabilities[0] = probability
    detector = _FakeDetector(probabilities, decision_threshold=0.65)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    assert result["num_trades"] == expected_trades


def test_take_profit_exit_closes_position_and_records_trade():
    closes = np.full(N, 100.0)
    closes[DECISION_START + 1:] = 130.0  # +30% the bar after entry, well past any ATR-sized take-profit
    df = _make_ohlcv_df(closes)
    probabilities = np.zeros(WINDOW_LEN)
    probabilities[0] = 0.90  # only bar 0 of the window (df index DECISION_START) triggers entry
    detector = _FakeDetector(probabilities)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    assert result["num_trades"] == 1
    trade = result["trades"][0]
    assert trade["exit_reason"] == "Take-profit"
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["profit_pct"] == pytest.approx(0.30, abs=1e-6)
    assert result["total_return"] == pytest.approx(0.30, abs=1e-6)
    assert result["win_rate"] == 1.0


def test_stop_loss_exit_closes_position_and_records_trade():
    closes = np.full(N, 100.0)
    closes[DECISION_START + 1:] = 75.0  # -25% the bar after entry, well past any ATR-sized stop-loss
    df = _make_ohlcv_df(closes)
    probabilities = np.zeros(WINDOW_LEN)
    probabilities[0] = 0.90
    detector = _FakeDetector(probabilities)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    assert result["num_trades"] == 1
    trade = result["trades"][0]
    assert trade["exit_reason"] == "Stop-loss"
    assert trade["profit_pct"] == pytest.approx(-0.25, abs=1e-6)
    assert result["total_return"] == pytest.approx(-0.25, abs=1e-6)
    assert result["win_rate"] == 0.0


def test_max_time_exit_when_price_stays_within_stop_and_take_bands():
    df = _make_ohlcv_df(np.full(N, 100.0))  # never moves -> can never hit stop or take-profit
    probabilities = np.zeros(WINDOW_LEN)
    probabilities[0] = 0.90
    detector = _FakeDetector(probabilities)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    assert result["num_trades"] == 1
    trade = result["trades"][0]
    assert trade["exit_reason"] == "Max time"
    assert trade["profit_pct"] == pytest.approx(0.0, abs=1e-9)


def test_no_trade_when_probability_never_reaches_threshold():
    df = _make_ohlcv_df(np.full(N, 100.0))
    probabilities = np.full(WINDOW_LEN, 0.40)  # peaks at 40%, below the 65% threshold
    detector = _FakeDetector(probabilities, decision_threshold=0.65)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    assert result["num_trades"] == 0
    assert result["trades"] == []
    assert result["total_return"] == pytest.approx(0.0)
    assert result["peak_probability"] == pytest.approx(0.40)
    assert "40%" in result["no_trades_reason"]
    assert "65%" in result["no_trades_reason"]


def test_no_reentry_while_position_is_open():
    """Probability stays above threshold on every bar. If entry checking ignored open
    positions, this would open a new position every single bar; instead each position
    must run to its own exit before the next one can open, so consecutive trades must
    never overlap."""
    df = _make_ohlcv_df(np.full(N, 100.0))
    probabilities = np.full(WINDOW_LEN, 0.90)
    detector = _FakeDetector(probabilities)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    trades = result["trades"]
    assert len(trades) >= 2
    for previous, current in zip(trades, trades[1:]):
        assert current["entry_date"] >= previous["exit_date"]
    assert all(trade["exit_reason"] == "Max time" for trade in trades)


def test_position_opened_on_the_last_eligible_bar_is_still_closed_and_recorded():
    """Entries are confined to the decision window; exits are not.

    Both used to live inside `if decision_start <= index < decision_end`, so a position
    opened on the last eligible bar had no later in-window bar to close on -- not even by
    max-time. It was dropped from trades and win_rate while the equity curve went on
    marking it to market, so the two reported numbers described different sets of trades.
    The window ends lookforward_periods bars before the data does precisely so that every
    position taken inside it has room to reach an exit.
    """
    closes = np.full(N, 100.0)
    closes[DECISION_END:] = 50.0  # -50% starting the bar right after the window closes
    df = _make_ohlcv_df(closes)
    probabilities = np.zeros(WINDOW_LEN)
    probabilities[-1] = 0.90  # entry forced onto the last in-window bar (df index DECISION_END - 1)
    detector = _FakeDetector(probabilities)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    assert result["num_trades"] == 1
    trade = result["trades"][0]
    assert trade["exit_reason"] == "Stop-loss"
    assert trade["profit_pct"] == pytest.approx(-0.5, abs=1e-6)
    # The two numbers now describe the same trade: the loss is realized, not merely
    # marked, and win_rate counts the position that num_trades reports.
    assert result["win_rate"] == 0.0
    assert result["avg_profit"] == pytest.approx(-0.5, abs=1e-6)
    assert result["total_return"] == pytest.approx(-0.5, abs=1e-6)


def test_trade_count_and_equity_curve_agree_on_the_last_bar():
    """The general form of the defect above: every position the backtest opens must end
    up in `trades`, whatever bar it was opened on."""
    closes = np.linspace(100.0, 130.0, N)
    df = _make_ohlcv_df(closes)
    probabilities = np.full(WINDOW_LEN, 0.90)  # enter as early and often as possible
    result = walk_forward_backtest(_FakeDetector(probabilities), df, decision_start=DECISION_START)

    realized = 1.0
    for trade in result["trades"]:
        realized *= 1 + trade["profit_pct"]
    assert result["total_return"] == pytest.approx(realized - 1, abs=1e-9), (
        "equity curve ends on an unrealized position that never reached `trades`"
    )


def test_buy_hold_curve_matches_price_ratio_to_first_close():
    closes = np.tile([100.0, 102.0, 99.0, 105.0], N // 4)
    df = _make_ohlcv_df(closes)
    probabilities = np.zeros(WINDOW_LEN)  # never enters, keeps this test isolated to buy_hold_curve
    detector = _FakeDetector(probabilities)

    result = walk_forward_backtest(detector, df, decision_start=DECISION_START)

    expected = (closes / closes[0]).tolist()
    assert result["buy_hold_curve"] == pytest.approx(expected)
