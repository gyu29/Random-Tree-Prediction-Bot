"""Swing-trade label creation.

Vectorized replacement for the original row-by-row `.iloc` loop (a well-known
slow pandas anti-pattern). See tests/test_labeling.py for a row-for-row
equivalence check against the original loop's semantics.

Also the single source of truth for the label threshold: `effective_threshold()`
is the exact value used both to create labels here AND for stop-loss/take-profit
sizing and UI display elsewhere (app/detector.py, ui/). The original code trained
on half of `swing_threshold` while showing users the full value -- that mismatch
is what this shared helper closes.
"""
import numpy as np


# Floor on the upside move a label requires. It exists to stop a threshold collapsing to
# something a symbol clears by drifting, at which point the label stops describing a
# swing and the take-profit sized from it stops being a target.
#
# Lowered from 5% to 3% because 5% was not a judgement about swings, it was an accident
# that made one category untrainable. credit_conditions holds investment-grade credit
# funds, where a 5%-in-ten-days move is close to a tail event: only 1.78% of its rows
# qualified, and the handful that did clustered early enough that its calibration slice
# held six positive examples -- too few to fit a probability mapping, so the category
# could not be calibrated at all and had to be gated. At 3% the same data yields a 5.28%
# positive rate, squarely inside the 2-8% band the working categories sit in, and 78
# calibration positives. 4% was tried first and still fell short at 17.
#
# 3% of an investment-grade credit ETF is a real move, not noise: LQD's daily range runs
# around half a percent. The floor stays because the argument for one stands -- it is
# now set where it stops being a swing for the least volatile assets here, rather than
# for the most.
SWING_THRESHOLD_FLOOR = 0.03


def effective_threshold(swing_threshold):
    """The actual upside-move threshold a label requires. Single source of truth --
    every consumer (labeling, stop/take-profit, UI) must call this instead of
    deriving its own adjusted/default value. See SWING_THRESHOLD_FLOOR."""
    return max(SWING_THRESHOLD_FLOOR, swing_threshold)


def create_swing_labels(df, swing_threshold=0.15, lookforward_periods=10, min_hold_periods=3,
                        mode="peak"):
    """Adds swing_label / swing_profit_potential / swing_risk columns to df.

    For each row i the exit window is [i + min_hold_periods, i + lookforward_periods],
    inclusive. Rows without a full future window are left at 0.

    Two ways to ask whether that window was a success:

    "peak"      the maximum high in the window reaches the threshold. Asks "did it ever
                touch +X%". Correct for a short hold, where touching and exiting are the
                same act.

    "terminal"  the median close across the window reaches it. Asks "was it up X% when
                the window closed". Correct for a long hold, where you are still holding
                at the end and an excursion you did not sell into earned you nothing.

    The distinction is not cosmetic at long horizons. Over a window of months the peak of
    dozens of daily highs is large for almost any volatile instrument, so peak clears a
    threshold the median close does not and reports far more positives at the same bar.
    How much more depends on the window, and steeply: the "+100% within a year" figure
    this docstring used to quote was measured over a 126-252 session window and does not
    carry to the 63-126 one in use. The median close, rather than the single closing
    price, is used so the label does not hinge on one arbitrary day.
    """
    df = df.copy()
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    threshold = effective_threshold(swing_threshold)
    window_size = lookforward_periods - min_hold_periods + 1
    if window_size < 1:
        raise ValueError("lookforward_periods must be >= min_hold_periods")

    price = df[price_col]

    # future_max_high[i] / future_min_low[i] cover high/low over
    # [i + min_hold_periods, i + lookforward_periods] inclusive -- reverse, roll, reverse
    # back is the standard trick for a forward-looking rolling window in pandas.
    shifted_high = df["high"].shift(-min_hold_periods)
    shifted_low = df["low"].shift(-min_hold_periods)
    future_max_high = shifted_high[::-1].rolling(window=window_size, min_periods=window_size).max()[::-1]
    future_min_low = shifted_low[::-1].rolling(window=window_size, min_periods=window_size).min()[::-1]

    if mode == "terminal":
        shifted_close = price.shift(-min_hold_periods)
        future_close = shifted_close[::-1].rolling(window=window_size, min_periods=window_size).median()[::-1]
        has_full_window = future_close.notna()
        upside_potential = (future_close - price) / price
    elif mode == "peak":
        has_full_window = future_max_high.notna()
        upside_potential = (future_max_high - price) / price
    else:
        raise ValueError(f"unknown label mode {mode!r}; expected 'peak' or 'terminal'")

    downside_risk = (price - future_min_low) / price
    is_swing = has_full_window & (upside_potential >= threshold)

    df["swing_label"] = is_swing.astype(int)
    df["swing_profit_potential"] = np.where(is_swing, upside_potential, 0.0)
    df["swing_risk"] = np.where(is_swing, downside_risk, 0.0)

    return df
