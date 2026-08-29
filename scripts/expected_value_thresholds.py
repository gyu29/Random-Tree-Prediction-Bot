"""Computes each category's decision_threshold from realized expected value.

The obvious formula is wrong here, and it is worth recording why. A trade entered at
predicted probability p looks like it should break even at p* = L / (W + L), with W the
average gain on winners and L the average loss on losers. That requires p to be the
probability *the trade wins*. It is not: the model predicts P(swing label), the chance
of a threshold-sized move inside the lookforward window, while a trade wins whenever it
exits above its entry -- a far easier event. Measured on validation, market_beta's
trades carry a mean entry probability of 0.34% against a 61% realized win rate. Feeding
the first number into a formula expecting the second puts p* near 50%, which no
calibrated model here ever reaches, and reports "no threshold exists" for reasons that
are entirely an artifact of the mismatch.

What calibration actually buys is a probability that is monotone and comparable, so it
can *order* trades. So the question becomes empirical: as entry probability rises, does
realized profit per trade rise with it, and where does it cross zero?

This takes every trade the model would open with no threshold at all, bins them by
entry probability, and reads off the marginal return in each bin. The threshold is the
lowest bin edge from which every higher bin is non-negative on a shrunk mean
(mean - one standard error, the same discount scripts/select_thresholds.py applies).
Above that point taking trades is worth it; below it, it is not.

If no such point exists -- if the marginal curve is flat or negative all the way up --
then the model's ranking does not identify profitable trades at any threshold, and that
is reported rather than a number being manufactured. That is a real answer about the
model, not a failure of the search.

Selection uses validation/ only. test/ is scored afterwards for reporting.

Requires calibrated models: an uncalibrated score is not monotone in probability across
categories and the bins would not be comparable. Categories whose model declined
calibration (see app/ensemble.MIN_POSITIVES_FOR_CALIBRATION) are skipped with that
reason printed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_loader import category_test_dir, category_validation_dir, list_categories  # noqa: E402
from app.detector import SwingTradeDetector, simulate_trades  # noqa: E402
from app.market_data.alpha_vantage_provider import AlphaVantageProvider  # noqa: E402
from scripts.select_thresholds import load_symbol_data, score_once  # noqa: E402

PROBABILITY_BINS = 8          # quantile bins over entry probability
MIN_TRADES_PER_BIN = 25       # below this a bin's mean is noise, not a marginal return
MIN_TRADES_FOR_THRESHOLD = 40  # the selected set must be at least this large
SHRINKAGE_STANDARD_ERRORS = 1.0


def win_loss_profile(detector, scored_data, threshold):
    """Average gain on winners and average loss on losers at `threshold`."""
    profits = []
    for scoring in scored_data.values():
        result = simulate_trades(detector, scoring, decision_threshold=threshold)
        profits.extend(trade["profit_pct"] for trade in result["trades"])
    array = np.asarray(profits, dtype=float)
    wins = array[array > 0]
    losses = array[array <= 0]
    return {
        "num_trades": len(array),
        "num_wins": len(wins),
        "num_losses": len(losses),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(-losses.mean()) if len(losses) else 0.0,  # positive magnitude
        "win_rate": float(len(wins) / len(array)) if len(array) else 0.0,
        "mean_profit": float(array.mean()) if len(array) else 0.0,
    }


def null_trades(detector, scored_data):
    """Every trade the model would open with no threshold at all, as (entry probability,
    realized profit). The largest sample available, and the one the marginal curve is
    estimated from."""
    probabilities, profits = [], []
    for scoring in scored_data.values():
        for trade in simulate_trades(detector, scoring, decision_threshold=0.0)["trades"]:
            probabilities.append(trade["entry_probability"])
            profits.append(trade["profit_pct"])
    return np.asarray(probabilities, dtype=float), np.asarray(profits, dtype=float)


def marginal_ev_curve(probabilities, profits, bins=PROBABILITY_BINS):
    """Realized profit per trade within each entry-probability bin.

    Quantile edges, not equal-width: entry probabilities are heavily concentrated near
    the base rate, and equal-width bins would put almost every trade in the first one.
    """
    if len(probabilities) == 0:
        return []
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(probabilities, quantiles))
    if len(edges) < 3:
        return []

    curve = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (probabilities >= lower) & (
            (probabilities <= upper) if upper == edges[-1] else (probabilities < upper)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        values = profits[in_bin]
        mean = float(values.mean())
        standard_error = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else float("inf")
        curve.append({
            "lower": float(lower), "upper": float(upper), "num_trades": count,
            "mean_profit": mean, "standard_error": standard_error if count > 1 else 0.0,
            "shrunk_mean": (mean - SHRINKAGE_STANDARD_ERRORS * standard_error) if count > 1 else float("-inf"),
            "win_rate": float((values > 0).mean()),
        })
    return curve


def solve_threshold(detector, scored_data):
    """The lowest entry probability from which every higher bin is non-negative.

    Walks the curve from the top down and stops at the first bin that fails, so the
    selected region is contiguous and open-ended upward -- a threshold is a floor, and a
    floor that admits a bad bin above it is not one.

    Returns (threshold, note, curve). threshold is None when no such region exists.
    """
    probabilities, profits = null_trades(detector, scored_data)
    curve = marginal_ev_curve(probabilities, profits)
    if not curve:
        return None, f"only {len(probabilities)} trades with no threshold -- no curve to estimate", curve

    usable = [b for b in curve if b["num_trades"] >= MIN_TRADES_PER_BIN]
    if len(usable) < 2:
        return None, (f"only {len(usable)} bin(s) reached {MIN_TRADES_PER_BIN} trades -- "
                      f"the marginal curve is not estimable"), curve

    selected = []
    for candidate in reversed(usable):
        if candidate["shrunk_mean"] < 0:
            break
        selected.insert(0, candidate)

    if not selected:
        top = usable[-1]
        # The top bin failing while lower ones pass means profit *falls* as predicted
        # probability rises. No floor can express that: the trades a threshold would
        # keep are the ones losing money.
        inverted = any(b["shrunk_mean"] >= 0 for b in usable)
        detail = ("the relationship is inverted -- marginal return falls as predicted probability "
                  "rises, so the trades a threshold would keep are the losing ones"
                  if inverted else "marginal return is negative across the whole curve")
        return None, (f"no usable floor exists: the highest bin "
                      f"({top['lower']:.2%}-{top['upper']:.2%}) shrinks to {top['shrunk_mean']:+.2%} "
                      f"on {top['num_trades']} trades. {detail}."), curve

    if selected[0] is usable[0]:
        # The profitable region starts at the bottom bin, so every trade qualifies and the
        # "threshold" is not one -- it is the model declining to discriminate. Emitting 0
        # here would silently turn the app into an unconditional trader.
        return None, (f"every bin from the lowest ({usable[0]['lower']:.2%}) upward is non-negative, "
                      f"so no threshold excludes anything -- the ranking does not separate "
                      f"profitable trades from unprofitable ones. Trading on this model is "
                      f"indistinguishable from trading on every bar."), curve

    threshold = selected[0]["lower"]
    taken = int(np.sum(probabilities >= threshold))
    if taken < MIN_TRADES_FOR_THRESHOLD:
        return None, (f"the profitable region starts at {threshold:.2%} but holds only {taken} "
                      f"trades (need {MIN_TRADES_FOR_THRESHOLD})"), curve

    mean_above = float(profits[probabilities >= threshold].mean())
    mean_below = float(profits[probabilities < threshold].mean())
    return threshold, (f"marginal return is non-negative in every bin from {threshold:.2%} upward "
                       f"({len(selected)} of {len(usable)} usable bins) and negative below: "
                       f"{taken} trades at {mean_above:+.2%}/trade above, "
                       f"{mean_below:+.2%}/trade below"), curve


def run_category(category):
    detector = SwingTradeDetector(category, AlphaVantageProvider(api_key=None))
    stats = detector.training_stats
    if not stats.get("is_calibrated"):
        return {"category": category, "threshold": None, "deployed": detector.decision_threshold,
                "note": "model is not calibrated -- its scores are not comparable, so the marginal "
                        "return curve cannot be read", "curve": [], "validation": None, "test": None}

    validation = score_once(detector, load_symbol_data(category_validation_dir(category)))
    threshold, note, curve = solve_threshold(detector, validation)
    result = {"category": category, "threshold": threshold, "deployed": detector.decision_threshold,
              "note": note, "curve": curve,
              "calibration_method": stats.get("calibration_method"),
              "expected_calibration_error": stats.get("expected_calibration_error")}
    if threshold is not None:
        test = score_once(detector, load_symbol_data(category_test_dir(category)))
        result["validation"] = win_loss_profile(detector, validation, threshold)
        result["test"] = win_loss_profile(detector, test, threshold)
        result["test_at_deployed"] = win_loss_profile(detector, test, detector.decision_threshold)
    return result


def main():
    rows = []
    for category in list_categories():
        print(f"\n{'=' * 78}\n{category}\n{'=' * 78}")
        result = run_category(category)
        if result["curve"]:
            print(f"  {'entry probability':>22s} {'trades':>7s} {'win':>6s} {'mean':>8s} {'shrunk':>8s}")
            for band in result["curve"]:
                thin = "" if band["num_trades"] >= MIN_TRADES_PER_BIN else "  (thin)"
                print(f"  {band['lower']:>9.2%} - {band['upper']:<9.2%} {band['num_trades']:>7d} "
                      f"{band['win_rate']:>6.1%} {band['mean_profit']:>+8.2%} "
                      f"{band['shrunk_mean']:>+8.2%}{thin}")
        if result["threshold"] is None:
            print(f"  NO THRESHOLD: {result['note']}")
        else:
            print(f"  {result['note']}")
            for label, key in [("validation", "validation"), ("test (out-of-sample)", "test"),
                               (f"test at deployed {result['deployed']:.0%}", "test_at_deployed")]:
                profile = result[key]
                print(f"    {label:32s} {profile['num_trades']:>4d} trades  "
                      f"win {profile['win_rate']:>5.1%}  mean {profile['mean_profit']:+.2%}")
        rows.append(result)

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    table = pd.DataFrame([{
        "category": r["category"],
        "deployed": r["deployed"],
        "expected_value_threshold": r["threshold"],
        "calibration": r.get("calibration_method"),
        "ece": r.get("expected_calibration_error"),
        "test_trades": r["test"]["num_trades"] if r.get("test") else None,
        "test_win_rate": r["test"]["win_rate"] if r.get("test") else None,
        "test_mean": r["test"]["mean_profit"] if r.get("test") else None,
        "test_mean_at_deployed": r["test_at_deployed"]["mean_profit"] if r.get("test_at_deployed") else None,
    } for r in rows])
    print(table.to_string(index=False))

    solved = [r for r in rows if r["threshold"] is not None]
    print(f"\n{len(solved)}/{len(rows)} categories have a computed threshold.")
    for result in rows:
        if result["threshold"] is None:
            print(f"  {result['category']}: {result['note']}")
    if solved:
        print("\nCALIBRATED_DECISION_THRESHOLDS = {")
        for result in rows:
            value = result["threshold"] if result["threshold"] is not None else result["deployed"]
            # 4 decimals: calibrated thresholds are sub-1%, and rounding to 2 would print
            # 0.0023 as 0.00 -- a value that turns the app into an unconditional trader.
            print(f'    "{result["category"]}": {value:.4f},'
                  + ("" if result["threshold"] is not None else "  # unchanged: no threshold found"))
        print("}")

    out_path = os.environ.get("EXPECTED_VALUE_OUTPUT")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, default=str)
        print(f"\nFull trace written to {out_path}")
    return rows


if __name__ == "__main__":
    main()
