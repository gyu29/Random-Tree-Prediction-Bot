"""Picks each category's decision_threshold by sweeping candidates against validation/.

`decision_threshold` is the minimum predicted probability required to enter a trade.
It doesn't affect the trees, so it can be changed on a trained model without
retraining (see app/model_registry.update_decision_threshold).

Selection uses validation/ ONLY. test/ is scored afterwards and printed, but never
consulted while choosing -- picking the threshold that looks best on test is
guaranteed to look good on test whether or not it generalizes, which is the whole
reason the three-way split exists.

Why not "best validation Sharpe among thresholds with >=5 trades", the rule this
script used to apply: it repeatedly picked the thinnest sample on offer.
docs/2026-08-24-calibration-investigation.md
records inflation_safe_haven's naive argmax landing on
5 validation trades with a Sharpe of 4.13 -- not a credible number at that size -- and
small_cap's pick looking fine on validation (+7.6%) before losing money on test
(-9.1%). A mean computed from five trades has an enormous standard error, and taking
the maximum over a grid of candidates systematically finds whichever one got luckiest.

So the score is the mean per-trade return discounted by its own standard error:

    score = mean(profit) - SHRINKAGE * std(profit) / sqrt(n)

which is the return you can be reasonably confident is at least there. A thin sample
pays a large penalty, a thick one barely any, so the rule prefers a modest edge
measured over 200 trades to a spectacular one measured over five. A candidate must
also clear MIN_TRADES_FOR_SELECTION and score above zero; when none does, the model's
existing trained default is kept and the run says so rather than inventing a pick.

SHRINKAGE is fixed at one standard error and is not a tuning knob -- turning it until
the answer looks good would just reintroduce the selection bias this is here to avoid.

The score alone still isn't enough, because a constrained argmax reports the
constraint. On the first run of this rule, international_emerging's score rose
monotonically as the threshold tightened -- 0.35 through 0.85, never turning over --
so the "best" candidate was simply the last one before the trade floor excluded the
rest (0.85 scored highest of all, on seven trades, at a 100% win rate). Moving the
floor would only move the answer. So a pick must also be an *interior* optimum: strictly
better than the candidate on each side, and not sitting against the edge of the grid or
against the trade floor. When the surface has no interior peak, the search has found
nothing, and this script keeps the model's existing threshold and prints the shape it
saw instead of reporting a number it can't stand behind.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_loader import (  # noqa: E402
    DataProcessor,
    category_test_dir,
    category_validation_dir,
    list_categories,
)
from app.detector import SwingTradeDetector, score_for_backtest, simulate_trades  # noqa: E402
from app.market_data.alpha_vantage_provider import AlphaVantageProvider  # noqa: E402

THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 18)]  # 0.05 .. 0.85
MIN_TRADES_FOR_SELECTION = 20
SHRINKAGE = 1.0  # standard errors subtracted from the mean; see module docstring


def load_symbol_data(directory):
    data = {}
    if not os.path.isdir(directory):
        return data
    for filename in sorted(f for f in os.listdir(directory) if f.endswith(".csv")):
        symbol = DataProcessor.infer_symbol_from_path(filename).upper()
        try:
            data[symbol] = DataProcessor.load_and_validate_data(os.path.join(directory, filename))
        except Exception:
            continue
    return data


def score_once(detector, symbol_data):
    """Indicators + predict_proba per symbol, done once and reused for every candidate
    threshold -- neither depends on the threshold, and together they dominate the cost
    of a sweep. See app/detector.py's BacktestScoring."""
    scored = {}
    for symbol, df in symbol_data.items():
        try:
            scored[symbol] = score_for_backtest(detector, df)
        except ValueError:
            continue  # not enough rows for this symbol's window at this category's lookforward
    return scored


def pooled_metrics(detector, scored_data, threshold):
    all_profits = []
    drawdowns = []
    for scoring in scored_data.values():
        result = simulate_trades(detector, scoring, decision_threshold=threshold)
        all_profits.extend(trade["profit_pct"] for trade in result["trades"])
        drawdowns.append(result["max_drawdown"])

    returns = np.asarray(all_profits) if all_profits else np.asarray([0.0])
    count = len(all_profits)
    mean_profit = float(np.mean(returns))
    # ddof=1: this is a sample of trades, and at n in the tens the difference matters.
    std_profit = float(np.std(returns, ddof=1)) if count > 1 else 0.0
    standard_error = std_profit / np.sqrt(count) if count > 1 else float("inf")
    return {
        "num_trades": count,
        "win_rate": (len([p for p in all_profits if p > 0]) / count) if count else 0.0,
        # Sum, not product -- see scripts/train_all_categories.py for why compounding
        # trades drawn from many symbols in series describes an account nobody could hold.
        "total_profit_equal_weight": float(np.sum(all_profits)) if all_profits else 0.0,
        "sharpe": float(mean_profit / np.std(returns) * (12 ** 0.5)) if np.std(returns) else 0.0,
        "mean_profit": mean_profit,
        "standard_error": standard_error if count > 1 else 0.0,
        # The selection score. -inf for a sample too thin to have a standard error at
        # all, so it can never win the argmax by accident.
        "score": (mean_profit - SHRINKAGE * standard_error) if count > 1 else float("-inf"),
        "avg_max_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
    }


def select_for_category(category):
    provider = AlphaVantageProvider(api_key=None)
    detector = SwingTradeDetector(category, provider)
    default_threshold = detector.decision_threshold

    validation_data = score_once(detector, load_symbol_data(category_validation_dir(category)))
    test_data = score_once(detector, load_symbol_data(category_test_dir(category)))

    validation_sweep = [
        {"threshold": t, **pooled_metrics(detector, validation_data, t)}
        for t in THRESHOLDS
    ]

    selected, selection_note = _select(validation_sweep, default_threshold)

    return {
        "category": category,
        "default_threshold": default_threshold,
        "selected_threshold": selected,
        "changed": selected != default_threshold,
        "selection_note": selection_note,
        "validation_symbols": len(validation_data),
        "test_symbols": len(test_data),
        "validation_sweep": validation_sweep,
        "selected_on_validation": pooled_metrics(detector, validation_data, selected),
        # Reported, never used for selection.
        "selected_on_test": pooled_metrics(detector, test_data, selected),
        "default_on_test": pooled_metrics(detector, test_data, default_threshold),
    }


def _is_eligible(row):
    return row["num_trades"] >= MIN_TRADES_FOR_SELECTION and row["score"] > 0


def _select(sweep, default_threshold):
    """The rule from the module docstring: the best eligible candidate, but only if it
    is an interior peak. Returns (threshold, note) and never raises -- a category with
    no defensible pick keeps its existing threshold."""
    eligible = [row for row in sweep if _is_eligible(row)]
    if not eligible:
        thick = [row for row in sweep if row["num_trades"] >= MIN_TRADES_FOR_SELECTION]
        return default_threshold, (
            f"no candidate cleared the bar ({len(thick)} had >={MIN_TRADES_FOR_SELECTION} "
            f"trades, none scored above zero) -- kept the trained default"
        )

    best = max(eligible, key=lambda row: row["score"])
    position = sweep.index(best)
    lower = sweep[position - 1] if position > 0 else None
    upper = sweep[position + 1] if position + 1 < len(sweep) else None

    if lower is None or upper is None:
        return default_threshold, (
            f"best score is at {best['threshold']:.0%}, the edge of the swept grid, so the "
            f"peak may lie outside it -- kept the trained default"
        )
    if upper["score"] > best["score"]:
        return default_threshold, (
            f"best eligible score is at {best['threshold']:.0%} but {upper['threshold']:.0%} "
            f"scores higher on {upper['num_trades']} trades and was excluded only by the "
            f"{MIN_TRADES_FOR_SELECTION}-trade floor -- the argmax is the floor, not a peak. "
            f"Kept the trained default"
        )
    if lower["score"] > best["score"]:
        return default_threshold, (
            f"score is still rising below {best['threshold']:.0%} -- no interior peak. "
            f"Kept the trained default"
        )

    return best["threshold"], (
        f"interior peak among {len(eligible)} eligible candidate(s): scores "
        f"{lower['score']:+.2%} at {lower['threshold']:.0%} and {upper['score']:+.2%} at "
        f"{upper['threshold']:.0%} both sit below {best['score']:+.2%}"
    )


def _format_metrics(metrics):
    return (f"{metrics['num_trades']:4d} trades  win {metrics['win_rate']:5.1%}  "
            f"mean {metrics['mean_profit']:+.2%}  score {metrics['score']:+.2%}  "
            f"sharpe {metrics['sharpe']:5.2f}  total {metrics['total_profit_equal_weight']:+.1f}u")


def main():
    rows = []
    for category in list_categories():
        print(f"\n{'=' * 78}\n{category}\n{'=' * 78}")
        result = select_for_category(category)

        print("validation sweep (selection uses this and nothing else):")
        for row in result["validation_sweep"]:
            marker = "<-- selected" if (result["changed"]
                                        and row["threshold"] == result["selected_threshold"]) else ""
            print(f"  {row['threshold']:>5.0%}  {_format_metrics(row)}  "
                  f"{'eligible' if _is_eligible(row) else '        '}  {marker}")

        print(f"\nSelected: {result['selected_threshold']:.0%} "
              f"(was {result['default_threshold']:.0%}) -- {result['selection_note']}")
        print(f"  validation @ selected:              {_format_metrics(result['selected_on_validation'])}")
        print(f"  test @ selected (out-of-sample):    {_format_metrics(result['selected_on_test'])}")
        print(f"  test @ previous {result['default_threshold']:.0%}:{'':16s}"
              f"{_format_metrics(result['default_on_test'])}")
        rows.append(result)

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    summary = pd.DataFrame([{
        "category": r["category"],
        "was": r["default_threshold"],
        "selected": r["selected_threshold"],
        "val_trades": r["selected_on_validation"]["num_trades"],
        "val_mean": r["selected_on_validation"]["mean_profit"],
        "val_score": r["selected_on_validation"]["score"],
        "test_trades": r["selected_on_test"]["num_trades"],
        "test_win": r["selected_on_test"]["win_rate"],
        "test_mean": r["selected_on_test"]["mean_profit"],
        "test_sharpe": r["selected_on_test"]["sharpe"],
        "test_mean_at_previous": r["default_on_test"]["mean_profit"],
    } for r in rows])
    print(summary.to_string(index=False))

    changed = [r for r in rows if r["changed"]]
    print(f"\n{len(changed)}/{len(rows)} categories have a defensible new threshold; "
          f"the rest keep the value they were trained with:")
    for r in rows:
        if not r["changed"]:
            print(f"  {r['category']}: {r['selection_note']}")
    if changed:
        print("\nCALIBRATED_DECISION_THRESHOLDS = {")
        for r in rows:
            print(f'    "{r["category"]}": {r["selected_threshold"]:.2f},')
        print("}")
        print("\nApply with app.model_registry.update_decision_threshold(category, value); "
              "no retraining needed.")

    out_path = os.environ.get("THRESHOLD_SELECTION_OUTPUT")
    if out_path:
        with open(out_path, "w") as handle:
            json.dump(rows, handle, indent=2, default=str)
        print(f"\nFull sweep detail written to {out_path}")

    return rows


if __name__ == "__main__":
    main()
