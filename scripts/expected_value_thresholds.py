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

A floor then has to earn its keep: trades above it must out-earn trades below it by more
than noise. That test, not the shape of the curve, is what decides -- and because the
floor is picked by searching the candidate edges for the largest separation, the bar is
raised to pay for the search -- by permutation, which calibrates to how correlated the
nested candidates actually are rather than assuming they are independent. An earlier version instead required the bottom bin to be *losing* money, which
rejected any model whose trades were all profitable but very unevenly so -- growth_tech
earns +1.34% above a 0.20% floor against +0.64% below it, a ranking that plainly works
and that the old rule threw away.

If no floor separates, that is reported rather than a number being manufactured. It is a
real answer about the model, not a failure of the search.

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
# A floor has to earn its keep: trades above it must out-earn trades below it by more
# than noise. The floor is chosen by searching the candidate bin edges for the largest
# separation, so the bar has to pay for that search.
#
# The correction is a permutation max-t rather than Bonferroni. Bonferroni assumes the
# tests are independent; these are nested subsets of one sample, so the statistic at
# adjacent floors is almost the same number and the correction badly overshoots. It cost
# international_emerging its floor by four hundredths of a standard error while five of
# five cross-validation folds said the model beat its null.
#
# Permuting instead measures the real thing: break the link between predicted probability
# and realized return by shuffling one against the other, recompute the separation at
# every candidate, keep the largest, and repeat. The 95th percentile of those maxima is
# how large the best-of-N looks when nothing is there -- calibrated to whatever
# correlation the candidates actually have, with no independence assumption to violate.
SEPARATION_ALPHA = 0.05
PERMUTATIONS = 2000
PERMUTATION_SEED = 20260830  # fixed so a threshold is reproducible from the same data

# Trades are not independent observations and the tests must stop pretending they are.
# Several fire on the same day across correlated symbols -- measured within-date
# correlation runs 0.09 to 0.46 -- and each is held for up to lookforward_periods bars, so
# it overlaps the ones entered after it. A two-sample t over individual trades understated
# every standard error in this project by 6% to 29% from same-date clustering alone,
# before counting the overlap.
#
# So the resampling unit is a block of consecutive entry dates, long enough to span a
# holding period. Both the standard error and the critical value are computed by
# resampling those blocks, which carries whatever correlation the trades actually have
# instead of assuming none.
BLOCK_LENGTH_DAYS = 10
BOOTSTRAP_REPLICATES = 2000


def permutation_critical_t(values, probabilities, candidates, standard_errors, blocks,
                           minimum_side, permutations=PERMUTATIONS, alpha=SEPARATION_ALPHA,
                           seed=PERMUTATION_SEED):
    """How large the best-of-N studentized separation looks when there is nothing to find.

    Whole blocks of returns are reassigned to a different stretch of history, so the link
    a floor would exploit is broken while the clustering and overlap inside each block
    survive. Permuting individual trades would destroy that structure and produce a null
    far too tight -- a bar too low, and categories shipping on it.

    Each candidate's separation is divided by the same bootstrap standard error used for
    the observed statistic, computed once rather than re-bootstrapped inside every
    permutation. Standardizing matters because candidates differ in how many trades sit
    on each side, and an unstandardized maximum would simply pick the thinnest one.

    This also absorbs the multiplicity of searching candidate floors, which are nested
    subsets of one sample and so far from the independent tests Bonferroni assumes.
    """
    rng = np.random.default_rng(seed)
    masks = [probabilities >= candidate for candidate in candidates]
    maxima = np.empty(permutations)
    for index in range(permutations):
        rows = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        permuted = values[rows]
        best = 0.0
        for mask, standard_error in zip(masks, standard_errors):
            if not standard_error or mask.sum() < minimum_side or (~mask).sum() < minimum_side:
                continue
            separation = _separation(permuted, mask)
            if np.isfinite(separation):
                best = max(best, separation / standard_error)
        maxima[index] = best
    return float(np.quantile(maxima, 1 - alpha))


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
    """Every trade the model would open with no threshold at all.

    Returns (probabilities, profits, entry_dates). Entry dates are carried because the
    significance tests resample by date block rather than by trade -- see
    BLOCK_LENGTH_DAYS.
    """
    probabilities, profits, dates = [], [], []
    for scoring in scored_data.values():
        for trade in simulate_trades(detector, scoring, decision_threshold=0.0)["trades"]:
            probabilities.append(trade["entry_probability"])
            profits.append(trade["profit_pct"])
            dates.append(pd.Timestamp(trade["entry_date"]).normalize())
    order = np.argsort(np.asarray(dates))
    return (np.asarray(probabilities, dtype=float)[order],
            np.asarray(profits, dtype=float)[order],
            np.asarray(dates)[order])


def date_blocks(dates, block_length=BLOCK_LENGTH_DAYS):
    """Trade indices grouped into blocks of `block_length` consecutive entry dates.

    Blocks, not individual dates, because a trade held ten bars overlaps every trade
    entered during those bars. Resampling single dates would break that dependence and
    understate the variance the same way resampling single trades does.
    """
    unique = np.unique(dates)
    by_date = {day: np.flatnonzero(dates == day) for day in unique}
    return [
        np.concatenate([by_date[day] for day in unique[start:start + block_length]])
        for start in range(0, len(unique), block_length)
    ]


def _separation(values, mask):
    above, below = values[mask], values[~mask]
    if len(above) < 2 or len(below) < 2:
        return np.nan
    return float(above.mean() - below.mean())


def block_bootstrap_se(values, mask, blocks, replicates=BOOTSTRAP_REPLICATES, seed=PERMUTATION_SEED):
    """Standard error of the separation, from resampling blocks of consecutive entry dates.

    Resamples whole blocks of consecutive entry dates with replacement, so correlated and
    overlapping trades travel together and the spread of the resampled separations
    reflects how much this sample could really have differed.
    """
    if not blocks or not np.isfinite(_separation(values, mask)):
        return 0.0
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        rows = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        draws[index] = _separation(values[rows], mask[rows])
    return float(np.nanstd(draws, ddof=1))


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
    probabilities, profits, dates = null_trades(detector, scored_data)
    blocks = date_blocks(dates)
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

    # Candidate floors: every usable bin edge except the lowest, which would exclude
    # nothing. The curve's shape narrows the field -- only edges at or above `selected`
    # sit in the region whose bins are all non-negative -- but where every bin is
    # non-negative that constraint is vacuous and the whole range is fair game.
    candidates = [band["lower"] for band in usable[1:]]
    if not candidates:
        return None, "only one usable bin -- nothing to separate", curve

    scored_candidates = []
    for candidate in candidates:
        above, below = profits[probabilities >= candidate], profits[probabilities < candidate]
        if len(above) < MIN_TRADES_FOR_THRESHOLD or len(below) < MIN_TRADES_FOR_THRESHOLD:
            continue
        mask = probabilities >= candidate
        standard_error = block_bootstrap_se(profits, mask, blocks)
        if not standard_error:
            continue
        scored_candidates.append({
            "threshold": candidate, "separation": above.mean() - below.mean(),
            "standard_error": standard_error,
            "t": (above.mean() - below.mean()) / standard_error,
            "above": above, "below": below,
        })
    if not scored_candidates:
        return None, (f"no candidate floor leaves {MIN_TRADES_FOR_THRESHOLD} trades on both "
                      f"sides of it"), curve

    best = max(scored_candidates, key=lambda c: c["t"])
    required = permutation_critical_t(
        profits, probabilities, [c["threshold"] for c in scored_candidates],
        [c["standard_error"] for c in scored_candidates], blocks, MIN_TRADES_FOR_THRESHOLD,
    )
    if best["t"] < required:
        return None, (f"the best of {len(scored_candidates)} candidate floors is {best['threshold']:.2%}, "
                      f"where trades above earn {best['separation']:+.2%} more than those below -- "
                      f"{best['t']:.2f} standard errors, short of the {required:.2f} that the best of "
                      f"{len(scored_candidates)} candidates reaches by chance alone. Not "
                      f"distinguishable from noise."), curve

    return best["threshold"], (
        f"floor {best['threshold']:.2%}, best of {len(scored_candidates)} candidates: trades above it "
        f"earn {best['above'].mean():+.2%} against {best['below'].mean():+.2%} below, a separation of "
        f"{best['separation']:+.2%} at {best['t']:.2f} standard errors (needed {required:.2f} after "
        f"correcting for the search) on {len(best['above'])} trades"), curve


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
