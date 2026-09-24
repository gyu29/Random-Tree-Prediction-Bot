# Paper results

Every table and figure in the write-up is produced by one script, `scripts/paper_run.py`,
from one frozen dataset. Nothing here is edited by hand.

## Reproduce

Python 3.11 with the pinned `requirements.txt` (the versions the saved models were
trained with; `run_manifest.json` records the exact set a run used).

```bash
pip install -r requirements.txt
```

The price data is not in the repository (`train/`, `validation/`, `test/` and
`market_context.csv` are gitignored). Put the archived copy in place, then:

```bash
python scripts/paper_run.py --require-clean
```

The script first checks every data file against `paper/data_manifest.json` and stops if
any differs. Re-downloading from Yahoo does **not** give the same dataset: adjusted
prices are revised after every dividend. Keep the archive that matches the manifest.

`--require-clean` refuses to run with uncommitted changes, so the commit recorded in
`results/run_manifest.json` is the code that produced the numbers. A run without it
works, but its tables are stamped as not citable.

Runtime is about 15 minutes: one model fit per category, plus five cross-validation
fits per category. `--folds 0` skips cross-validation (T9).

## What is frozen

`data_manifest.json` holds the SHA-256 of all 484 data files and the split dates. The
snapshot was downloaded on 2026-09-04. All eight categories share these splits:

| split | dates |
|---|---|
| train | category start to 2017-01-24 |
| validation | 2017-07-26 to 2021-11-09 |
| test | 2022-05-12 to 2026-09-03 |

The gaps are the 126-session embargoes. Rows there are dropped because their labels
would read across the seam.

To pin a new snapshot (this starts a new dataset, and every number changes):
`python scripts/paper_run.py --freeze-data`.

## Outputs (`results/`)

| file | contents |
|---|---|
| `tables.md` | every table, formatted, with its notes |
| `tables/T*.csv` | the same tables, unformatted |
| `figures/fig*.pdf`, `.png` | figures (PDF for the paper, PNG for previews) |
| `results.json` | every number the tables and figures are drawn from |
| `run_manifest.json` | commit, dirty flag, library versions, seeds, settings, data hash |
| `trades/` | every simulated trade behind T7 and T8, for audit |
| `logs/` | each model's training output |

| table | question it answers |
|---|---|
| T1 | What data, and how is it split? |
| T2 | How many independent series does each category really hold? |
| T3 | How much did the old per-symbol split leak? |
| T4 | How does the label definition change the positive rate? |
| T5 | How much evidence exists at each holding horizon? |
| T6 | Are the models calibrated, and do they rank better than chance out of sample? |
| T7 | Does a probability floor separate good trades from bad ones on validation? |
| T8 | Out of sample, does trading above that floor beat ignoring the model? |
| T9 | Is ranking quality stable across walk-forward folds? |

| figure | shows |
|---|---|
| fig1 | Marginal return by predicted-probability bin, with block-bootstrap error bars |
| fig2 | How much the independent-trades standard error understates the real one |
| fig3 | Each category's evidence against the bar it had to clear |
| fig4 | Independent outcomes available at each holding horizon |
| fig5 | Leakage under the old per-symbol split |
| fig6 | Out-of-sample calibration (reliability diagrams) |
| fig7 | Tickers vs effective independent series |

## Rules the run follows

- Each floor is chosen on validation only, by
  `scripts/expected_value_thresholds.solve_threshold`. Test is scored once, at that
  floor.
- Significance uses the project's block bootstrap over entry dates, with the project's
  bar of 2 standard errors. The Bonferroni bar and p-values in T8 are descriptive only
  and decide nothing.
- Models are refit in memory with the project's trainer and seeds, and never written to
  `models/`. T6's last column checks that the refit reproduces the saved models.
