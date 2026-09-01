# Recursive Self-Improving (RSI) Data Science Pipeline

A complete, working data science portfolio project that predicts income
level from U.S. census data. But the prediction itself is not the point.
The point is the **architecture**: five small programs ("agents"), each with
one job, that hand work to each other in a chain — and a feedback loop that
lets the whole system read its own results and change how it behaves on the
next run, automatically, without a human editing any code in between.

This README is written for someone starting from **zero** — you don't need
to already know machine learning, Python packaging, or what "hyperparameter
tuning" means. Every technical term used later in this document is explained
either inline or in the glossary (Section 3). Read this top to bottom and
you should be able to explain, run, modify, and extend the whole project —
even present it in a job interview.

---

## Table of contents

1. [What problem does it solve?](#1-what-problem-does-it-actually-solve)
2. [What is "recursive self-improvement" (RSI)?](#2-what-is-recursive-self-improvement-rsi-in-plain-terms)
3. [Glossary — every technical term explained](#3-glossary--every-technical-term-explained-read-this-if-youre-new-to-ml)
4. [The five agents, explained in depth](#4-the-five-agents-explained-in-depth)
5. [The feedback loop, step by step](#5-the-feedback-loop-step-by-step)
6. [Full file structure](#6-full-file-structure)
7. [What's inside each file](#7-whats-inside-each-file-detailed)
8. [Prerequisites](#8-prerequisites)
9. [Step-by-step: how to run this project](#9-step-by-step-how-to-run-this-project)
10. [OS-specific command reference](#10-os-specific-command-reference)
11. [Understanding config.yaml — every field explained](#11-understanding-configyaml--every-field-explained)
12. [What "done" looks like after a pass](#12-what-done-looks-like-after-a-pass)
13. [Real results from a verified 3-pass run](#13-real-results-from-a-verified-3-pass-run)
14. [How to read report.md and leaderboard.csv yourself](#14-how-to-read-reportmd-and-leaderboardcsv-yourself)
15. [Known limitations](#15-known-limitations-be-upfront-about-these)
16. [Ideas for extending this project](#16-ideas-for-extending-this-project)
17. [Troubleshooting / FAQ](#17-troubleshooting--faq)

---

## 1. What problem does it actually solve?

The dataset is the **Adult Census Income** dataset — a very well-known
dataset in the machine learning world, originally taken from 1994 U.S.
Census data. It has 48,842 rows, one per person, with columns like:

- `age`, `education`, `education_num` (years of schooling)
- `workclass` (private, government, self-employed, etc.)
- `occupation`, `marital_status`, `relationship`, `race`, `sex`
- `capital_gain`, `capital_loss` (investment income/loss)
- `hours_per_week`, `native_country`
- `income` — the **target**: either `<=50K` or `>50K` per year

The task is **binary classification**: given everything else about a
person, predict whether their income is above or below $50K/year.

This dataset was chosen deliberately because it's unglamorous. It has a
realistic mix of numeric and categorical columns, some missing values, and
enough size and structure that feature engineering and model choice
genuinely matter. The dataset itself is not the interesting part of this
project — the pipeline that wraps around it is.

## 2. What is "recursive self-improvement" (RSI), in plain terms?

Most beginner data science projects look like this:

```
load data → clean data → train one model → report accuracy → done
```

You run it once, get a number, and stop. If you want to try something
different, you manually edit the code and run it again yourself.

This project instead runs in **passes**, and automates that manual loop.
Each pass:

1. **Engineers features** — cleans and transforms the raw data
2. **Benchmarks a broad set of models** — tries many algorithms, not just one
3. **Tunes the best ones** — searches for better settings for the top
   performers
4. **Writes a report analyzing what happened** — not just numbers, but
   *why* those numbers look the way they do
5. **Turns that report into instructions for the next pass** — new models
   to try, new features to test, winning settings to reuse

Pass 2 doesn't start from scratch. It starts from what Pass 1 concluded.
That hand-off — the system's own output becoming its next input — is the
"recursive" part. Over several passes, the pipeline is meant to get smarter
about *how it works on this problem*, not just produce a static answer once.

**Important honesty note, up front:** this does not mean the accuracy number
always goes up. Sometimes a pass tries something — a new feature, a new
model, a tuning run — and it doesn't help. That is a real, useful, reportable
finding, not a failure of the project. A system that can say "I tried X, it
didn't help, here's why I think that happened" is more valuable in a
portfolio than one that only ever reports made-up-looking, always-improving
numbers. Section 13 shows exactly what happened when this pipeline was
actually run three times in a row, including the parts that didn't improve.

---

## 3. Glossary — every technical term explained (read this if you're new to ML)

If you already know machine learning, skip to Section 4. If you don't, read
this first — every term below appears later in this README and in the code.

- **Classification**: predicting a category (like `<=50K` vs `>50K`) rather
  than a number. The opposite is **regression**, where you predict a
  continuous number (like a house price).

- **Feature**: one input column used to make a prediction (e.g. `age`,
  `hours_per_week`). The full set of features is called the **feature
  space**.

- **Target**: the column you're trying to predict (here, `income`).

- **Model / algorithm**: a mathematical method that learns patterns from
  data. Examples used in this project: logistic regression, decision trees,
  random forests, gradient boosting, XGBoost, LightGBM, k-nearest neighbors,
  naive Bayes, support vector machines.

- **Training**: the process of a model looking at examples (features +
  known target values) and adjusting itself to predict the target well.

- **Cross-validation (CV)**: instead of training on all your data once and
  testing on the same data (which would be cheating — the model has already
  seen the answers), you split the data into several equal chunks called
  **folds**. You train on all-but-one fold and test on the remaining fold,
  then repeat this so every fold gets a turn being the test set. This
  project uses **5-fold CV** for benchmarking (`cv_folds: 5` in
  `config.yaml`) — meaning every model is trained and tested 5 separate
  times on different slices of the data, and the scores are averaged. This
  gives a much more honest estimate of how well a model would perform on
  data it's never seen, compared to testing on data it trained on.

- **Stratified k-fold**: a version of cross-validation that makes sure each
  fold has roughly the same proportion of `>50K` vs `<=50K` people as the
  full dataset. Important here because the classes are imbalanced (about
  76% earn `<=50K`, 24% earn `>50K`) — without stratification, some folds
  could accidentally end up with very few `>50K` examples, making the score
  noisy and unreliable.

- **ROC-AUC (Receiver Operating Characteristic — Area Under the Curve)**:
  the scoring metric this project uses (`scoring: roc_auc` in
  `config.yaml`). It measures how well a model ranks positive examples
  (`>50K`) above negative ones (`<=50K`), across every possible decision
  threshold, not just one fixed cutoff. It ranges from 0.5 (no better than
  random guessing) to 1.0 (perfect separation). A score of ~0.93, which
  this project achieves, means the model is very good at ranking who is
  more likely to earn `>50K`. ROC-AUC is a good choice here specifically
  *because* the two income classes are imbalanced — plain accuracy would be
  misleading (a model that always guesses `<=50K` would already be "76%
  accurate" while being useless).

- **Overfitting**: when a model performs great on the data it trained on
  but much worse on new data — it "memorized" instead of "learned general
  patterns." This project measures this as the **overfit gap**: the
  difference between a model's average score on training folds vs. test
  folds. A large gap is a red flag.

- **Hyperparameters**: settings you choose for a model *before* training
  that control how it learns (e.g. how many trees a random forest builds,
  how deep each tree can go, how fast a model learns). These are different
  from the patterns the model learns from data — you set hyperparameters
  yourself (or search for good ones); the model doesn't learn them from
  the data directly.

- **Hyperparameter tuning**: systematically trying different
  hyperparameter combinations to find a set that performs better than the
  defaults. This project uses **RandomizedSearchCV** — instead of trying
  every possible combination (which would take forever), it randomly
  samples a fixed number of combinations (`n_iter` in `config.yaml`) from
  a defined range for each hyperparameter, and keeps the best one it found.

- **Feature engineering**: transforming raw columns into better inputs for
  a model. Examples used in this project:
  - **Imputation**: filling in missing values (e.g. replacing a missing
    `workclass` with the label `"missing"`, or a missing number with its
    median)
  - **Encoding**: converting text categories into numbers a model can use.
    This project uses **one-hot encoding** for categories with few unique
    values (each category becomes its own 0/1 column) and **label
    encoding** for categories with many unique values (each category gets
    assigned an integer)
  - **Scaling**: rescaling numeric columns so they're on a comparable
    range (e.g. `age` from 0–100 and `capital_gain` from 0–99,999 don't
    naturally compare well — scaling fixes that). This matters a lot for
    some models (logistic regression, KNN, SVM) and not at all for others
    (tree-based models like random forest), which is exactly the kind of
    pattern Agent 3 in this project is built to notice automatically.
  - **Interaction terms**: creating a new feature by combining two existing
    ones (e.g. `age × education_num`) in case their *combination* matters
    more than either alone.
  - **Aggregated / domain-specific features**: features built from
    knowledge of what the columns actually mean (e.g. a flag for "has any
    capital gains at all," which can matter more than the exact dollar
    amount).

- **Leaderboard**: a ranked table of every model tried, sorted by score —
  this project writes one (`leaderboard.csv`) every pass.

- **Pipeline / orchestrator**: the code that runs steps in the correct
  order automatically, so you don't have to run five separate scripts by
  hand every time.

- **Agent** (as used in this project — not the "AI chatbot" sense): a
  self-contained piece of code with one clear job and a defined
  input/output, designed to be run as one step in a larger chain. Think of
  it like one worker on an assembly line, not a general-purpose assistant.

- **Config file / `config.yaml`**: a plain-text settings file (in YAML
  format) that controls what the pipeline does, without needing to edit
  Python code. YAML is just a human-readable way of writing structured
  settings (lists, key-value pairs) — like a more readable JSON.

- **JSON**: a common structured text format (`{"key": "value"}`) used here
  for machine-readable outputs each agent writes, so the next agent (or the
  next pass) can read them back in reliably.

- **CSV**: comma-separated values — a plain-text spreadsheet format, used
  here for the raw dataset and the leaderboard/metrics tables.

---

## 4. The five agents, explained in depth

### Agent 1 — Model Benchmarking (`agents/agent1_benchmarking.py`)

**Job:** given a ready-to-use feature set, try a broad range of models and
rank them.

**How it actually works:**
1. `MODEL_DEFAULTS` is a dictionary mapping a model's short name (e.g.
   `"random_forest"`) to its scikit-learn class and default settings.
2. `build_model(name, tuned_params)` creates one model instance, merging in
   any hyperparameters carried forward from a previous pass's tuning
   (Section 5 explains how those arrive).
3. `run()` loops over every requested model name, fits it with 5-fold
   stratified cross-validation, and records: the average test score, the
   score's standard deviation across folds (a measure of how *stable* the
   model is), the average training score, and the **overfit gap** (training
   score minus test score).
4. Results are sorted into `leaderboard.csv`, and a JSON summary
   (`agent1_summary.json`) is written flagging: the top 5 models, any
   models with unusually high fold-to-fold variance ("unstable"), and any
   models with a large overfit gap.

Models currently supported: logistic regression, decision tree, random
forest, extra trees, gradient boosting, histogram-based gradient boosting,
AdaBoost, k-nearest neighbors, naive Bayes, linear SVM, XGBoost, and
LightGBM (11-12 depending on whether a new model has been added by the
feedback loop).

### Agent 2 — Feature Engineering (`agents/agent2_feature_eng.py`)

**Job:** turn the raw CSV into a numeric table a model can actually use.

**How it actually works** — each transform is its own small function, and
they always run in this fixed, dependency-safe order regardless of what
order they're listed in `config.yaml` (because, for example, you can't
one-hot encode a column that still has missing values in it):

1. `missing_value_imputation` — fills missing numbers with the column's
   median, missing categories with the literal string `"missing"`
2. `interaction_terms` — creates `age_x_education` and `net_capital`
   (capital_gain minus capital_loss)
3. `aggregated_features` — creates `hours_per_age` (a ratio feature)
4. `domain_specific_transformations` — creates simple 0/1 flags like
   `has_capital_gain`
5. `categorical_encoding` — one-hot encodes categories with 10 or fewer
   unique values, label-encodes categories with more than 10 (to avoid
   creating hundreds of sparse columns for something like `native_country`)
6. `numeric_scaling` — applies `StandardScaler` (rescales each column to
   have mean 0 and standard deviation 1) to numeric columns, skipping
   binary 0/1 columns since scaling those wouldn't do anything useful

Every transform actually applied gets a plain-English line in a **change
log**, which ends up both in `agent2_summary.json` and in the final
`report.md` — so you can see exactly what changed and why, not just a
before/after row count.

### Agent 3 — Results Aggregation (`agents/agent3_aggregation.py`)

**Job:** don't just list numbers — explain what they mean *together*.

This is the "reasoning" layer. It doesn't use an LLM or any AI model itself
— it's a set of clear, auditable rules that look at Agent 1, Agent 2, and
Agent 5's output and produce written-out patterns, for example:

- If `numeric_scaling` was applied, it compares the average score of
  scale-sensitive models (logistic regression, KNN, naive Bayes, SVM)
  against scale-invariant models (all the tree-based ones), and states
  which group benefited more — this is a real comparison computed from the
  actual leaderboard numbers, not a canned statement.
- If Agent 1 flagged any models as overfitting or unstable, it surfaces
  those by name.
- If Agent 5 ran a tuning pass, it reports which models' tuning *actually
  beat* the untuned baseline and which didn't — an honest pass/fail per
  model, not just "tuning happened."
- If there's a previous pass's leaderboard available, it computes the
  literal score delta between this pass's best model and the previous
  pass's best model, and states plainly whether it improved, declined, or
  stayed flat.

It then writes out **recommendations** — concrete suggestions like "try
adding the `interaction_terms` transform" or "deepen hyperparameter tuning
for `lightgbm`, `xgboost`, `gradient_boosting`" — which Agent 4 reads next.

### Agent 4 — Reporting (`agents/agent4_reporting.py`)

**Job:** two outputs, one for a human and one for the machine.

1. **`report.md`** — a full markdown report: dataset summary, modeling
   summary, the leaderboard as a table, the feature engineering change log,
   best-performing approaches, weaknesses/failure cases, the patterns Agent
   3 found, and (if tuning ran) a tuning results section.
2. **`next_pass_config.json`** — this is the actual mechanism of the
   feedback loop. Agent 4 reads Agent 3's recommendations and turns them
   into a real config object for the *next* pass:
   - If a recommendation mentions a missing transform by name
     (`interaction_terms`, `aggregated_features`, or
     `domain_specific_transformations`), it gets added to next pass's
     transform list.
   - If a top-performing model has a known "related family" not yet tried
     (currently: `gradient_boosting` → `hist_gradient_boosting`), that gets
     added to next pass's model list.
   - Any model where Agent 5's tuning **beat the untuned baseline** has its
     winning hyperparameters saved into `tuned_params`, so next pass's
     Agent 1 benchmarks that model pre-tuned instead of with defaults.

This file is genuinely machine-readable and gets consumed automatically —
nobody has to copy numbers between passes by hand.

### Agent 5 — Hyperparameter Tuning (`agents/agent5_tuning.py`)

**Job:** take the models Agent 3 flagged as worth deeper attention and
actually search for better settings, instead of just recommending it in
prose.

**How it actually works:**
1. `PARAM_DISTRIBUTIONS` defines, per model, which hyperparameters to
   search and what range of values to try (e.g. for `random_forest`:
   number of trees between 100–300, max depth between 3–20, etc.)
2. For each of the top N models (`n_top_models` in `config.yaml`) that
   *hasn't already been tuned in a previous pass*, it runs
   `RandomizedSearchCV` — trying `n_iter` random combinations from those
   ranges, evaluated with cross-validation.
3. It compares the best score found to Agent 1's untuned baseline for that
   same model, and honestly records whether tuning helped
   (`beat_baseline: true/false`) — this project does not silently keep only
   the wins.

One important implementation detail: models that already parallelize
internally (like random forest, which uses `n_jobs=-1` to use multiple CPU
cores) have that forced down to a single core (`n_jobs=1`) *before* being
wrapped in the search, because the search itself also parallelizes across
hyperparameter combinations. Running both parallelization layers at once
oversubscribes your CPU and can make the whole thing hang — a real bug this
project hit and fixed during development (see Section 17 for a fuller
explanation if you hit something similar in your own projects).

---

## 5. The feedback loop, step by step

This is the mechanism that makes "recursive" a literal, checkable fact
about this codebase rather than a marketing term.

```
        ┌─────────────────────────────────────────────────────┐
        │                                                       │
        ▼                                                       │
   Agent 2 (features) → Agent 1 (benchmark) → Agent 5 (tune)     │
                              │                     │            │
                              ▼                     ▼            │
                        Agent 3 (aggregate results, find patterns)
                              │
                              ▼
                     Agent 4 (report.md + next_pass_config.json)
                              │
                              └──── read by orchestrator/feedback_loop.py
                                    at the START of the next pass ────┘
```

Concretely, here's what happens when you run
`python orchestrator/pipeline.py --pass 2`:

1. `orchestrator/pipeline.py` calls
   `feedback_loop.load_effective_config(2, base_config)`.
2. That function checks: does `outputs/pass_1/next_pass_config.json`
   exist? If pass 1 hasn't been run yet, no — so pass 2 just uses
   `config.yaml` as-is. If it does exist, its `model_families`,
   `transforms`, and `tuned_params` **override** the corresponding fields
   in the base config for this pass only (the original `config.yaml` file
   itself is never modified).
3. The rest of the pass runs exactly the same five-agent sequence, just
   with this "effective" config instead of the raw base one.
4. At the end, Agent 4 writes a *new* `next_pass_config.json` into
   `outputs/pass_2/`, ready for pass 3 to pick up the same way.

This means **`config.yaml` always represents your starting point**, and
each `outputs/pass_N/next_pass_config.json` is a permanent, inspectable
record of exactly what changed and why, pass by pass — useful both for
debugging and for writing up what the system learned in your portfolio.

See the diagram: **`diagrams/pipeline_architecture.png`**

![Pipeline architecture](diagrams/pipeline_architecture.png)

---

## 6. Full file structure

```
rsi-datascience-project/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── config.yaml                      # All settings: dataset, models, transforms, tuning
├── .gitignore                       # Standard Python/Jupyter ignore rules
│
├── data/
│   ├── raw/
│   │   └── adult_income.csv         # The dataset (48,842 rows, 15 columns)
│   ├── processed/                   # Empty — reserved if you want to save manual intermediate data
│   └── splits/                      # Empty — reserved; this project uses cross-validation, not a fixed split
│
├── agents/
│   ├── __init__.py                  # Makes this folder an importable Python package (intentionally empty)
│   ├── agent1_benchmarking.py       # Agent 1: fits and ranks ~9-12 models
│   ├── agent2_feature_eng.py        # Agent 2: cleans/transforms features
│   ├── agent3_aggregation.py        # Agent 3: synthesizes patterns from agents 1, 2, 5
│   ├── agent4_reporting.py          # Agent 4: writes report.md + next_pass_config.json
│   └── agent5_tuning.py             # Agent 5: hyperparameter search on top models
│
├── orchestrator/
│   ├── __init__.py                  # Empty, same purpose as above
│   ├── data_utils.py                # Shared helpers: load config.yaml, load the CSV
│   ├── feedback_loop.py             # Reads prior pass's next_pass_config.json, merges it in
│   └── pipeline.py                  # THE MAIN ENTRY POINT — runs one full pass end to end
│
├── outputs/
│   ├── pass_1/                      # Everything Pass 1 produced (see Section 12)
│   │   ├── engineered_features.csv
│   │   ├── leaderboard.csv
│   │   ├── agent1_summary.json
│   │   ├── agent2_summary.json
│   │   ├── agent3_summary.json
│   │   ├── agent5_tuning.json
│   │   ├── report.md
│   │   └── next_pass_config.json
│   ├── pass_2/                      # Same file set, produced by pass 2
│   └── pass_3/                      # ...and so on, one folder per pass you run
│
├── metrics/
│   └── pass_comparison.csv          # One row per pass — the trend across passes at a glance
│
├── diagrams/
│   ├── pipeline_architecture.png    # The visual diagram of the 5-agent loop
│   └── generate_diagram.py          # Script that generated the PNG (re-run if you change the architecture)
│
└── notebooks/
    └── exploration.ipynb            # Optional: look at the raw data by hand before trusting the agents
```

---

## 7. What's inside each file (detailed)

| File | What it contains, in detail |
|---|---|
| `config.yaml` | Every tunable setting: which dataset file to load, which column is the target, which class counts as "positive," which models to try first, which feature transforms to try first, how many CV folds, which scoring metric, and Agent 5's tuning budget. **This is the one file you're most likely to edit.** |
| `requirements.txt` | Pinned minimum versions of pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, pyyaml — everything `pip install -r requirements.txt` needs. |
| `.gitignore` | Tells Git to ignore Python cache folders, virtual environments, Jupyter checkpoints, and the largest generated CSV so a git repo doesn't balloon in size. |
| `data/raw/adult_income.csv` | The untouched raw dataset: 48,842 rows, 15 columns (14 features + `income` target), including real missing values in `workclass`, `occupation`, and `native_country`. |
| `agents/agent1_benchmarking.py` | `MODEL_DEFAULTS` (dict of model name → class + default settings), `build_model()` (instantiates a model, optionally merging in tuned hyperparameters), and `run()` (does the actual cross-validated fitting loop and writes `leaderboard.csv` + `agent1_summary.json`). Has a `__main__` block so you can run it standalone: `python agents/agent1_benchmarking.py`. |
| `agents/agent2_feature_eng.py` | One private function per transform (`_missing_value_imputation`, `_categorical_encoding`, `_numeric_scaling`, `_interaction_terms`, `_aggregated_features`, `_domain_specific_transformations`), a fixed `APPLICATION_ORDER` list ensuring transforms run in a dependency-safe sequence, and `run()` which applies them and writes `engineered_features.csv` + `agent2_summary.json`. |
| `agents/agent3_aggregation.py` | `SCALE_SENSITIVE_MODELS` / `SCALE_INVARIANT_MODELS` sets (used to reason about scaling's effect), and `run()` which computes patterns and recommendations, writing `agent3_summary.json`. |
| `agents/agent4_reporting.py` | `RELATED_MODEL_SUGGESTIONS` (maps a strong model to a related one worth trying), `_build_next_pass_config()` (the actual feedback-loop logic), `_render_markdown()` (builds `report.md`), and `run()` which ties both together. |
| `agents/agent5_tuning.py` | `PARAM_DISTRIBUTIONS` (per-model hyperparameter search ranges using `scipy.stats.randint`/`uniform`), and `run()` which executes `RandomizedSearchCV` per model and writes `agent5_tuning.json`. |
| `orchestrator/pipeline.py` | `run_pass(pass_number, base_config)` — calls all five agents in the correct order for one pass — and `update_metrics_log()` which appends/updates a row in `metrics/pass_comparison.csv`. This file's `__main__` block is what `python orchestrator/pipeline.py --pass N` actually runs. |
| `orchestrator/feedback_loop.py` | `load_effective_config(pass_number, base_config)` — the function described in Section 5 that merges the prior pass's `next_pass_config.json` into the current pass's settings. |
| `orchestrator/data_utils.py` | Two tiny helpers: `load_config()` (reads `config.yaml`) and `load_raw_data()` (reads the CSV named in the config). |
| `outputs/pass_N/engineered_features.csv` | The full dataset after Agent 2's transforms — every row, every engineered column, plus the original target column. |
| `outputs/pass_N/leaderboard.csv` | Every model attempted that pass: `model`, `mean_test_score`, `std_test_score`, `mean_train_score`, `overfit_gap`, `fit_time_sec`, `used_tuned_params`, `status`. |
| `outputs/pass_N/agent1_summary.json` | Structured findings: `top_models`, `unstable_models`, `overfitting_models`, `best_model`, `best_score`. |
| `outputs/pass_N/agent2_summary.json` | `transforms_applied`, `shape_before`/`shape_after`, and the full plain-English `change_log`. |
| `outputs/pass_N/agent5_tuning.json` | Per-model: `baseline_score`, `tuned_score`, `improvement`, `beat_baseline`, and the actual `best_params` dictionary found. |
| `outputs/pass_N/agent3_summary.json` | `patterns` (list of plain-English findings), `recommendations` (list of next-step suggestions), and `pass_over_pass_comparison` if a prior pass exists. |
| `outputs/pass_N/report.md` | The complete human-readable write-up — open this first when reviewing a pass. |
| `outputs/pass_N/next_pass_config.json` | The exact settings the *next* pass will use, plus `carried_forward_reasoning` (a copy of the recommendations that produced these changes). |
| `metrics/pass_comparison.csv` | One row per pass: `pass_number, best_model, best_score, n_features, n_models_evaluated, overfitting_models, unstable_models, models_tuned, tuning_improvements`. The fastest way to see the overall trend. |
| `diagrams/pipeline_architecture.png` | The rendered diagram shown above. |
| `diagrams/generate_diagram.py` | Pure matplotlib script that generated that PNG — re-run it if you change the architecture and want an updated picture. |
| `notebooks/exploration.ipynb` | A few cells that load the raw CSV and print its shape, dtypes, missing values, and class balance — nothing more. Entirely optional. |

---

## 8. Prerequisites

- **Python 3.10 or newer** — the code uses modern type hints like
  `dict | None`, which older Python versions don't understand.
- **pip** (comes with Python) to install dependencies.
- **~200MB of free disk space** — mostly `engineered_features.csv`, which
  gets regenerated per pass (roughly 15-19MB each for this dataset).
- **A few CPU cores help.** Agent 1 and Agent 5 use `n_jobs=-1` internally
  (use every available core) for cross-validation, so more cores means
  faster passes. It will still work on a single core — just more slowly.
- **(Optional) Jupyter**, only if you want to open
  `notebooks/exploration.ipynb`. Install with `pip install jupyter` or use
  the Jupyter extension in VS Code.

---

## 9. Step-by-step: how to run this project

**Every command below assumes your terminal's current directory is the
project root** — the folder containing this `README.md` and `config.yaml`.
The code uses relative paths for everything, so running from anywhere else
will fail to find the dataset and will write outputs to the wrong place.

### Step 1 — Get into the project folder and install dependencies

```bash
cd rsi-datascience-project
pip install -r requirements.txt
```

If you want an isolated environment (recommended, but optional), see the
OS-specific commands in Section 10 for creating a virtual environment
first.

### Step 2 — (Optional) Look at the raw data with your own eyes

```bash
jupyter notebook notebooks/exploration.ipynb
```

This just loads the CSV and prints shape, dtypes, missing values, and class
balance — a sanity check before trusting five agents to work on it
automatically. Skip this if you just want results.

### Step 3 — Run Pass 1

```bash
python orchestrator/pipeline.py --pass 1
```

This runs Agent 2 → Agent 1 → Agent 5 → Agent 3 → Agent 4 in sequence and
prints progress to the console as it goes (see Section 12 for exactly what
that output looks like). **Expect this to take several minutes** — Agent 1
is fitting up to 11-12 models with 5-fold cross-validation, and Agent 5 is
running a hyperparameter search on top of that. This is genuinely
CPU-intensive, not a hang — let it finish.

When it's done, read the report:

```bash
cat outputs/pass_1/report.md
```

(On Windows, use `type outputs\pass_1\report.md` in cmd, or
`Get-Content outputs/pass_1/report.md` in PowerShell — see Section 10.)

### Step 4 — Run Pass 2

```bash
python orchestrator/pipeline.py --pass 2
```

Watch the console — you'll see a `[feedback_loop]` line listing exactly
what got carried forward from Pass 1 (new models to try, new transforms,
tuned hyperparameters). This is the recursive part actually happening, not
just described in a README.

### Step 5 — Run Pass 3 (and beyond)

```bash
python orchestrator/pipeline.py --pass 3
```

Keep incrementing `--pass` for as many passes as you want. Each one reads
the previous pass's `next_pass_config.json` automatically.

### Step 6 — Compare all passes at a glance

```bash
cat metrics/pass_comparison.csv
```

One file, one row per pass — the fastest way to see whether the score is
trending up, flat, or down, and how much tuning and feature engineering
changed each time.

### Resetting and starting completely over

```bash
rm -rf outputs
rm -f metrics/pass_comparison.csv
mkdir outputs
python orchestrator/pipeline.py --pass 1
```

This does **not** touch `config.yaml` or `data/raw/adult_income.csv` — your
base settings and dataset are untouched, only the generated results are
cleared.

---

## 10. OS-specific command reference

### macOS / Linux (bash or zsh)

```bash
cd rsi-datascience-project

# Optional but recommended: virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3 orchestrator/pipeline.py --pass 1
python3 orchestrator/pipeline.py --pass 2
python3 orchestrator/pipeline.py --pass 3

cat outputs/pass_1/report.md
cat metrics/pass_comparison.csv
```

Reset:
```bash
rm -rf outputs
rm -f metrics/pass_comparison.csv
mkdir outputs
```

### Windows — VS Code terminal (PowerShell, the default)

Open the terminal with `` Ctrl+` `` after opening the project folder in
VS Code.

```powershell
cd rsi-datascience-project

python -m venv venv
.\venv\Scripts\Activate.ps1
# If PowerShell blocks the script the first time, run this once:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

pip install -r requirements.txt

python orchestrator/pipeline.py --pass 1
python orchestrator/pipeline.py --pass 2
python orchestrator/pipeline.py --pass 3

Get-Content outputs/pass_1/report.md
Get-Content metrics/pass_comparison.csv
```

Reset:
```powershell
Remove-Item -Recurse -Force outputs
Remove-Item -Force metrics\pass_comparison.csv
New-Item -ItemType Directory -Path outputs
```

### Windows — Command Prompt (cmd)

```cmd
cd rsi-datascience-project

python -m venv venv
venv\Scripts\activate.bat

pip install -r requirements.txt

python orchestrator\pipeline.py --pass 1
python orchestrator\pipeline.py --pass 2
python orchestrator\pipeline.py --pass 3

type outputs\pass_1\report.md
type metrics\pass_comparison.csv
```

Reset:
```cmd
rmdir /s /q outputs
del metrics\pass_comparison.csv
mkdir outputs
```

**Windows-specific notes:**
- Use `python`, not `python3` — Windows installs typically only register
  the `python` command.
- If `python` isn't recognized, make sure "Add python.exe to PATH" was
  checked during installation, or select an interpreter in VS Code via
  `Ctrl+Shift+P` → "Python: Select Interpreter".
- Don't close the terminal or press `Ctrl+C` mid-run — each agent only
  writes its output files once it fully finishes.

---

## 11. Understanding `config.yaml` — every field explained

```yaml
dataset:
  path: data/raw/adult_income.csv   # Which CSV file to load
  target: income                     # Which column is the prediction target
  positive_class: ">50K"             # Which value of the target counts as the "positive" class for ROC-AUC
  # Note: this project evaluates models using k-fold cross-validation on
  # the full dataset, not a single train/test split — see the glossary
  # entry for "cross-validation" in Section 3 if that's unfamiliar.

pass_number: 1                       # The base/starting pass number. Usually leave this at 1;
                                      # the actual pass number you're running comes from the
                                      # --pass command-line argument, not this field.

agent1_benchmarking:
  model_families:                    # Which models Agent 1 tries on pass 1 (later passes may add more)
    - logistic_regression
    - decision_tree
    - random_forest
    - extra_trees
    - gradient_boosting
    - adaboost
    - xgboost
    - lightgbm
    - knn
    - naive_bayes
    - svm_linear
  cv_folds: 5                        # Number of cross-validation folds (see glossary)
  scoring: roc_auc                   # Metric used to rank models (see glossary)
  tuned_params: {}                   # Leave this empty — it gets auto-populated by the
                                      # feedback loop as passes complete; don't hand-edit it
                                      # unless you know exactly which hyperparameters you want
                                      # to force for a specific model.

agent2_feature_engineering:
  transforms:                        # Which transforms Agent 2 applies on pass 1
    - missing_value_imputation
    - categorical_encoding
    - numeric_scaling
    # interaction_terms, aggregated_features, and domain_specific_transformations
    # exist in the code but aren't in this starting list — the feedback loop
    # adds them automatically once Agent 3 recommends trying them.

agent5_tuning:
  n_top_models: 3                    # How many of Agent 1's top models Agent 5 attempts to tune each pass
  n_iter: 8                          # How many random hyperparameter combinations to try, per model
  cv_folds: 3                        # CV folds used *during tuning* — kept lower than Agent 1's 5
                                      # folds purely for speed, since tuning already multiplies
                                      # the number of model fits by n_iter

agent4_reporting:
  output_path: outputs               # Base folder where each pass's outputs/pass_N/ subfolder is created
```

**What you're most likely to want to change:**
- Swap in your own dataset: change `dataset.path`, `dataset.target`, and
  `dataset.positive_class`.
- Make passes faster (at the cost of thoroughness): lower `cv_folds`,
  `n_iter`, or the number of entries in `model_families`.
- Make tuning more thorough (at the cost of speed): raise `n_iter` or
  `agent5_tuning.cv_folds`.

---

## 12. What "done" looks like after a pass

After `python orchestrator/pipeline.py --pass 1` finishes, expect console
output roughly like this (using real numbers from a verified run):

```
============================================================
PASS 1
============================================================

[Agent 2] Feature engineering...
  Applied: ['missing_value_imputation', 'categorical_encoding', 'numeric_scaling']
  Shape: [48842, 14] -> [48842, 33]

[Agent 1] Model benchmarking...
  [ok] logistic_regression  roc_auc=0.9002 (+/- 0.0038)  1.1s
  [ok] decision_tree        roc_auc=0.7532 (+/- 0.0044)  1.4s
  ... (9 more models) ...
  Best: lightgbm (0.9296)

[Agent 5] Hyperparameter tuning...
  [ok] lightgbm             baseline=0.9296 tuned=0.9284 (no improvement)
  [ok] xgboost              baseline=0.9249 tuned=0.9286 (improved)
  [ok] gradient_boosting    baseline=0.9222 tuned=0.9270 (improved)

[Agent 3] Aggregating results...
  6 patterns found

[Agent 4] Writing report + next-pass config...
  Report: outputs/pass_1/report.md

============================================================
Pass 1 complete: lightgbm = 0.9296
============================================================
```

And `outputs/pass_1/` will contain exactly 8 files:

- **`engineered_features.csv`** — the dataset after Agent 2's transforms
- **`leaderboard.csv`** — every model, ranked, with score/std/overfit gap
- **`agent1_summary.json`** — Agent 1's structured findings
- **`agent2_summary.json`** — what transforms were applied and the shape change
- **`agent5_tuning.json`** — tuning results per model
- **`agent3_summary.json`** — synthesized patterns and recommendations
- **`report.md`** — the full human-readable report (open this first)
- **`next_pass_config.json`** — exactly what the next pass will use

If any of these 8 files is missing after a pass reports "complete," the run
did not finish correctly — see the Troubleshooting section.

---

## 13. Real results from a verified 3-pass run

This isn't a mocked-up example — this is what actually happened when
Passes 1, 2, and 3 were run end to end on this exact codebase.

**Dataset:** 48,842 rows → Agent 2 expanded 14 raw columns to 33-36
engineered features depending on which transforms were active that pass.

### Pass-over-pass summary

| Pass | Best model | Score (ROC-AUC) | Features | Overfitting models | Models tuned | Tuning improvements |
|---|---|---|---|---|---|---|
| 1 | lightgbm | 0.9296 | 33 | 3 | 3 | 2 |
| 2 | xgboost | 0.9288 | 36 | 3 | 2 | 0 |
| 3 | xgboost | 0.9288 | 36 | 3 | 2 | 0 |

### Pass 1 — establishing the baseline

Agent 1 benchmarked 11 models. `lightgbm` led at 0.9296. Agent 5 tuned the
top 3 (`lightgbm`, `xgboost`, `gradient_boosting`) and found real
improvements for two of them:

| Model | Baseline | Tuned | Result |
|---|---|---|---|
| lightgbm | 0.9296 | 0.9284 | did **not** beat baseline |
| xgboost | 0.9249 | 0.9286 | **improved** by +0.0037 |
| gradient_boosting | 0.9222 | 0.9270 | **improved** by +0.0049 |

Agent 3 also caught that `random_forest`, `extra_trees`, and `decision_tree`
were overfitting, and that scaling helped scale-sensitive models (logistic
regression, KNN) without affecting tree-based ones — both found
automatically from the numbers, not hardcoded.

Agent 4 carried forward into Pass 2: a new model family
(`hist_gradient_boosting`, since `gradient_boosting` did well and Agent 4
knows its faster cousin is worth testing), two new feature transforms
(`interaction_terms`, `aggregated_features`) that hadn't been tried yet, and
the tuned hyperparameters for `xgboost` and `gradient_boosting`.

### Pass 2 — the feedback loop pays off, then a new plateau appears

With the new transforms and `hist_gradient_boosting` added, and `xgboost` /
`gradient_boosting` now benchmarked pre-tuned instead of with defaults,
**`xgboost` (tuned) became the new best model at 0.9288** — a direct result
of carrying Pass 1's tuning forward.

Agent 5 then tried tuning the *newly discovered* top performers,
`hist_gradient_boosting` and `lightgbm` (skipping `xgboost` and
`gradient_boosting` since they were already tuned) — but neither improved
on their own baselines this time:

| Model | Baseline | Tuned | Result |
|---|---|---|---|
| hist_gradient_boosting | 0.9285 | 0.9280 | did not beat baseline |
| lightgbm | 0.9284 | 0.9276 | did not beat baseline |

Since nothing new got carried forward this time (the only recommendation
was "add regularization," which isn't yet a concrete action — see Section
15), Pass 3's config ended up identical to Pass 2's.

### Pass 3 — a genuine, honest plateau

With an unchanged config, unchanged features, and fixed random seeds
throughout the codebase, Pass 3 produced **exactly the same leaderboard and
the same tuning outcome as Pass 2.** This is not a bug — it's the real
outcome of a deterministic system given the same inputs, and it exposes
precisely where this version of the recursion runs out of new ideas to try.
That's a genuinely useful, honest finding for a portfolio write-up: the
system correctly recognized it had nothing new to explore, rather than
pretending to improve.

---

## 14. How to read `report.md` and `leaderboard.csv` yourself

You don't have to take the agents' word for it — everything they conclude
is derived from numbers you can check by hand.

**Opening `outputs/pass_N/leaderboard.csv`:** each row is one model. Sort
by `mean_test_score` descending (it's already sorted this way) to see the
ranking. Look at `std_test_score` — if it's noticeably larger than other
models', that model's score is less trustworthy, since it varied a lot
between folds. Look at `overfit_gap` — anything much larger than 0 means
the model did meaningfully better on training data than on held-out data,
a sign it may not generalize as well as its test score suggests.

**Opening `outputs/pass_N/report.md`:** read top to bottom — dataset
summary, then modeling summary, then the leaderboard as a table, then the
feature engineering log (exactly what Agent 2 did and why), then
best/worst performers, then Agent 3's patterns in plain English, then (if
applicable) the tuning section, then the explicit recommendations for the
next pass. Every claim in this report traces back to a number in
`leaderboard.csv`, `agent1_summary.json`, `agent2_summary.json`, or
`agent5_tuning.json` sitting right next to it in the same folder — nothing
in the report is invented or summarized from outside those files.

---

## 15. Known limitations (be upfront about these)

Being honest about what a project *doesn't* do yet is a stronger signal of
understanding than pretending everything is perfect. This is exactly what
the verified 3-pass run above demonstrated:

1. **The best score doesn't always go up between passes.** Pass 1's
   untuned `lightgbm` (0.9296) actually beat Pass 2 and 3's tuned `xgboost`
   (0.9288) on raw score. Adding new features and models moved the
   *ranking* of best model but not the ceiling — a real result, not a bug.
   If this happens to you on your own dataset, report it rather than
   hiding it.
2. **The pipeline plateaued at Pass 3, and this is traceable to a specific
   cause.** Agent 3's recommendation "add regularization for overfitting
   models" isn't tied to any concrete action anywhere in the codebase —
   Agent 1 doesn't automatically apply it, and Agent 5 doesn't tune away
   overfitting specifically. Once Agent 5 had already tuned the reachable
   top models and no new transforms or model families were left to add,
   the config stopped changing between passes, and — since every part of
   the pipeline uses fixed random seeds — an unchanged config means an
   identical, fully deterministic result. This isn't unpredictable
   behavior; it's the system correctly running out of new ideas within its
   current action space.
3. **No automatic feature selection.** Agent 2 only ever *adds* features —
   nothing currently removes a feature that turns out to be unhelpful or
   redundant, even if Agent 3 could in principle detect this from feature
   importances.
4. **Single dataset, single task type.** The pipeline is architecturally
   dataset-agnostic (see Section 16), but Agent 2's domain-specific
   transforms currently reference this dataset's exact column names, so
   swapping datasets requires a small code change, not just a config
   change.

---

## 16. Ideas for extending this project

- **Close the regularization gap**: have Agent 1 automatically cap tree
  depth or add regularization parameters for any model Agent 3 flags as
  overfitting, so that recommendation becomes a real action instead of
  just a description in the report.
- **Add a 6th agent for feature selection**: drop low-importance features
  (e.g. using permutation importance or SHAP values) flagged by Agent 3,
  closing the "only ever adds features" gap above.
- **Make Agent 2 dataset-agnostic**: parameterize the domain-specific
  transforms so they read column names from `config.yaml` instead of being
  hardcoded, so swapping in a new dataset needs zero code changes.
- **Add SHAP-based feature importance** to Agent 3's pattern detection, so
  it can explain *which features* drove a model's score, not just which
  model scored best.
- **Try a different dataset**: change `config.yaml`'s `dataset` section —
  the orchestrator and agents 1, 3, 4, 5 are already dataset-agnostic as
  long as the task is tabular binary classification.
- **Extend to multi-class or regression targets**: Agent 1's models and
  Agent 5's search grids would need small adjustments (e.g. swapping
  classifiers for regressors, changing the scoring metric), but the
  overall five-agent architecture and feedback loop would carry over
  unchanged.

---

## 17. Troubleshooting / FAQ

**"No such file or directory: data/raw/adult_income.csv"**
You're not running the command from the project root. `cd` into the folder
containing this README first, then re-run the command.

**A pass takes a long time / seems to hang — is something broken?**
No — this is expected. Agent 1 fits up to 11-12 models with 5-fold
cross-validation, and Agent 5 runs a hyperparameter search on top of that.
On a modest machine, a full pass can genuinely take several minutes.
**Don't interrupt it partway through** — each agent only writes its output
files once it fully finishes, so an interrupted run leaves an incomplete
`outputs/pass_N/` folder (see the next FAQ entry).

**A pass folder is missing some of the 8 expected files.**
This means the run was interrupted or crashed partway through (e.g. you
closed the terminal, or hit Ctrl+C). Delete that specific `outputs/pass_N/`
folder and re-run that pass — since every model uses fixed random seeds,
re-running an interrupted pass from a clean folder will reproduce the same
result, nothing is lost by restarting.

**`ModuleNotFoundError` when running an agent file directly.**
Always run agent files either through the orchestrator
(`python orchestrator/pipeline.py --pass N`) or directly with
`python agents/agent1_benchmarking.py`, **from the project root** — not
from inside the `agents/` folder itself, and not with a bare `python3
agent1_benchmarking.py` from within `agents/`. The code adds the project
root to Python's import path assuming it's being launched from there.

**I get a hang or an extremely slow run specifically during Agent 5
(tuning).**
This project already fixes the most common cause of this: several models
(random forest, extra trees, XGBoost, LightGBM) parallelize internally
using every CPU core (`n_jobs=-1`). If you also wrap them in a parallel
hyperparameter search, you oversubscribe your CPU and things can crawl or
lock up. `agents/agent5_tuning.py` forces any such model down to a single
core (`n_jobs=1`) before handing it to `RandomizedSearchCV`, so the *outer*
search is the only thing parallelizing. If you add a new model to
`agent5_tuning.PARAM_DISTRIBUTIONS` yourself and it has an `n_jobs`
parameter, apply the same pattern.

**Why does re-running the exact same pass number from a clean `outputs/`
give me the exact same numbers every time?**
By design. Every cross-validation split and every model's internal
randomness is seeded (`random_state=42` throughout the codebase). This
makes results reproducible and comparisons across passes trustworthy —
if a number changes between two runs of the *same* pass with the *same*
config, that would indicate a real bug, not expected variation.

**Why does Pass 3 sometimes look identical to Pass 2?**
See Section 15, item 2. If Agent 4 finds nothing new to add to the config
(no new transforms, no new model families, no new tuning wins), the
config for the next pass is genuinely unchanged — and an unchanged config,
combined with fixed random seeds, produces an unchanged result. This is
the pipeline correctly recognizing it has exhausted its current ideas, not
a bug.

**I want to change the dataset — what exactly do I need to edit?**
1. In `config.yaml`, update `dataset.path`, `dataset.target`, and
   `dataset.positive_class`.
2. In `agents/agent2_feature_eng.py`, the functions
   `_interaction_terms`, `_aggregated_features`, and
   `_domain_specific_transformations` reference this dataset's specific
   column names (`age`, `education_num`, `capital_gain`, `capital_loss`,
   `hours_per_week`). Adjust or remove these for your new dataset's
   columns — everything else in the pipeline (Agents 1, 3, 4, 5, and the
   orchestrator) works unchanged regardless of dataset, since they operate
   on whatever columns exist rather than hardcoded names.

**Do I need a GPU?**
No. Every model used here (including XGBoost and LightGBM) runs on CPU by
default in this project's configuration.

**Can I run just one agent by itself, without the whole pipeline?**
Yes — every agent file has a `__main__` block that runs it standalone
against whatever config and prior outputs already exist on disk (e.g.
`python agents/agent3_aggregation.py` will re-run Agent 3 using whatever
`agent1_summary.json` / `agent2_summary.json` already exist for the
current `pass_number` in `config.yaml`). This is mainly useful for
debugging or understanding one agent in isolation.
