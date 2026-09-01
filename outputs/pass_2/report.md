# Pipeline report — pass 2

Generated: 2026-09-01T20:50:24.830035+00:00

## Dataset summary
- Final feature count: 36 (started at 14)
- Transforms applied: missing_value_imputation, interaction_terms, aggregated_features, categorical_encoding, numeric_scaling

## Modeling summary
- Models evaluated: 12/12
- Scoring metric: roc_auc
- Best model: **lightgbm** (0.9284)

## Leaderboard
| Model | Score | Std | Overfit gap |
|---|---|---|---|
| lightgbm | 0.9284 | 0.0024 | 0.0295 |
| hist_gradient_boosting | 0.9282 | 0.0025 | 0.0142 |
| xgboost | 0.9282 | 0.0025 | 0.0135 |
| gradient_boosting | 0.9278 | 0.0020 | 0.0089 |
| random_forest | 0.9068 | 0.0034 | 0.0932 |
| logistic_regression | 0.9031 | 0.0042 | 0.0006 |
| svm_linear | 0.9029 | 0.0043 | 0.0006 |
| adaboost | 0.9023 | 0.0031 | 0.0016 |
| knn | 0.8904 | 0.0034 | 0.0319 |
| extra_trees | 0.8878 | 0.0034 | 0.1122 |
| naive_bayes | 0.8649 | 0.0038 | 0.0004 |
| decision_tree | 0.7485 | 0.0050 | 0.2515 |

## Feature engineering summary
- workclass: filled 2799 missing values with 'missing' category
- occupation: filled 2809 missing values with 'missing' category
- native_country: filled 857 missing values with 'missing' category
- created age_x_education interaction term
- created net_capital (capital_gain - capital_loss)
- created hours_per_age ratio feature
- workclass: one-hot encoded (9 categories)
- education: label encoded (16 categories, too many for one-hot)
- marital_status: one-hot encoded (7 categories)
- occupation: label encoded (15 categories, too many for one-hot)
- relationship: one-hot encoded (6 categories)
- race: one-hot encoded (5 categories)
- sex: one-hot encoded (2 categories)
- native_country: label encoded (42 categories, too many for one-hot)
- scaled 12 numeric columns with StandardScaler: ['age', 'fnlwgt', 'education', 'education_num', 'occupation', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'age_x_education', 'net_capital', 'hours_per_age']

## Best-performing approaches
- lightgbm
- hist_gradient_boosting
- xgboost
- gradient_boosting
- random_forest

## Weaknesses and failure cases
- Overfitting (train/test gap): random_forest, extra_trees, decision_tree

## Patterns observed
- Numeric scaling was applied. Scale-sensitive models (mean roc_auc=0.8903) benefit most from this transform; tree-based models (mean 0.8900) are largely unaffected by scaling.
- Models showing overfitting (large train/test gap): random_forest, extra_trees, decision_tree. These are candidates for regularization or depth limits next pass.
- Feature engineering added 22 new columns (14 -> 36). Best model after engineering: lightgbm at 0.9284.
- Compared to the previous pass, the best score declined by 0.0012 (lightgbm=0.9296 -> lightgbm=0.9284).
- Hyperparameter tuning did not beat the untuned baseline for: lightgbm, hist_gradient_boosting. Default hyperparameters remain in use.

## Hyperparameter tuning
- **lightgbm**: baseline 0.9284 -> tuned 0.9276 (did not beat baseline)
- **hist_gradient_boosting**: baseline 0.9282 -> tuned 0.9280 (did not beat baseline)

## Pass-over-pass comparison
- Previous best: lightgbm (0.9296)
- Current best: lightgbm (0.9284)
- Change: declined by 0.0012

## Recommendations for the next iteration
- Add regularization or reduce depth for: random_forest, extra_trees, decision_tree