# Pipeline report — pass 1

Generated: 2026-09-01T20:48:36.244316+00:00

## Dataset summary
- Final feature count: 33 (started at 14)
- Transforms applied: missing_value_imputation, categorical_encoding, numeric_scaling

## Modeling summary
- Models evaluated: 11/11
- Scoring metric: roc_auc
- Best model: **lightgbm** (0.9296)

## Leaderboard
| Model | Score | Std | Overfit gap |
|---|---|---|---|
| lightgbm | 0.9296 | 0.0029 | 0.0257 |
| xgboost | 0.9249 | 0.0022 | 0.0471 |
| gradient_boosting | 0.9222 | 0.0024 | 0.0026 |
| random_forest | 0.9075 | 0.0031 | 0.0925 |
| adaboost | 0.9044 | 0.0039 | 0.0008 |
| logistic_regression | 0.9002 | 0.0044 | 0.0006 |
| svm_linear | 0.9000 | 0.0045 | 0.0006 |
| extra_trees | 0.8895 | 0.0039 | 0.1105 |
| knn | 0.8877 | 0.0033 | 0.0324 |
| naive_bayes | 0.8612 | 0.0029 | 0.0005 |
| decision_tree | 0.7490 | 0.0059 | 0.2510 |

## Feature engineering summary
- workclass: filled 2799 missing values with 'missing' category
- occupation: filled 2809 missing values with 'missing' category
- native_country: filled 857 missing values with 'missing' category
- workclass: one-hot encoded (9 categories)
- education: label encoded (16 categories, too many for one-hot)
- marital_status: one-hot encoded (7 categories)
- occupation: label encoded (15 categories, too many for one-hot)
- relationship: one-hot encoded (6 categories)
- race: one-hot encoded (5 categories)
- sex: one-hot encoded (2 categories)
- native_country: label encoded (42 categories, too many for one-hot)
- scaled 9 numeric columns with StandardScaler: ['age', 'fnlwgt', 'education', 'education_num', 'occupation', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

## Best-performing approaches
- lightgbm
- xgboost
- gradient_boosting
- random_forest
- adaboost

## Weaknesses and failure cases
- Unstable across folds: decision_tree
- Overfitting (train/test gap): random_forest, extra_trees, decision_tree

## Patterns observed
- Numeric scaling was applied. Scale-sensitive models (mean roc_auc=0.8873) benefit most from this transform; tree-based models (mean 0.8896) are largely unaffected by scaling.
- Models showing overfitting (large train/test gap): random_forest, extra_trees, decision_tree. These are candidates for regularization or depth limits next pass.
- Models with high variance across folds: decision_tree. Results for these should be treated cautiously.
- Feature engineering added 19 new columns (14 -> 33). Best model after engineering: lightgbm at 0.9296.
- Hyperparameter tuning improved: xgboost (0.9249 -> 0.9284, +0.0034); gradient_boosting (0.9222 -> 0.9270, +0.0049). These tuned settings will be carried into the next pass.
- Hyperparameter tuning did not beat the untuned baseline for: lightgbm. Default hyperparameters remain in use.

## Hyperparameter tuning
- **lightgbm**: baseline 0.9296 -> tuned 0.9284 (did not beat baseline)
- **xgboost**: baseline 0.9249 -> tuned 0.9284 (beat baseline)
  - Winning params: `{'colsample_bytree': 0.6571467271687763, 'learning_rate': 0.19875765715516733, 'max_depth': 6, 'n_estimators': 51, 'subsample': 0.8887995089067299}`
- **gradient_boosting**: baseline 0.9222 -> tuned 0.9270 (beat baseline)
  - Winning params: `{'learning_rate': 0.28218028561456754, 'max_depth': 3, 'n_estimators': 113, 'subsample': 0.996884623716487}`

## Recommendations for the next iteration
- Add regularization or reduce depth for: random_forest, extra_trees, decision_tree
- Try adding interaction_terms transform if not already applied
- Try adding aggregated_features transform if not already applied