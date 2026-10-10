# NSL-KDD results — binary (KDDTest+, 3 seed(s))

| Model | accuracy | macro_f1 | weighted_f1 | detection_rate | false_positive_rate | attack_auc | novel_attack_detection_rate |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.754 ± 0.001 | 0.753 ± 0.001 | 0.752 ± 0.001 | 0.625 ± 0.002 | 0.077 ± 0.000 | 0.772 ± 0.004 | 0.411 ± 0.009 |
| Random Forest | 0.771 ± 0.012 | 0.770 ± 0.012 | 0.768 ± 0.013 | 0.618 ± 0.021 | 0.027 ± 0.000 | 0.960 ± 0.001 | 0.250 ± 0.073 |
| XGBoost | 0.796 ± 0.009 | 0.795 ± 0.009 | 0.794 ± 0.009 | 0.663 ± 0.016 | 0.029 ± 0.000 | 0.968 ± 0.003 | 0.388 ± 0.052 |
| C-LSTM | 0.786 ± 0.010 | 0.786 ± 0.010 | 0.784 ± 0.010 | 0.645 ± 0.017 | 0.027 ± 0.001 | 0.955 ± 0.009 | 0.300 ± 0.058 |

## Per-class recall

| Model | normal | attack |
|---|---|---|
| Logistic Regression | 0.923 ± 0.000 | 0.625 ± 0.002 |
| Random Forest | 0.973 ± 0.000 | 0.618 ± 0.021 |
| XGBoost | 0.971 ± 0.000 | 0.663 ± 0.016 |
| C-LSTM | 0.973 ± 0.001 | 0.645 ± 0.017 |

Hyperparameters chosen on validation (seed 0): see `chosen_configs.json`.
