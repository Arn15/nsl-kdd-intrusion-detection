# Adversarial robustness — binary detectors on KDDTest+

Attack detection rate on the 12833 attack connections in KDDTest+ (higher is better) under constrained evasion. eps = max change per modifiable feature as a fraction of its valid range. **Worst case**: a connection counts as evading if any attack succeeds. Attacks: white-box PGD (40 steps, random and zero start) on differentiable models, black-box score queries (300 iterations) on every model, and transfer from the other differentiable models. Adversarial training used eps = 0.1.

## Worst-case detection rate

| Model | eps=0 | eps=0.05 | eps=0.1 | eps=0.2 | eps=0.3 |
|---|---|---|---|---|---|
| Logistic Regression | 0.621 | 0.001 | 0.001 | 0.001 | 0.000 |
| Random Forest | 0.612 | 0.539 | 0.535 | 0.524 | 0.518 |
| XGBoost | 0.673 | 0.565 | 0.546 | 0.532 | 0.527 |
| C-LSTM | 0.667 | 0.241 | 0.109 | 0.017 | 0.005 |
| C-LSTM + adversarial training | 0.662 | 0.626 | 0.612 | 0.562 | 0.505 |

## By attack at eps = 0.1

| Model | White-box PGD | Black-box queries | Transfer | Worst case |
|---|---|---|---|---|
| Logistic Regression | 0.001 | 0.001 | 0.201 | 0.001 |
| Random Forest | – | 0.535 | 0.544 | 0.535 |
| XGBoost | – | 0.546 | 0.571 | 0.546 |
| C-LSTM | 0.205 | 0.120 | 0.411 | 0.109 |
| C-LSTM + adversarial training | 0.636 | 0.613 | 0.693 | 0.612 |

## By attack at eps = 0.2

| Model | White-box PGD | Black-box queries | Transfer | Worst case |
|---|---|---|---|---|
| Logistic Regression | 0.001 | 0.001 | 0.208 | 0.001 |
| Random Forest | – | 0.524 | 0.540 | 0.524 |
| XGBoost | – | 0.532 | 0.557 | 0.532 |
| C-LSTM | 0.086 | 0.019 | 0.199 | 0.017 |
| C-LSTM + adversarial training | 0.680 | 0.563 | 0.731 | 0.562 |

## Worst-case detection by attack category at eps = 0.2

| Model | DOS | PROBE | R2L | U2R |
|---|---|---|---|---|
| Logistic Regression | 0.000 | 0.000 | 0.001 | 0.104 |
| Random Forest | 0.715 | 0.573 | 0.000 | 0.000 |
| XGBoost | 0.729 | 0.572 | 0.000 | 0.000 |
| C-LSTM | 0.015 | 0.030 | 0.009 | 0.000 |
| C-LSTM + adversarial training | 0.719 | 0.657 | 0.085 | 0.209 |

## Clean performance (no attack)

| Model | Accuracy | Macro-F1 | Detection | FPR | Novel-attack detection |
|---|---|---|---|---|---|
| Logistic Regression | 0.751 | 0.751 | 0.621 | 0.077 | 0.398 |
| Random Forest | 0.768 | 0.767 | 0.612 | 0.027 | 0.233 |
| XGBoost | 0.802 | 0.801 | 0.673 | 0.029 | 0.422 |
| C-LSTM | 0.798 | 0.798 | 0.667 | 0.028 | 0.226 |
| C-LSTM + adversarial training | 0.796 | 0.796 | 0.662 | 0.027 | 0.327 |
