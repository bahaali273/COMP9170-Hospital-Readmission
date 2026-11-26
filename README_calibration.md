# Probability Calibration & Reliability Analysis

BCIT COMP 9170 · 30-day hospital readmission (UCI Diabetes 130-US Hospitals)

## Research question

Are the predicted 30-day readmission probabilities from the deployed Random Forest reliable,
and can calibration improve their agreement with observed outcomes?

## What this does

`calibration_analysis.py` takes the **already-trained** Random Forest from this repo
(`models/rf_model.part01-03`) and compares three probability sources on the held-out test set:

| Variant | Description |
|---|---|
| Baseline | Raw Random Forest `predict_proba` |
| Sigmoid | Platt scaling via `CalibratedClassifierCV(method="sigmoid")` |
| Isotonic | Isotonic regression via `CalibratedClassifierCV(method="isotonic")` |

The Random Forest is frozen and never refit. Calibrators are fit on the **validation split only**;
the test split is used once, for reporting.

### Metrics reported

- **Probability quality:** Brier score, log loss, expected calibration error (ECE, 10 equal-frequency bins),
  calibration intercept and slope (ideal: 0 and 1), mean predicted probability vs observed rate
- **Ranking:** PR-AUC, ROC-AUC
- **Decision metrics:** precision, recall, F1 at a validation-selected threshold
- **Uncertainty:** paired bootstrap (1,000 resamples of the test set) for each variant minus baseline

### Design decisions

- **Same splits as the models.** Train/test split from `notebooks/01_cleaning_condensed.ipynb`
  (80/20, seed 42, stratified, one encounter per patient). Validation split from
  `notebooks/04_baseline_models.ipynb` (20% of train, seed 42, stratified). The saved model has
  not seen the validation rows.
- **Thresholds are re-selected per variant.** Calibration changes the score scale, so the deployed
  threshold of 0.33 does not carry over. Each variant uses the F1-optimal threshold on validation;
  calibrated variants use out-of-fold validation scores so the threshold is not tuned on in-sample fits.
- **No assumed improvement.** Differences are reported with 95% bootstrap intervals, and
  `CI_excludes_0` flags whether each difference is distinguishable from zero.
- **Sigmoid preserves ranking.** It is a monotonic transform, so ROC-AUC and PR-AUC are unchanged;
  isotonic can create tied scores and slightly change them.

## How to run

Run from the repository root (Colab or local).

```bash
# 1. Match the version the model was saved with (needed to unpickle safely and reproduce exactly)
pip install scikit-learn==1.6.1 matplotlib scipy pandas joblib

# 2. Create data/processed/train_clean.csv and test_clean.csv
#    Open and run notebooks/01_cleaning_condensed.ipynb
#    (the raw UCI zip is already in data/)

# 3. Run the analysis
python calibration_analysis.py
```

`FrozenEstimator` (used to freeze the model) requires scikit-learn >= 1.6. The model file is a
pickle (via joblib); only load it from this trusted repository.

To smoke-test the script without the project data, run `python calibration_analysis.py --demo`
(synthetic data; the numbers are meaningless).

### Outputs

| File | Contents |
|---|---|
| `calibration_metrics.csv` | All metrics for Baseline, Sigmoid, Isotonic on the test set |
| `calibration_bootstrap_deltas.csv` | Variant-minus-baseline differences with 95% CIs |
| `reliability_diagram.png` | Reliability diagram (observed frequency vs mean predicted probability) and score histograms |

## Results (test set, n = 13,994, observed readmission rate 8.97%)

Produced with scikit-learn 1.9.1, so baseline PR-AUC and ROC-AUC (0.1558, 0.6455) differ slightly
from the dashboard (0.1568, 0.6477). Re-run with scikit-learn 1.6.1 to match exactly.

| | Brier | Log loss | ECE | Mean prediction | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline RF | 0.1134 | 0.3928 | 0.1756 | 0.265 | 0.1558 | 0.6455 | 0.167 | 0.379 | 0.232 |
| Sigmoid | 0.0796 | 0.2904 | 0.0054 | 0.089 | 0.1558 | 0.6455 | 0.169 | 0.372 | 0.232 |
| Isotonic | 0.0796 | 0.2930 | 0.0109 | 0.089 | 0.1502 | 0.6442 | 0.167 | 0.374 | 0.231 |

## Discussion

- **The raw probabilities are not reliable.** The Random Forest predicts about 26.5% average risk
  against an observed rate of about 9%. A likely cause is `class_weight="balanced_subsample"` in
  the training notebook, which shifts predicted probabilities away from the true prevalence.
- **Calibration substantially improves probability quality.** Brier score, log loss and ECE all
  improve, and the bootstrap intervals for these differences exclude zero.
- **Sigmoid and isotonic perform similarly.** Sigmoid is slightly better on log loss and ECE and
  keeps ranking metrics unchanged. Isotonic lowers PR-AUC slightly (about -0.006, interval excludes zero)
  and has a calibration slope of 0.83, which may indicate mild overfitting on the ~11k-row calibration set.
- **Trade-offs are small.** Precision does not change significantly. Recall decreases by about
  0.5-0.6 percentage points for both calibrators (intervals exclude zero).
- **Calibration does not improve discrimination.** ROC-AUC stays near 0.645, so the model is not
  better at identifying who will be readmitted; only the meaning of its probabilities improves.

## Limitations

- Calibration is fit on one validation split; a different split could change isotonic results more than sigmoid.
- Calibration was assessed on the same historical UCI data (1999-2008, US hospitals). It does not
  show that probabilities will be calibrated at a new hospital or in a later period.
- Thresholds were chosen by maximizing F1, which is one of several reasonable operating points.
- This is a research prototype, not a clinical decision-support tool.
