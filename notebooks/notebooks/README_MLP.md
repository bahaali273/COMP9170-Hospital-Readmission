
# MLP Neural Network — Hospital Readmission Prediction

**Course:** COMP 9170 — Data Science  
**Author:** Bahaa Al-Juboori  
**Notebook:** `02_mlp_neural_network.ipynb`

## Objective
Predict whether a patient will be readmitted to hospital
within 30 days using binary classification.

## Dataset
UCI Diabetes 130-US Hospitals (1999–2008).

Processed data generated using the team's shared
preprocessing pipeline.

- Training: 44,780 samples
- Validation: 11,196 samples
- Test: 13,994 samples
- Input features: 133
- Positive class prevalence: approximately 8.97%

## Models
1. Baseline MLP: 133 → 64 → 32 → 1
2. Regularized MLP: Dropout (0.3) and L2 (0.001)

Both models use Adam, binary cross-entropy,
and class weighting.

## Final Test Results
Selected model: Regularized MLP

| Metric | Score |
|---|---:|
| PR-AUC | 0.1579 |
| ROC-AUC | 0.6416 |
| Precision | 0.1597 |
| Recall | 0.4072 |
| F1-score | 0.2294 |

Decision threshold: 0.56, selected on validation data.

## Reproducibility
Run the shared preprocessing notebook first to generate
`train_clean.csv` and `test_clean.csv`.

Upload both CSV files to `/content/` in Google Colab,
then run `02_mlp_neural_network.ipynb` from top to bottom.

Random seed: 42.

## Limitations
- The dataset has a substantial class imbalance.
- The model produces many false positives.
- Results are experimental and not clinically validated.
- Further comparison with traditional ML models is required.
