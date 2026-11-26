"""
Probability calibration & reliability analysis for the 30-day readmission Random Forest.

Design rules (so the result is defensible):
  * Calibrators are fit ONLY on the validation split. The test split is touched once, for reporting.
  * Thresholds are re-selected on validation for every variant (calibration changes the score
    scale, so the original 0.33 threshold is not comparable). Calibrated-variant thresholds are
    picked on out-of-fold calibrated validation scores, so they are not tuned on in-sample fits.
  * Nothing is assumed to improve: every metric is reported with a paired bootstrap CI on the
    test set for the *difference* vs the baseline.

Usage in Colab: edit load_data() to return your existing objects, then run the file / paste in cells.
Run `python calibration_analysis.py --demo` to smoke-test on synthetic data.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
    def freeze(est): return FrozenEstimator(est)
    PREFIT_KW = {}
except ImportError:  # older versions
    def freeze(est): return est
    PREFIT_KW = {"cv": "prefit"}

SEED = 42
EPS = 1e-6


# ----------------------------------------------------------------------------- data
def load_data():
    """Return rf (ALREADY FITTED on train), X_val, y_val, X_test, y_test.
    Use the same splits as the dashboard. Do not refit rf here."""
    if "--demo" in sys.argv:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        X, y = make_classification(n_samples=40000, n_features=30, n_informative=6,
                                   weights=[0.91, 0.09], flip_y=0.15, random_state=SEED)
        Xtr, Xr, ytr, yr = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
        Xv, Xt, yv, yt = train_test_split(Xr, yr, test_size=0.5, stratify=yr, random_state=SEED)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        return rf, Xv, yv, Xt, yt
    # --- Project data: run from the repo root (COMP9170-Hospital-Readmission) after notebook 01 has
    # --- produced data/processed/train_clean.csv and test_clean.csv. Reproduces notebook 04's split.
    import io, joblib
    from sklearn.model_selection import train_test_split
    buf = io.BytesIO()
    for i in (1, 2, 3):
        buf.write(open(f"models/rf_model.part{i:02d}", "rb").read())
    buf.seek(0)
    bundle = joblib.load(buf)            # trusted file from the team repo (pickle)
    rf, feats = bundle["model"], list(bundle["feature_names"])
    train = pd.read_csv("data/processed/train_clean.csv")
    test = pd.read_csv("data/processed/test_clean.csv")
    X, y = train.drop(columns="readmit_30"), train["readmit_30"]
    _, Xv, _, yv = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)  # same as notebook 04
    return rf, Xv[feats], yv, test[feats], test["readmit_30"]


# ----------------------------------------------------------------------------- metrics
def ece(y, p, n_bins=10):
    """Expected calibration error with equal-frequency bins."""
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in bins)


def cal_slope_intercept(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b*logit(p). Ideal: a=0, b=1."""
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return lr.intercept_[0], lr.coef_[0, 0]


def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return grid[int(np.argmax(f1s))]


def metric_row(y, p, thr):
    pred = p >= thr
    a, b = cal_slope_intercept(y, p)
    return {
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
        "ECE": ece(y, p),
        "CalIntercept": a, "CalSlope": b,
        "MeanPred": p.mean(), "ObservedRate": y.mean(),
        "PR_AUC": average_precision_score(y, p),
        "ROC_AUC": roc_auc_score(y, p),
        "Threshold": thr,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


def paired_bootstrap(y, probs, thrs, base="Baseline", n_boot=1000, keys=("Brier", "LogLoss", "ECE", "PR_AUC", "Precision", "Recall")):
    """CI for (variant - baseline) differences, resampling test rows jointly."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    out = {k: {name: [] for name in probs if name != base} for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        rows = {name: metric_row(y[idx], p[idx], thrs[name]) for name, p in probs.items()}
        for name in probs:
            if name == base:
                continue
            for k in keys:
                out[k][name].append(rows[name][k] - rows[base][k])
    recs = []
    for k in keys:
        for name, v in out[k].items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            recs.append({"Metric": k, "Variant": name, "DeltaMean": np.mean(v), "CI95_low": lo, "CI95_high": hi,
                         "CI_excludes_0": not (lo <= 0 <= hi)})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- main
def main():
    rf, Xv, yv, Xt, yt = load_data()
    yv, yt = np.asarray(yv), np.asarray(yt)

    # Baseline scores
    p_val = {"Baseline": rf.predict_proba(Xv)[:, 1]}
    p_test = {"Baseline": rf.predict_proba(Xt)[:, 1]}
    thr = {"Baseline": best_f1_threshold(yv, p_val["Baseline"])}

    # Calibrated variants: fit on validation, with the RF frozen (never refit).
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for label, method in [("Sigmoid", "sigmoid"), ("Isotonic", "isotonic")]:
        cal = CalibratedClassifierCV(freeze(rf), method=method, **({"cv": skf} if not PREFIT_KW else PREFIT_KW))
        # out-of-fold calibrated validation scores -> honest threshold selection
        oof = cross_val_predict(CalibratedClassifierCV(freeze(rf), method=method, cv=skf) if not PREFIT_KW
                                else CalibratedClassifierCV(rf, method=method, cv=skf),
                                Xv, yv, cv=skf, method="predict_proba")[:, 1] if not PREFIT_KW else None
        cal.fit(Xv, yv)
        p_test[label] = cal.predict_proba(Xt)[:, 1]
        p_val[label] = oof if oof is not None else cal.predict_proba(Xv)[:, 1]
        thr[label] = best_f1_threshold(yv, p_val[label])

    # Table 1: test metrics
    table = pd.DataFrame({name: metric_row(yt, p, thr[name]) for name, p in p_test.items()}).T
    pd.set_option("display.width", 200)
    print("\n=== Test-set metrics (thresholds chosen on validation, F1-optimal per variant) ===")
    print(table.round(4).to_string())
    table.round(5).to_csv("calibration_metrics.csv")

    # Table 2: paired bootstrap on differences
    print("\n=== Paired bootstrap, variant minus baseline (test set) ===")
    ci = paired_bootstrap(yt, p_test, thr)
    print(ci.round(5).to_string(index=False))
    ci.round(5).to_csv("calibration_bootstrap_deltas.csv", index=False)

    # Reliability diagram
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})
    for name, p in p_test.items():
        frac, mean_p = calibration_curve(yt, p, n_bins=10, strategy="quantile")
        ax[0].plot(mean_p, frac, marker="o", label=f"{name} (Brier {brier_score_loss(yt, p):.4f})")
    lim = max(max(p.max() for p in p_test.values()) * 1.05, yt.mean() * 3)
    ax[0].plot([0, lim], [0, lim], "k--", label="Perfect calibration")
    ax[0].set_xlabel("Mean predicted probability (equal-frequency bins)")
    ax[0].set_ylabel("Observed 30-day readmission frequency")
    ax[0].set_title("Reliability diagram (test set)")
    ax[0].legend()
    for name, p in p_test.items():
        ax[1].hist(p, bins=40, alpha=0.5, label=name)
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Count"); ax[1].set_title("Score distributions")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("reliability_diagram.png", dpi=200)
    print("\nSaved: calibration_metrics.csv, calibration_bootstrap_deltas.csv, reliability_diagram.png")


if __name__ == "__main__":
    main()
