"""
UEBA Isolation Forest Detector
--------------------------------
Implements the exact model configuration from the manuscript:

  "Evaluating the Robustness of Isolation Forest in UEBA:
   A Case Study on the NSL-KDD Dataset"
  — Judah Idowu (manuscript in preparation)

Model configuration:
  - Algorithm: Isolation Forest (scikit-learn)
  - n_estimators: 100
  - contamination: 0.01 (baseline)
  - Training data: normal traffic only

Baseline performance on NSL-KDD test set (125,973 samples):
  - Overall accuracy: 92.51%
  - Class 0 (normal): P=0.88, R=0.99, F1=0.93
  - Class 1 (attack): P=0.99, R=0.85, F1=0.91
  - True Negatives: 66,669 | False Positives: 674
  - False Negatives: 8,760 | True Positives: 49,870
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, accuracy_score
)
from scipy.stats import pearsonr
import joblib
import json
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Isolation Forest Detector
# ---------------------------------------------------------------------------

class UEBADetector:
    """
    Isolation Forest-based UEBA anomaly detector.

    Trained exclusively on normal traffic to model expected
    behavioural baselines. Anomalies are detected as statistical
    outliers from this learned distribution.

    Configuration matches manuscript exactly:
        n_estimators=100, contamination=0.01
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: int = 42
    ):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.random_state  = random_state

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "UEBADetector":
        """Train on normal traffic only."""
        self.model.fit(X_train)
        self.is_fitted = True
        print(f"[Detector] Trained on {len(X_train):,} normal samples  "
              f"| n_estimators={self.n_estimators}  "
              f"| contamination={self.contamination}")
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Raw anomaly scores. More negative = more anomalous.
        Consistent with sklearn's score_samples() convention.
        """
        assert self.is_fitted
        return self.model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary predictions using the model's internal contamination threshold.
        Returns: 0 = normal, 1 = anomaly (attack)
        sklearn convention: -1=anomaly → remap to 1
        """
        assert self.is_fitted
        raw = self.model.predict(X)
        return np.where(raw == -1, 1, 0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Full evaluation matching manuscript metrics exactly."""
        assert self.is_fitted

        preds  = self.predict(X)
        scores = self.decision_scores(X)

        report = classification_report(y, preds, output_dict=True, zero_division=0)
        cm     = confusion_matrix(y, preds)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)

        try:
            auc_roc = roc_auc_score(y, -scores)   # negate: higher = more anomalous
            avg_p   = average_precision_score(y, -scores)
        except Exception:
            auc_roc = avg_p = 0.0

        return {
            "accuracy": round(accuracy_score(y, preds), 4),
            "auc_roc":  round(auc_roc, 4),
            "avg_precision": round(avg_p, 4),
            # Class 0 — Normal
            "precision_normal": round(report["0"]["precision"], 4),
            "recall_normal":    round(report["0"]["recall"], 4),
            "f1_normal":        round(report["0"]["f1-score"], 4),
            # Class 1 — Attack
            "precision_attack": round(report["1"]["precision"], 4),
            "recall_attack":    round(report["1"]["recall"], 4),
            "f1_attack":        round(report["1"]["f1-score"], 4),
            # Macro / weighted
            "macro_avg_f1":    round(report["macro avg"]["f1-score"], 4),
            "weighted_avg_f1": round(report["weighted avg"]["f1-score"], 4),
            # Confusion matrix
            "true_negatives":  int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives":  int(tp),
        }

    def save(self, path: str = "models/detector.pkl"):
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[Detector] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "UEBADetector":
        det = cls()
        det.model = joblib.load(path)
        det.is_fitted = True
        return det


# ---------------------------------------------------------------------------
# Feature Sensitivity Analysis
# ---------------------------------------------------------------------------

class FeatureSensitivityAnalyzer:
    """
    Computes the correlation-based sensitivity matrix between individual
    features and the Isolation Forest anomaly decision scores.

    Methodology: For each feature f_i, compute the Pearson correlation
    coefficient between f_i values and the raw score_samples() output.
    Higher absolute correlation = higher leverage point for evasion.

    Top 10 results from the manuscript (reproduced for reference):
        1. flag_SF                   0.8779
        2. same_srv_rate             0.8378
        3. dst_host_srv_count        0.8207
        4. dst_host_same_srv_rate    0.8086
        5. logged_in                 0.8002
        6. serror_rate               0.7695
        7. srv_serror_rate           0.7657
        8. dst_host_serror_rate      0.7650
        9. dst_host_srv_serror_rate  0.7643
       10. flag_S0                   0.7618
    """

    def __init__(self, detector: UEBADetector, feature_names: list[str]):
        self.detector      = detector
        self.feature_names = feature_names

    def compute(
        self,
        X: np.ndarray,
        top_k: int = 20,
        output_path: str = "results/feature_sensitivity.json"
    ) -> pd.DataFrame:
        """
        Compute Pearson correlation between each feature and anomaly scores.

        Args:
            X: Feature matrix (any split — test set recommended)
            top_k: Number of top features to return
        """
        assert self.detector.is_fitted
        scores = self.detector.decision_scores(X)   # raw IF scores

        correlations = []
        for i, fname in enumerate(self.feature_names):
            col = X[:, i]
            if col.std() < 1e-10:
                r = 0.0
            else:
                r, _ = pearsonr(col, scores)
            correlations.append({
                "feature": fname,
                "correlation": round(float(r), 4),
                "abs_correlation": round(float(abs(r)), 4),
            })

        df = pd.DataFrame(correlations).sort_values(
            "abs_correlation", ascending=False
        ).reset_index(drop=True)

        import os; os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        print(f"[Sensitivity] Saved {len(df)} feature correlations to {output_path}")

        print(f"\nTop {top_k} Features (by |Pearson r| with anomaly score):")
        print(f"{'Rank':<6} {'Feature':<40} {'|r|':>8}")
        print("-" * 56)
        for i, row in df.head(top_k).iterrows():
            print(f"{i+1:<6} {row['feature']:<40} {row['abs_correlation']:>8.4f}")

        return df


# ---------------------------------------------------------------------------
# True Positive extractor
# ---------------------------------------------------------------------------

def extract_true_positives(
    X_test: np.ndarray,
    y_test: np.ndarray,
    detector: UEBADetector
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the True Positive subset: attack samples correctly detected.

    Manuscript: 49,870 TPs from 58,630 total attacks.
    These are the samples used for the adversarial evasion experiments.
    """
    preds     = detector.predict(X_test)
    tp_mask   = (y_test == 1) & (preds == 1)
    fn_mask   = (y_test == 1) & (preds == 0)

    X_tp = X_test[tp_mask]
    print(f"\n[TP Extractor]")
    print(f"  Total attacks in test set : {(y_test==1).sum():,}")
    print(f"  True Positives (detected) : {tp_mask.sum():,}")
    print(f"  False Negatives (missed)  : {fn_mask.sum():,}")
    print(f"  TP subset shape: {X_tp.shape}")

    return X_tp, tp_mask
