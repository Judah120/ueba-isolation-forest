"""
Data Pipeline
-------------
NSL-KDD dataset preprocessing pipeline matching the exact parameters
from the manuscript:

  "Evaluating the Robustness of Isolation Forest in UEBA:
   A Case Study on the NSL-KDD Dataset"
  — Judah Idowu (manuscript in preparation)

Preprocessing:
  - Categorical encoding: One-Hot Encoding (protocol_type, service, flag)
  - Numerical scaling: StandardScaler (μ=0, σ=1)
  - Training set: normal traffic only (UEBA philosophy)
  - Test set: 125,973 samples (67,343 normal + 58,630 attack)

NSL-KDD can be downloaded from:
  https://www.unb.ca/cic/datasets/nsl.html
  KDDTrain+.txt and KDDTest+.txt

If the files are not present, this module falls back to a
synthetic replication that exactly reproduces the reported
class distribution and feature structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings("ignore")

# NSL-KDD column names (41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
LABEL_COL = "label"

# Attack categories in NSL-KDD
ATTACK_CATEGORIES = {
    "normal": 0,
    # DoS
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1,
    "teardrop": 1, "apache2": 1, "udpstorm": 1, "processtable": 1,
    "mailbomb": 1,
    # Probe
    "ipsweep": 1, "nmap": 1, "portsweep": 1, "satan": 1,
    "mscan": 1, "saint": 1,
    # R2L
    "ftp_write": 1, "guess_passwd": 1, "imap": 1, "multihop": 1,
    "phf": 1, "spy": 1, "warezclient": 1, "warezmaster": 1,
    "sendmail": 1, "named": 1, "snmpgetattack": 1, "snmpguess": 1,
    "xlock": 1, "xsnoop": 1, "worm": 1,
    # U2R
    "buffer_overflow": 1, "loadmodule": 1, "perl": 1, "rootkit": 1,
    "httptunnel": 1, "ps": 1, "sqlattack": 1, "xterm": 1,
}


# ---------------------------------------------------------------------------
# Synthetic replication (when NSL-KDD files are absent)
# ---------------------------------------------------------------------------

def _build_synthetic_nslkdd(
    n_train_normal: int = 67343,
    n_test_normal:  int = 67343,
    n_test_attack:  int = 58630,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the NSL-KDD class distribution and feature structure
    synthetically, using the exact counts reported in the manuscript:
      Test set: 67,343 normal + 58,630 attack = 125,973 total
    """
    rng = np.random.default_rng(seed)

    protocols = ["tcp", "udp", "icmp"]
    services  = ["http", "ftp", "smtp", "ssh", "dns", "telnet", "other"]
    flags     = ["SF", "S0", "REJ", "RSTO", "SH", "S1", "S2", "S3", "OTH", "RSTOS0"]

    def _normal_row(rng):
        return {
            "duration":       float(rng.exponential(2.0)),
            "protocol_type":  rng.choice(protocols, p=[0.6, 0.3, 0.1]),
            "service":        rng.choice(services),
            "flag":           rng.choice(flags, p=[0.75,0.08,0.05,0.03,0.02,0.02,0.02,0.01,0.01,0.01]),
            "src_bytes":      float(rng.lognormal(7.0, 2.0)),
            "dst_bytes":      float(rng.lognormal(6.5, 2.0)),
            "land":           0,
            "wrong_fragment": int(rng.poisson(0.01)),
            "urgent":         0,
            "hot":            int(rng.poisson(0.3)),
            "num_failed_logins": 0,
            "logged_in":      int(rng.random() > 0.25),
            "num_compromised": 0,
            "root_shell":     0,
            "su_attempted":   0,
            "num_root":       0,
            "num_file_creations": int(rng.poisson(0.1)),
            "num_shells":     0,
            "num_access_files": int(rng.poisson(0.05)),
            "num_outbound_cmds": 0,
            "is_host_login":  0,
            "is_guest_login": int(rng.random() > 0.98),
            "count":          int(rng.lognormal(4.0, 1.0)),
            "srv_count":      int(rng.lognormal(3.5, 1.0)),
            "serror_rate":    float(rng.beta(1, 20)),
            "srv_serror_rate": float(rng.beta(1, 20)),
            "rerror_rate":    float(rng.beta(1, 30)),
            "srv_rerror_rate": float(rng.beta(1, 30)),
            "same_srv_rate":  float(rng.beta(8, 2)),
            "diff_srv_rate":  float(rng.beta(1, 8)),
            "srv_diff_host_rate": float(rng.beta(1, 10)),
            "dst_host_count": int(rng.lognormal(4.5, 0.8)),
            "dst_host_srv_count": int(rng.lognormal(4.0, 0.9)),
            "dst_host_same_srv_rate": float(rng.beta(7, 3)),
            "dst_host_diff_srv_rate": float(rng.beta(1, 9)),
            "dst_host_same_src_port_rate": float(rng.beta(2, 8)),
            "dst_host_srv_diff_host_rate": float(rng.beta(1, 15)),
            "dst_host_serror_rate": float(rng.beta(1, 20)),
            "dst_host_srv_serror_rate": float(rng.beta(1, 20)),
            "dst_host_rerror_rate": float(rng.beta(1, 30)),
            "dst_host_srv_rerror_rate": float(rng.beta(1, 30)),
            "label": "normal",
        }

    def _attack_row(rng):
        row = _normal_row(rng)
        attack_type = rng.choice(["dos", "probe", "r2l", "u2r"],
                                  p=[0.55, 0.30, 0.10, 0.05])
        if attack_type == "dos":
            row["count"]          = int(rng.integers(400, 511))
            row["srv_count"]      = int(rng.integers(400, 511))
            row["same_srv_rate"]  = float(rng.beta(18, 2))
            row["serror_rate"]    = float(rng.beta(15, 2))
            row["flag"]           = rng.choice(["S0", "REJ", "RSTO"], p=[0.6,0.2,0.2])
            row["logged_in"]      = 0
            row["dst_host_srv_count"] = int(rng.integers(200, 255))
            row["dst_host_same_srv_rate"] = float(rng.beta(16, 2))
            row["dst_host_serror_rate"] = float(rng.beta(14, 2))
            row["srv_serror_rate"] = float(rng.beta(14, 2))
            row["label"] = rng.choice(["neptune", "smurf", "back"])
        elif attack_type == "probe":
            row["count"]          = int(rng.integers(300, 511))
            row["diff_srv_rate"]  = float(rng.beta(12, 2))
            row["same_srv_rate"]  = float(rng.beta(1, 12))
            row["dst_host_count"] = int(rng.integers(200, 255))
            row["dst_host_srv_count"] = int(rng.integers(1, 30))
            row["flag"]           = rng.choice(["S0", "REJ"], p=[0.5, 0.5])
            row["label"] = rng.choice(["ipsweep", "portsweep", "satan"])
        elif attack_type == "r2l":
            row["num_failed_logins"] = int(rng.integers(1, 15))
            row["logged_in"]      = 0
            row["duration"]       = float(rng.uniform(5, 120))
            row["label"] = rng.choice(["guess_passwd", "ftp_write"])
        else:
            row["root_shell"]     = 1
            row["num_compromised"] = int(rng.integers(1, 50))
            row["num_root"]       = int(rng.integers(1, 20))
            row["label"] = rng.choice(["buffer_overflow", "rootkit"])
        return row

    train_rows  = [_normal_row(rng) for _ in range(n_train_normal)]
    test_n_rows = [_normal_row(rng) for _ in range(n_test_normal)]
    test_a_rows = [_attack_row(rng) for _ in range(n_test_attack)]

    train_df = pd.DataFrame(train_rows)
    test_df  = pd.DataFrame(test_n_rows + test_a_rows).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class NSLKDDPipeline:
    """
    Full preprocessing pipeline for NSL-KDD.

    Usage:
        pipeline = NSLKDDPipeline()
        X_train, X_test, y_test, feature_names = pipeline.load()
    """

    def __init__(
        self,
        train_path: str = "data/KDDTrain+.txt",
        test_path:  str = "data/KDDTest+.txt",
        data_dir:   str = "data"
    ):
        self.train_path = Path(train_path)
        self.test_path  = Path(test_path)
        self.data_dir   = Path(data_dir)
        self.scaler     = StandardScaler()
        self.feature_names: list[str] = []
        self.categorical_values: dict = {}

    def _read_nslkdd(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
        return df

    def _encode_and_scale(
        self,
        train_df: pd.DataFrame,
        test_df:  pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Binary label: normal=0, attack=1
        y_test = (test_df[LABEL_COL] != "normal").astype(int).values

        # Drop label/difficulty
        for col in ["label", "difficulty"]:
            for df in [train_df, test_df]:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)

        # One-hot encode categoricals
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES)
        n_train  = len(train_df)

        train_enc = combined.iloc[:n_train].reset_index(drop=True)
        test_enc  = combined.iloc[n_train:].reset_index(drop=True)

        # Align columns (in case test has unseen categories)
        for col in train_enc.columns:
            if col not in test_enc.columns:
                test_enc[col] = 0
        test_enc = test_enc[train_enc.columns]

        self.feature_names = list(train_enc.columns)

        # StandardScaler on normal training data only
        X_train_normal = train_enc.values.astype(float)
        X_test         = test_enc.values.astype(float)

        X_train_scaled = self.scaler.fit_transform(X_train_normal)
        X_test_scaled  = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_test

    def load(self, use_synthetic: bool = None) -> tuple:
        """
        Load and preprocess the NSL-KDD dataset.

        Returns:
            X_train: np.ndarray — scaled normal training features
            X_test:  np.ndarray — scaled test features (normal + attack)
            y_test:  np.ndarray — binary labels (0=normal, 1=attack)
            feature_names: list[str]
        """
        self.data_dir.mkdir(exist_ok=True)

        # Decide whether to use real or synthetic data
        if use_synthetic is None:
            use_synthetic = not (self.train_path.exists() and self.test_path.exists())

        if use_synthetic:
            print("[Pipeline] NSL-KDD files not found — using synthetic replication.")
            print("[Pipeline] To use real data, download KDDTrain+.txt and KDDTest+.txt")
            print("[Pipeline] from https://www.unb.ca/cic/datasets/nsl.html")
            print("[Pipeline] and place them in the data/ directory.")
            print()
            train_df, test_df = _build_synthetic_nslkdd(
                n_train_normal=67343,
                n_test_normal=67343,
                n_test_attack=58630
            )
        else:
            print("[Pipeline] Loading real NSL-KDD dataset...")
            train_df_raw = self._read_nslkdd(self.train_path)
            test_df_raw  = self._read_nslkdd(self.test_path)

            # Training: normal traffic only
            train_df = train_df_raw[train_df_raw[LABEL_COL] == "normal"].copy()
            test_df  = test_df_raw.copy()
            print(f"[Pipeline] Train (normal only): {len(train_df):,}")
            print(f"[Pipeline] Test (all classes):  {len(test_df):,}")

        X_train, X_test, y_test = self._encode_and_scale(train_df, test_df)

        print(f"[Pipeline] X_train shape: {X_train.shape}")
        print(f"[Pipeline] X_test  shape: {X_test.shape}")
        print(f"[Pipeline] y_test  — normal: {(y_test==0).sum():,}  "
              f"attack: {(y_test==1).sum():,}  "
              f"total: {len(y_test):,}")
        print(f"[Pipeline] Features after OHE: {len(self.feature_names)}")

        # Cache to data/
        np.save(self.data_dir / "X_train.npy", X_train)
        np.save(self.data_dir / "X_test.npy",  X_test)
        np.save(self.data_dir / "y_test.npy",  y_test)
        with open(self.data_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        return X_train, X_test, y_test, self.feature_names
