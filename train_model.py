import os
import sys
from datetime import datetime
import argparse
import time

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from trading_system.config.config_loader import load_db_config
from trading_system.engine.logger import log, set_log_prefix

# ============================================================
# FLAGS & CONSTANTS
# ============================================================
DRY_RUN = False

LABEL_TYPE = "DIRECTION_5M"
LABEL_VERSION = "v1"
FEATURE_VERSION = "v1"
HORIZON_MINUTES = 5

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Timing helper
# ============================================================
def log_timing(label, start_time):
    elapsed = time.time() - start_time
    log(f"{label} completed in {elapsed:0.2f} seconds.")

# ============================================================
# Logging setup
# ============================================================
def setup_logging():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"TRAIN_{ts}"
    set_log_prefix(f"train_model_{ts}")
    log(f"==== Model Training Started (run_id={run_id}) ====")
    return run_id

# ============================================================
# Build SQLAlchemy engine (reuse pattern from labeling script)
# ============================================================
def build_engine(cfg):
    server = cfg["SERVER"]
    database = cfg["DATABASE"]
    username = cfg["USERNAME"]
    password = cfg["PASSWORD"]

    conn_str = (
        f"mssql+pyodbc://{username}:{password}"
        f"@{server}/{database}"
        f"?driver=ODBC+Driver+18+for+SQL+Server"
        f"&Encrypt=yes&TrustServerCertificate=no"
    )

    log("Creating SQLAlchemy engine for Model Training...")
    try:
        engine = sa.create_engine(conn_str, fast_executemany=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        log("Connected to Azure SQL successfully.")
        return engine
    except Exception as e:
        log(f"Failed to connect to Azure SQL: {e}")
        raise

# ============================================================
# Load labeled data
# ============================================================
def load_labeled_features(engine):
    """
    Load labeled features for the given horizon/label/version combo.
    We keep this dynamic and only filter by metadata columns.
    """
    query = """
        SELECT *
        FROM dbo.tblLabeledFeatures
        WHERE HorizonMinutes = :horizon
          AND LabelType = :label_type
          AND LabelVersion = :label_version
          AND FeatureVersion = :feature_version
        ORDER BY Symbol, PriceTimestamp
    """

    df = pd.read_sql(
        sa.text(query),
        engine,
        params={
            "horizon": HORIZON_MINUTES,
            "label_type": LABEL_TYPE,
            "label_version": LABEL_VERSION,
            "feature_version": FEATURE_VERSION,
        },
    )

    log(f"Loaded {len(df):,} rows from tblLabeledFeatures for training.")
    return df

# ============================================================
# Build X and y dynamically
# ============================================================
def build_features_and_labels(df):
    if df.empty:
        log("No rows available in tblLabeledFeatures for training.")
        return None, None, None

    df = df.copy()

    # Ensure Label exists
    if "Label" not in df.columns:
        raise RuntimeError("Expected column 'Label' not found in tblLabeledFeatures.")

    # Drop non-feature/meta columns
    meta_cols = {
        "Symbol",
        "PriceTimestamp",
        "LoadTimestamp",
        "SourceRunID",
        "RunId",
        "HorizonMinutes",
        "LabelType",
        "LabelVersion",
        "FeatureVersion",
        "IsTrainableRow",
        "IsOutlier",
        "RegimeTag",
        "MarketSession",
	"FutureClose_tN",
	"Return_t_to_tN"
    }

    # y = Label
    y = df["Label"]

    # Candidate feature columns = all numeric columns excluding meta + Label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in numeric_cols
        if c not in meta_cols and c != "Label"
    ]

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for training.")

    X = df[feature_cols]

    log(f"Feature columns used for training ({len(feature_cols)}): {feature_cols}")
    log(f"Training dataset shape: X={X.shape}, y={y.shape}")

    return X, y, feature_cols

# ============================================================
# Train/test split (time-based)
# ============================================================
def time_based_split(X, y, train_ratio=0.8):
    n = len(X)
    if n == 0:
        raise RuntimeError("No rows available for train/test split.")

    split_index = int(n * train_ratio)
    if split_index == 0 or split_index == n:
        raise RuntimeError("Train/test split resulted in empty train or test set.")

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    log(
        f"Time-based split at index {split_index} "
        f"(train={len(X_train):,}, test={len(X_test):,})."
    )

    return X_train, X_test, y_train, y_test

# ============================================================
# Train model
# ============================================================
def train_model(X_train, y_train):
    log("Initializing HistGradientBoostingClassifier...")

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        class_weight="balanced"
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    log_timing("Model training", t0)

    return model
# ============================================================
# Evaluate model
# ============================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    log("Evaluating model...")

    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Test accuracy
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    log(f"Training Accuracy: {train_acc:0.4f}")
    log(f"Test Accuracy: {test_acc:0.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    log("Confusion Matrix:")
    log(str(cm))

    # Classification report
    report = classification_report(y_test, y_test_pred)
    log("Classification Report:")
    log("\n" + report)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }

# ============================================================
# Save model
# ============================================================
def save_model(model, feature_cols, run_id):
    """
    Save the trained model and metadata (feature columns, run_id, versions).
    """
    model_payload = {
        "model": model,
        "feature_columns": feature_cols,
        "label_type": LABEL_TYPE,
        "label_version": LABEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "horizon_minutes": HORIZON_MINUTES,
        "trained_run_id": run_id,
        "trained_at": datetime.utcnow().isoformat(),
    }

    filename = f"model_{LABEL_TYPE}_{HORIZON_MINUTES}m_{LABEL_VERSION}_{FEATURE_VERSION}_{run_id}.pkl"
    filepath = os.path.join(MODEL_DIR, filename)

    joblib.dump(model_payload, filepath)
    log(f"Model saved to: {filepath}")

    return filepath

### MODEL DIAGNOSTICS: PERMUTATION FEATURE IMPORTANCE (HGB COMPATIBLE)
from sklearn.inspection import permutation_importance

def log_feature_importance(model, X_test, y_test, feature_cols):
    """
    Computes and logs permutation feature importance for any model,
    including HistGradientBoostingClassifier.
    """
    log("Computing permutation feature importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    importances = result.importances_mean

    fi = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )

    log("=== Permutation Feature Importances (Descending) ===")
    for name, score in fi:
        log(f"{name}: {score:0.5f}")
# ============================================================
# Main
# ============================================================
def main():
    global DRY_RUN

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Train and evaluate but do NOT save the model.")
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True

    run_id = setup_logging()
    log(f"Dry-run mode: {DRY_RUN}")

    cfg = load_db_config()
    engine = build_engine(cfg)

    # Load labeled data
    t0 = time.time()
    df_labeled = load_labeled_features(engine)
    log_timing("Load labeled features", t0)

    if df_labeled.empty:
        log("No labeled rows found for training. Exiting.")
        log("==== Model Training Completed (no work) ====")
        return

    # Build X, y
    X, y, feature_cols = build_features_and_labels(df_labeled)
    if X is None or y is None:
        log("No features or labels available for training. Exiting.")
        log("==== Model Training Completed (no work) ====")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    eval_results = evaluate_model(model, X_train, y_train, X_test, y_test)

    # >>> ADD THIS LINE RIGHT HERE <<<
    log_feature_importance(model, X_test, y_test, feature_cols)

    # Save model (only if not dry-run)
    if DRY_RUN:
        log("[DRY_RUN] Skipping model save.")
    else:
        save_model(model, feature_cols, run_id)

    log("==== Model Training Completed Successfully ====")

if __name__ == "__main__":
    main()