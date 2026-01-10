import os
import sys
from datetime import datetime
import argparse
import time
import json

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import joblib

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from trading_system.config.config_loader import load_db_config
from trading_system.engine.logger import log, set_log_prefix


# ============================================================
# SECTION: FLAGS & GLOBAL CONFIGURATION
# ============================================================

DRY_RUN = True  # FALSE

LABEL_TYPE = "DIRECTION_5M"
LABEL_VERSION = "v1"
FEATURE_VERSION = "v1"
HORIZON_MINUTES = 5

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SECTION: TIMING & LOGGING HELPERS
# ============================================================

def log_timing(label: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    log(f"{label} completed in {elapsed:0.2f} seconds.")


def setup_logging() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"TRAIN_{ts}"
    set_log_prefix(f"train_model_{ts}")
    log(f"==== Model Training Started (run_id={run_id}) ====")
    return run_id


# ============================================================
# SECTION: DATABASE CONNECTION & HELPERS
# ============================================================

def build_engine(cfg) -> sa.engine.Engine:
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


def load_active_feature_set(engine) -> list:
    query = sa.text("""
        SELECT fsm.FeatureName
        FROM dbo.tblFeatureSetMembers fsm
        INNER JOIN dbo.tblFeatureSets fs
            ON fs.FeatureSetID = fsm.FeatureSetID
        WHERE fs.IsActive = 1
        ORDER BY fsm.Ordinal;
    """)

    df = pd.read_sql(query, engine)
    feature_names = df["FeatureName"].tolist()

    if not feature_names:
        raise RuntimeError("No active feature set found or it contains no features.")

    log(f"Active feature set loaded with {len(feature_names)} features: {feature_names}")
    return feature_names


def get_active_feature_set_id(engine) -> int:
    query = sa.text("""
        SELECT FeatureSetID
        FROM dbo.tblFeatureSets
        WHERE IsActive = 1;
    """)
    df = pd.read_sql(query, engine)

    if df.empty:
        raise RuntimeError("No active feature set found in tblFeatureSets (IsActive = 1).")

    if len(df) > 1:
        raise RuntimeError("Multiple active feature sets found. Ensure only one has IsActive = 1.")

    feature_set_id = int(df.iloc[0]["FeatureSetID"])
    log(f"Active FeatureSetID: {feature_set_id}")
    return feature_set_id


# ============================================================
# SECTION: DATA LOADING
# ============================================================

def load_labeled_features(engine) -> pd.DataFrame:
    query = """
        SELECT *
        FROM dbo.tblLabeledFeatures
        WHERE HorizonMinutes = :horizon
          AND LabelType = :label_type
          AND LabelVersion = :label_version
          AND FeatureVersion = :feature_version
        ORDER BY Symbol, PriceTimestamp;
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
# SECTION: FEATURE & LABEL CONSTRUCTION
# ============================================================

def build_features_and_labels(df: pd.DataFrame, active_feature_names: list):
    if df.empty:
        log("No rows available in tblLabeledFeatures for training.")
        return None, None, None

    df = df.copy()

    if "Label" not in df.columns:
        raise RuntimeError("Expected column 'Label' not found in tblLabeledFeatures.")

    y = df["Label"]

    available_features = [c for c in active_feature_names if c in df.columns]
    missing_features = [c for c in active_feature_names if c not in df.columns]

    if missing_features:
        log(f"[WARNING] Missing features: {missing_features}")

    if not available_features:
        raise RuntimeError("None of the active feature set columns were found.")

    X = df[available_features]

    log(f"Feature columns used for training ({len(available_features)}): {available_features}")
    log(f"Training dataset shape: X={X.shape}, y={y.shape}")

    return X, y, available_features


# ============================================================
# SECTION: TRAIN/TEST SPLIT
# ============================================================

def time_based_split(X, y, train_ratio: float = 0.8):
    n = len(X)
    split_index = int(n * train_ratio)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    log(f"Time-based split at index {split_index} (train={len(X_train):,}, test={len(X_test):,}).")

    return X_train, X_test, y_train, y_test


# ============================================================
# SECTION: MODEL TRAINING
# ============================================================

def train_model(X_train, y_train):
    log("Initializing HistGradientBoostingClassifier...")

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        class_weight="balanced",
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    log_timing("Model training", t0)

    return model


# ============================================================
# SECTION: MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    log("Evaluating model...")

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    log(f"Training Accuracy: {train_acc:0.4f}")
    log(f"Test Accuracy: {test_acc:0.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    log("Confusion Matrix:")
    log(str(cm))

    report = classification_report(y_test, y_test_pred)
    log("Classification Report:")
    log("\n" + report)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def log_feature_importance(model, X_test, y_test, feature_cols):
    log("Computing permutation feature importance...")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importances = result.importances_mean

    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    log("=== Permutation Feature Importances (Descending) ===")
    for name, score in fi:
        log(f"{name}: {score:0.5f}")


def build_metrics_json(eval_results: dict) -> str:
    payload = {
        "training_accuracy": float(eval_results["train_accuracy"]),
        "test_accuracy": float(eval_results["test_accuracy"]),
        "confusion_matrix": eval_results["confusion_matrix"].tolist(),
        "classification_report": eval_results["classification_report"],
    }
    return json.dumps(payload)


# ============================================================
# SECTION: MODEL PERSISTENCE
# ============================================================

def save_model(model, feature_cols, run_id: str) -> str:
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


# ============================================================
# SECTION: MODEL REGISTRY
# ============================================================

def register_model(
    engine,
    model_name: str,
    run_id: str,
    feature_set_id: int,
    label_type: str,
    label_version: str,
    horizon_minutes: int,
    metrics_json: str,
    artifact_path: str,
) -> None:

    query = sa.text("""
        INSERT INTO dbo.tblModelRegistry (
            ModelName,
            RunID,
            FeatureSetID,
            LabelType,
            LabelVersion,
            HorizonMinutes,
            MetricsJson,
            ArtifactPath
        )
        VALUES (
            :ModelName,
            :RunID,
            :FeatureSetID,
            :LabelType,
            :LabelVersion,
            :HorizonMinutes,
            :MetricsJson,
            :ArtifactPath
        );
    """)

    params = {
        "ModelName": model_name,
        "RunID": run_id,
        "FeatureSetID": feature_set_id,
        "LabelType": label_type,
        "LabelVersion": label_version,
        "HorizonMinutes": horizon_minutes,
        "MetricsJson": metrics_json,
        "ArtifactPath": artifact_path,
    }

    log("Registering model in dbo.tblModelRegistry...")
    with engine.begin() as conn:
        conn.execute(query, params)
    log("Model registered successfully in tblModelRegistry.")


# ============================================================
# SECTION: MAIN ORCHESTRATION
# ============================================================

def main():
    global DRY_RUN

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True  # TRUE

    run_id = setup_logging()
    log(f"Dry-run mode: {DRY_RUN}")

    cfg = load_db_config()
    engine = build_engine(cfg)

    t0 = time.time()
    df_labeled = load_labeled_features(engine)
    log_timing("Load labeled features", t0)

    if df_labeled.empty:
        log("No labeled rows found. Exiting.")
        return

    active_feature_names = load_active_feature_set(engine)
    active_feature_set_id = get_active_feature_set_id(engine)

    X, y, feature_cols = build_features_and_labels(df_labeled, active_feature_names)
    if X is None:
        return

    X_train, X_test, y_train, y_test = time_based_split(X, y)

    model = train_model(X_train, y_train)

    eval_results = evaluate_model(model, X_train, y_train, X_test, y_test)

    log_feature_importance(model, X_test, y_test, feature_cols)

    if DRY_RUN:
        log("[DRY_RUN] Skipping model save and registry insertion.")
    else:
        model_path = save_model(model, feature_cols, run_id)
        metrics_json = build_metrics_json(eval_results)

        model_name = f"HGB_{LABEL_TYPE}_{HORIZON_MINUTES}m_{LABEL_VERSION}_{FEATURE_VERSION}"

        register_model(
            engine=engine,
            model_name=model_name,
            run_id=run_id,
            feature_set_id=active_feature_set_id,
            label_type=LABEL_TYPE,
            label_version=LABEL_VERSION,
            horizon_minutes=HORIZON_MINUTES,
            metrics_json=metrics_json,
            artifact_path=model_path,
        )

    log("==== Model Training Completed Successfully ====")


if __name__ == "__main__":
    main()