import os
import sys
from datetime import datetime
import argparse
import time

import pandas as pd
import sqlalchemy as sa
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from trading_system.config.config_loader import load_db_config
from trading_system.engine.logger import log, set_log_prefix

# ============================================================
# FLAGS & CONSTANTS
# ============================================================
DRY_RUN = False
REBUILD_MODE = False

LABEL_TYPE = "DIRECTION_5M"
LABEL_VERSION = "v1"
FEATURE_VERSION = "v1"
HORIZON_MINUTES = 5

FLOAT_MAX = 1e308  # SQL-safe float clipping range

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
    run_id = f"LABEL_{ts}"
    set_log_prefix(f"labeled_features_{ts}")
    log(f"==== Labeled Features Started (run_id={run_id}) ====")
    return run_id

# ============================================================
# Build SQLAlchemy engine (Azure SQL)
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

    log("Creating SQLAlchemy engine for Labeled Features...")
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
# Dynamic schema helpers
# ============================================================
def get_merged_columns(engine):
    """
    Retrieve all columns from dbo.tblMergedFeatures in ordinal order.
    """
    query = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'tblMergedFeatures'
        ORDER BY ORDINAL_POSITION
    """
    df_cols = pd.read_sql(query, engine)
    cols = df_cols["COLUMN_NAME"].tolist()
    log(f"tblMergedFeatures columns: {cols}")
    return cols

# ============================================================
# Load source data (dynamic)
# ============================================================
def load_merged_features(engine):
    """
    Load all features from tblMergedFeatures dynamically.
    We always include:
      - Symbol
      - PriceTimestamp
      - ClosePrice (renamed to Close_t later)
    """
    all_cols = get_merged_columns(engine)

    required = {"Symbol", "PriceTimestamp", "ClosePrice"}
    missing = required - set(all_cols)
    if missing:
        raise RuntimeError(f"tblMergedFeatures is missing required columns: {missing}")

    select_cols = ", ".join(f"[{c}]" for c in all_cols)
    query = f"""
        SELECT
            {select_cols}
        FROM dbo.tblMergedFeatures
        ORDER BY Symbol, PriceTimestamp
    """

    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} rows from tblMergedFeatures (dynamic schema).")
    return df

def load_raw_prices_for_horizon(engine):
    query = """
        SELECT
            Symbol,
            PriceTimestamp,
            ClosePrice
        FROM dbo.tblRawPrices
        ORDER BY Symbol, PriceTimestamp
    """
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} rows from tblRawPrices.")
    return df

# ============================================================
# Incremental cutoff
# ============================================================
def load_incremental_cutoff(engine):
    """
    Determine the latest PriceTimestamp already labeled for this horizon/label/version combo.
    """
    query = """
        SELECT
            MAX(PriceTimestamp) AS LastLabeledTs
        FROM dbo.tblLabeledFeatures
        WHERE HorizonMinutes = :horizon
          AND LabelType = :label_type
          AND LabelVersion = :label_version
          AND FeatureVersion = :feature_version
    """

    with engine.connect() as conn:
        result = conn.execute(
            sa.text(query),
            {
                "horizon": HORIZON_MINUTES,
                "label_type": LABEL_TYPE,
                "label_version": LABEL_VERSION,
                "feature_version": FEATURE_VERSION,
            },
        ).fetchone()

    last_ts = result[0] if result and result[0] is not None else None

    if last_ts is None:
        log("No existing labeled rows found. Full backfill will be performed.")
    else:
        log(f"Last labeled PriceTimestamp: {last_ts}. Incremental mode will load newer rows.")

    return last_ts

def filter_incremental_merged_features(df_merged, last_labeled_ts):
    if last_labeled_ts is None:
        return df_merged
    df = df_merged[df_merged["PriceTimestamp"] > last_labeled_ts].copy()
    log(f"Filtered merged features to {len(df):,} incremental rows.")
    return df

# ============================================================
# Label computation
# ============================================================
def compute_future_close(df_raw, horizon_minutes):
    if df_raw.empty:
        log("No raw price rows to compute future close on.")
        return df_raw

    df = df_raw.copy()
    df["PriceTimestamp"] = pd.to_datetime(df["PriceTimestamp"])
    df = df.sort_values(["Symbol", "PriceTimestamp"])

    df["FutureTimestamp"] = df["PriceTimestamp"] + pd.to_timedelta(horizon_minutes, unit="m")

    df_future = df[["Symbol", "PriceTimestamp", "ClosePrice"]].copy()
    df_future = df_future.rename(columns={"PriceTimestamp": "FutureTimestamp", "ClosePrice": "FutureClose_tN"})

    df_merged = pd.merge(
        df,
        df_future,
        on=["Symbol", "FutureTimestamp"],
        how="left",
    )

    df_merged = df_merged.rename(columns={"ClosePrice": "Close_t"})
    return df_merged[["Symbol", "PriceTimestamp", "Close_t", "FutureClose_tN"]]

def join_features_and_labels(df_features, df_future_close):
    """
    Join dynamic feature set with future close.
    df_features: all columns from tblMergedFeatures (dynamic)
    df_future_close: Symbol, PriceTimestamp, Close_t, FutureClose_tN
    """
    if df_features.empty:
        log("No feature rows to join for labeling.")
        return df_features

    df_features = df_features.copy()
    df_features["PriceTimestamp"] = pd.to_datetime(df_features["PriceTimestamp"])

    df_future_close = df_future_close.copy()
    df_future_close["PriceTimestamp"] = pd.to_datetime(df_future_close["PriceTimestamp"])

    if "ClosePrice" in df_features.columns:
        df_features = df_features.rename(columns={"ClosePrice": "Close_t"})

    df = pd.merge(
        df_features,
        df_future_close[["Symbol", "PriceTimestamp", "FutureClose_tN"]],
        on=["Symbol", "PriceTimestamp"],
        how="left",
    )

    log(f"Joined features with future close. Rows: {len(df):,}.")
    return df

def compute_return_and_label(df, horizon_minutes):
    if df.empty:
        log("No rows to compute return and label on.")
        return df

    df = df.copy()

    if "Close_t" not in df.columns:
        raise RuntimeError("Expected column 'Close_t' not found after join.")

    df["Return_t_to_tN"] = np.where(
        df["FutureClose_tN"].notna() & df["Close_t"].notna() & (df["Close_t"] != 0),
        (df["FutureClose_tN"] / df["Close_t"]) - 1.0,
        np.nan,
    )

    df["Label"] = 0
    df.loc[df["Return_t_to_tN"] > 0, "Label"] = 1
    df.loc[df["Return_t_to_tN"] < 0, "Label"] = -1

    df["HorizonMinutes"] = horizon_minutes
    df["LabelType"] = LABEL_TYPE
    df["LabelVersion"] = LABEL_VERSION
    df["FeatureVersion"] = FEATURE_VERSION

    df["IsTrainableRow"] = np.where(df["Return_t_to_tN"].notna(), 1, 0)
    df["IsOutlier"] = 0
    df["RegimeTag"] = None
    df["MarketSession"] = None

    return df

# ============================================================
# Prepare final DataFrame (dynamic)
# ============================================================
def prepare_labeled_features_df(df, run_id):
    if df.empty:
        log("No rows to prepare for tblLabeledFeatures.")
        return df

    df = df.copy()
    df["LoadTimestamp"] = datetime.now()
    df["SourceRunID"] = run_id
    df["RunId"] = run_id
    df_out = df.copy()
    log(f"Prepared final DataFrame (dynamic): {df_out.shape[0]:,} rows, {df_out.shape[1]} columns.")
    return df_out

# ============================================================
# Truncate table
# ============================================================
def truncate_labeled_features(engine):
    if DRY_RUN:
        log("[DRY_RUN] Would TRUNCATE TABLE dbo.tblLabeledFeatures (skipped).")
        return

    with engine.begin() as conn:
        log("Truncating dbo.tblLabeledFeatures...")
        conn.execute(sa.text("TRUNCATE TABLE dbo.tblLabeledFeatures;"))
    log("dbo.tblLabeledFeatures truncated successfully.")

# ============================================================
# Batched insert (dynamic, with clipping & sanitization)
# ============================================================
def write_labeled_features(engine, df, dry_run):
    if df.empty:
        log("No rows to write to tblLabeledFeatures.")
        return

    if dry_run:
        log(f"[DRY_RUN] Would insert {len(df):,} rows into tblLabeledFeatures.")
        return

    df = df.copy()

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Dynamic numeric detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log(f"Numeric columns detected for conversion/clipping: {numeric_cols}")

    # Clip extreme values to SQL-safe float range
    for col in numeric_cols:
        df[col] = df[col].clip(-FLOAT_MAX, FLOAT_MAX)

    # Convert to numeric (coerce errors to NaN)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure timestamps are proper datetimes
    if "PriceTimestamp" in df.columns:
        df["PriceTimestamp"] = pd.to_datetime(df["PriceTimestamp"], errors="coerce")
    if "LoadTimestamp" in df.columns:
        df["LoadTimestamp"] = pd.to_datetime(df["LoadTimestamp"], errors="coerce")

    # Replace NaN with None for DB insert
    df = df.where(pd.notnull(df), None)

    rows = list(df.itertuples(index=False, name=None))

    clean_rows = []
    for r in rows:
        clean_rows.append(
            tuple(None if (isinstance(x, float) and np.isnan(x)) else x for x in r)
        )

    columns = df.columns.tolist()
    col_list = ", ".join(f"[{c}]" for c in columns)
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO dbo.tblLabeledFeatures ({col_list}) VALUES ({placeholders})"

    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.fast_executemany = True

    batch_size = 100
    total = len(clean_rows)
    log(f"Writing {total:,} rows to tblLabeledFeatures in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        end = start + batch_size
        batch = clean_rows[start:end]
        log(f"Inserting batch {start // batch_size + 1} ({len(batch)} rows)...")

        # ============================================================
        # FAST BATCH INSERT (production mode)
        # ============================================================
        cursor.executemany(sql, batch)
        conn.commit()

    cursor.close()
    conn.close()
    log("All batches inserted into tblLabeledFeatures successfully.")


# ============================================================
# Main
# ============================================================
def main():
    global DRY_RUN, REBUILD_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True
    if args.rebuild:
        REBUILD_MODE = True

    run_id = setup_logging()
    log(f"Dry-run mode: {DRY_RUN}, REBUILD_MODE: {REBUILD_MODE}")

    cfg = load_db_config()
    engine = build_engine(cfg)

    if REBUILD_MODE:
        truncate_labeled_features(engine)
        df_merged = load_merged_features(engine)
    else:
        last_labeled_ts = load_incremental_cutoff(engine)
        df_merged_all = load_merged_features(engine)
        df_merged = filter_incremental_merged_features(df_merged_all, last_labeled_ts)

    if df_merged.empty:
        log("No new rows found in tblMergedFeatures for labeling.")
        log("==== Labeled Features Completed (no work) ====")
        return

    df_raw = load_raw_prices_for_horizon(engine)

    if df_raw.empty:
        log("No raw prices available to compute future close.")
        log("==== Labeled Features Completed (no work) ====")
        return

    t0 = time.time()
    df_future_close = compute_future_close(df_raw, HORIZON_MINUTES)
    df_joined = join_features_and_labels(df_merged, df_future_close)
    df_labeled = compute_return_and_label(df_joined, HORIZON_MINUTES)
    log_timing("Label computation", t0)

    df_final = prepare_labeled_features_df(df_labeled, run_id)

    t1 = time.time()
    write_labeled_features(engine, df_final, DRY_RUN)
    log_timing("Write to tblLabeledFeatures", t1)

    log("==== Labeled Features Completed Successfully ====")


if __name__ == "__main__":
    main()