import os
import sys
import logging
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
# FLAGS
# ============================================================
DRY_RUN = False
REBUILD_MODE = False
SIGNAL_VERSION = 1  # bump when feature logic changes

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
    run_id = f"SIGNAL_{ts}"
    set_log_prefix(f"signal_features_{ts}")
    log(f"==== Signal Features Started (run_id={run_id}) ====")
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

    log("Creating SQLAlchemy engine for Signal Features...")
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
# Load source data from tblMergedFeatures
# ============================================================
def load_all_from_tblMergedFeatures(engine):
    query = """
        SELECT
            Symbol,
            PriceTimestamp,
            OpenPrice,
            HighPrice,
            LowPrice,
            ClosePrice,
            Volume,
            ADX_14,
            ATR_14,
            BBANDS_LOWER_20,
            BBANDS_MIDDLE_20,
            BBANDS_UPPER_20,
            BOLL_PCTB_20,
            BOLL_WIDTH_20,
            CCI_20,
            EMA_9,
            EMA_20,
            EMA_50,
            EMA_200,
            RSI_14,
            SMA_20,
            SMA_200,
            STOCH_K_14,
            STOCH_D_14,
            SUPERTREND_10,
            WILLIAMS_R_14,
            DayOfWeek,
            IsRegularSession,
            IsAfterHours,
            SourceIndicatorCount,
            MissingIndicatorCount
        FROM dbo.tblMergedFeatures
        ORDER BY Symbol, PriceTimestamp
    """
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} rows from tblMergedFeatures (full backfill).")
    return df

def load_incremental_from_tblMergedFeatures(engine):
    query = """
        WITH LastSignal AS (
            SELECT ISNULL(MAX(FeatureTimestamp), '1900-01-01') AS LastTs
            FROM dbo.tblSignalFeatures
        )
        SELECT
            m.Symbol,
            m.PriceTimestamp,
            m.OpenPrice,
            m.HighPrice,
            m.LowPrice,
            m.ClosePrice,
            m.Volume,
            m.ADX_14,
            m.ATR_14,
            m.BBANDS_LOWER_20,
            m.BBANDS_MIDDLE_20,
            m.BBANDS_UPPER_20,
            m.BOLL_PCTB_20,
            m.BOLL_WIDTH_20,
            m.CCI_20,
            m.EMA_9,
            m.EMA_20,
            m.EMA_50,
            m.EMA_200,
            m.RSI_14,
            m.SMA_20,
            m.SMA_200,
            m.STOCH_K_14,
            m.STOCH_D_14,
            m.SUPERTREND_10,
            m.WILLIAMS_R_14,
            m.DayOfWeek,
            m.IsRegularSession,
            m.IsAfterHours,
            m.SourceIndicatorCount,
            m.MissingIndicatorCount
        FROM dbo.tblMergedFeatures m
        CROSS JOIN LastSignal ls
        WHERE m.PriceTimestamp > ls.LastTs
        ORDER BY m.Symbol, m.PriceTimestamp
    """
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} incremental rows from tblMergedFeatures.")
    return df

# ============================================================
# Feature engineering
# ============================================================
def compute_features(df):
    if df.empty:
        log("No rows to compute features on.")
        return df

    df = df.sort_values(["Symbol", "PriceTimestamp"]).copy()

    grp = df.groupby("Symbol", group_keys=False)

    df["Range_Pct"] = (df["HighPrice"] - df["LowPrice"]) / df["ClosePrice"]
    df["Body_Pct"] = (df["ClosePrice"] - df["OpenPrice"]) / df["ClosePrice"]
    df["Momentum_10"] = grp["ClosePrice"].apply(lambda s: s.pct_change(10))
    df["Volatility_20"] = grp["ClosePrice"].apply(lambda s: s.pct_change().rolling(20).std())
    df["Return_1"] = grp["ClosePrice"].apply(lambda s: s.pct_change(1))
    df["Return_5"] = grp["ClosePrice"].apply(lambda s: s.pct_change(5))
    df["Return_20"] = grp["ClosePrice"].apply(lambda s: s.pct_change(20))

    log("Computed derived ML features (Range_Pct, Body_Pct, Momentum_10, Volatility_20, Returns).")
    return df

def compute_target(df):
    if df.empty:
        return df

    df = df.sort_values(["Symbol", "PriceTimestamp"]).copy()
    grp = df.groupby("Symbol", group_keys=False)
    df["Target"] = grp["ClosePrice"].apply(lambda s: s.shift(-1) / s - 1)
    log("Computed Target (next-bar return).")
    return df

# ============================================================
# Prepare final DataFrame for tblSignalFeatures
# ============================================================
def prepare_signal_features_df(df, run_id):
    if df.empty:
        log("No rows to prepare for tblSignalFeatures.")
        return df

    df = df.copy()
    df["FeatureTimestamp"] = df["PriceTimestamp"]
    df["FeatureRunId"] = run_id
    df["RunDateTime"] = datetime.now()
    df["SignalVersion"] = SIGNAL_VERSION

    feature_cols_for_missing = [
        "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "Volume",
        "ADX_14", "ATR_14",
        "BBANDS_LOWER_20", "BBANDS_MIDDLE_20", "BBANDS_UPPER_20",
        "BOLL_PCTB_20", "BOLL_WIDTH_20",
        "CCI_20",
        "EMA_9", "EMA_20", "EMA_50", "EMA_200",
        "RSI_14",
        "SMA_20", "SMA_200",
        "STOCH_K_14", "STOCH_D_14",
        "SUPERTREND_10",
        "WILLIAMS_R_14",
        "Range_Pct", "Body_Pct",
        "Return_1", "Return_5", "Return_20",
        "Volatility_20", "Momentum_10",
        "Target"
    ]

    existing = [c for c in feature_cols_for_missing if c in df.columns]
    df["MissingFeatureCount"] = df[existing].isna().sum(axis=1)

    cols = [
        "Symbol",
        "FeatureTimestamp",
        "OpenPrice",
        "HighPrice",
        "LowPrice",
        "ClosePrice",
        "Volume",
        "ADX_14",
        "ATR_14",
        "BBANDS_LOWER_20",
        "BBANDS_MIDDLE_20",
        "BBANDS_UPPER_20",
        "BOLL_PCTB_20",
        "BOLL_WIDTH_20",
        "CCI_20",
        "EMA_9",
        "EMA_20",
        "EMA_50",
        "EMA_200",
        "RSI_14",
        "SMA_20",
        "SMA_200",
        "STOCH_K_14",
        "STOCH_D_14",
        "SUPERTREND_10",
        "WILLIAMS_R_14",
        "DayOfWeek",
        "IsRegularSession",
        "IsAfterHours",
        "Return_1",
        "Return_5",
        "Return_20",
        "Volatility_20",
        "Momentum_10",
        "Range_Pct",
        "Body_Pct",
        "Target",
        "SignalVersion",
        "FeatureRunId",
        "RunDateTime",
        "MissingFeatureCount",
        "SourceIndicatorCount",
    ]

    df_out = df[cols].copy()
    log(f"Prepared final DataFrame for tblSignalFeatures: {df_out.shape[0]:,} rows, {df_out.shape[1]} columns.")
    return df_out

# ============================================================
# Truncate tblSignalFeatures (for REBUILD_MODE)
# ============================================================
def truncate_signal_features(engine):
    if DRY_RUN:
        log("[DRY_RUN] Would TRUNCATE TABLE dbo.tblSignalFeatures (skipped).")
        return

    with engine.begin() as conn:
        log("Truncating dbo.tblSignalFeatures...")
        conn.execute(sa.text("TRUNCATE TABLE dbo.tblSignalFeatures;"))
    log("dbo.tblSignalFeatures truncated successfully.")

# ============================================================
# Batched insert into tblSignalFeatures
# ============================================================
def write_signal_features(engine, df, dry_run):
    if df.empty:
        log("No rows to write to tblSignalFeatures.")
        return

    if dry_run:
        log(f"Dry-run mode: NOT writing {len(df):,} rows to tblSignalFeatures.")
        return

    numeric_cols = [
        "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "Volume",
        "ADX_14", "ATR_14",
        "BBANDS_LOWER_20", "BBANDS_MIDDLE_20", "BBANDS_UPPER_20",
        "BOLL_PCTB_20", "BOLL_WIDTH_20",
        "CCI_20",
        "EMA_9", "EMA_20", "EMA_50", "EMA_200",
        "RSI_14",
        "SMA_20", "SMA_200",
        "STOCH_K_14", "STOCH_D_14",
        "SUPERTREND_10",
        "WILLIAMS_R_14",
        "DayOfWeek",
        "Return_1", "Return_5", "Return_20",
        "Volatility_20", "Momentum_10",
        "Range_Pct", "Body_Pct",
        "Target",
        "SignalVersion",
        "MissingFeatureCount",
        "SourceIndicatorCount",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["IsRegularSession"] = df["IsRegularSession"].astype(int)
    df["IsAfterHours"] = df["IsAfterHours"].astype(int)

    # FIXED DATETIME CONVERSION
    df["FeatureTimestamp"] = pd.to_datetime(df["FeatureTimestamp"], errors="coerce")
    df["RunDateTime"] = pd.to_datetime(df["RunDateTime"], errors="coerce")

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
    sql = f"INSERT INTO dbo.tblSignalFeatures ({col_list}) VALUES ({placeholders})"

    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.fast_executemany = True

    batch_size = 100
    total = len(clean_rows)
    log(f"Writing {total:,} rows to tblSignalFeatures in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        end = start + batch_size
        batch = clean_rows[start:end]
        log(f"Inserting batch {start // batch_size + 1} ({len(batch)} rows)...")
        cursor.executemany(sql, batch)
        conn.commit()

    cursor.close()
    conn.close()
    log("All batches inserted into tblSignalFeatures successfully.")

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
        truncate_signal_features(engine)
        df_src = load_all_from_tblMergedFeatures(engine)
    else:
        df_src = load_incremental_from_tblMergedFeatures(engine)

    # SAFETY CHECK — NO NEW DATA
    if df_src.empty:
        log("No new rows found in tblMergedFeatures. Nothing to insert into tblSignalFeatures.")
        log("==== Signal Features Completed (no work) ====")
        return

    t0 = time.time()
    df_feat = compute_features(df_src)
    df_feat = compute_target(df_feat)
    log_timing("Feature + Target computation", t0)

    df_final = prepare_signal_features_df(df_feat, run_id)

    t1 = time.time()
    write_signal_features(engine, df_final, DRY_RUN)
    log_timing("Write to tblSignalFeatures", t1)

    log("==== Signal Features Completed Successfully ====")

if __name__ == "__main__":
    main()