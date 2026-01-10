import os
import sys
import logging
from logging.handlers import RotatingFileHandler
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
from trading_system.features.features_engineered import add_engineered_features


# ============================================================
# 0. SCRIPT-LEVEL DRY RUN FLAG
# ============================================================
DRY_RUN = False


# ============================================================
# Timing Helper
# ============================================================

def log_timing(label, start_time):
    elapsed = time.time() - start_time
    log(f"{label} completed in {elapsed:0.2f} seconds.")


# ============================================================
# Logging Setup
# ============================================================

def setup_logging():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"MERGE_{ts}"
    set_log_prefix(f"merge_features_{ts}")
    log(f"==== Merge Features Started (run_id={run_id}) ====")
    return run_id


# ============================================================
# Build SQLAlchemy Engine (Azure SQL)
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

    log("Creating SQLAlchemy engine...")
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
# Load Raw Prices
# ============================================================

def load_raw_prices(engine):
    query = """
        SELECT
            Symbol,
            PriceTimestamp,
            OpenPrice,
            HighPrice,
            LowPrice,
            ClosePrice,
            Volume
        FROM dbo.tblRawPrices
    """
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} raw price rows.")
    return df


# ============================================================
# Load Indicators
# ============================================================

def load_indicators(engine):
    query = """
        SELECT
            Symbol,
            PriceTimestamp,
            IndicatorName,
            FieldName,
            IndicatorValue
        FROM dbo.tblIndicators
    """
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} indicator rows.")
    return df


# ============================================================
# Pivot Indicators to Wide Format
# ============================================================

def pivot_indicators(df_ind):
    if df_ind.empty:
        log("Indicator table empty — skipping pivot.")
        return pd.DataFrame()

    df_ind = df_ind.copy()
    df_ind["FeatureName"] = df_ind["FieldName"]

    wide = df_ind.pivot_table(
        index=["Symbol", "PriceTimestamp"],
        columns="FeatureName",
        values="IndicatorValue",
        aggfunc="last",
    )

    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()

    log(f"Pivoted indicators to wide format: {wide.shape[1]} columns.")
    return wide


# ============================================================
# Merge Raw + Indicators
# ============================================================

def merge_features(df_prices, df_features, run_id):
    merged = pd.merge(
        df_prices,
        df_features,
        on=["Symbol", "PriceTimestamp"],
        how="left",
    )

    merged["DayOfWeek"] = merged["PriceTimestamp"].dt.dayofweek + 1
    merged["IsRegularSession"] = merged["PriceTimestamp"].dt.hour.between(9, 16)
    merged["IsAfterHours"] = ~merged["IsRegularSession"]
    merged["RunTimestamp"] = datetime.now()
    merged["MergeRunId"] = run_id

    # Identify indicator columns
    indicator_cols = [
        c for c in merged.columns
        if c not in [
            "Symbol", "PriceTimestamp",
            "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "Volume",
            "DayOfWeek", "IsRegularSession", "IsAfterHours",
            "RunTimestamp", "MergeRunId"
        ]
    ]

    # Sanitize indicator numeric values
    for col in indicator_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["SourceIndicatorCount"] = merged[indicator_cols].notna().sum(axis=1)
    merged["MissingIndicatorCount"] = merged[indicator_cols].isna().sum(axis=1)

    log(f"Merged dataset: {merged.shape[0]:,} rows, {merged.shape[1]:,} columns.")
    return merged



# ============================================================
# Write to SQL (Temp Table + pandas.to_sql + MERGE — Option C)
# ============================================================

def write_to_sql(engine, df, dry_run):
    if dry_run:
        log("Dry-run mode: NOT writing to SQL.")
        return

    # Explicit numeric columns
    numeric_cols = [
        "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "Volume",
        "ADX_14", "ATR_14",
        "BBANDS_LOWER_20", "BBANDS_MIDDLE_20", "BBANDS_UPPER_20",
        "BOLL_PCTB_20", "BOLL_WIDTH_20",
        "CCI_20",
        "EMA_20", "EMA_200", "EMA_50", "EMA_9",
        "RSI_14",
        "SMA_20", "SMA_200",
        "STOCH_D_14", "STOCH_K_14",
        "SUPERTREND_10",
        "WILLIAMS_R_14",
        "DayOfWeek",
        "SourceIndicatorCount", "MissingIndicatorCount"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # BIT columns
    df["IsRegularSession"] = df["IsRegularSession"].astype(int)
    df["IsAfterHours"] = df["IsAfterHours"].astype(int)

    # DATETIME columns
    df["PriceTimestamp"] = pd.to_datetime(df["PriceTimestamp"], errors="coerce")
    df["RunTimestamp"] = pd.to_datetime(df["RunTimestamp"], errors="coerce")

    # Replace NaN with None
    df = df.where(pd.notnull(df), None)

    columns = df.columns.tolist()
    col_list = ", ".join(f"[{c}]" for c in columns)

    # 1. Drop temp table if exists
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "IF OBJECT_ID('tempdb..##TempMergeFeatures') IS NOT NULL DROP TABLE ##TempMergeFeatures;"
        )

    # 2. Create empty temp table with correct schema
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"SELECT TOP 0 {col_list} INTO ##TempMergeFeatures FROM dbo.tblMergedFeatures;"
        )

    # 3. Use pandas.to_sql to load data into temp table
    log(f"Loading {len(df):,} rows into ##TempMergeFeatures via pandas.to_sql...")
    df.to_sql(
        name="##TempMergeFeatures",
        con=engine,
        if_exists="append",
        index=False,
        method=None

    )

    # 4. MERGE from temp table into target
    update_clause = ", ".join(
        f"Target.[{c}] = Source.[{c}]"
        for c in columns
        if c not in ["Symbol", "PriceTimestamp"]
    )

    merge_sql = f"""
    MERGE dbo.tblMergedFeatures AS Target
    USING ##TempMergeFeatures AS Source
        ON Target.Symbol = Source.Symbol
       AND Target.PriceTimestamp = Source.PriceTimestamp
    WHEN MATCHED THEN
        UPDATE SET {update_clause}
    WHEN NOT MATCHED THEN
        INSERT ({col_list})
        VALUES ({col_list});
    """

    log("Running MERGE from ##TempMergeFeatures → tblMergedFeatures...")
    with engine.begin() as conn:
        conn.exec_driver_sql(merge_sql)

    # 5. Drop temp table
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE ##TempMergeFeatures;")

    log("UPSERT MERGE completed successfully (pandas.to_sql method).")
# --- Log final SQL schema of tblMergedFeatures ---
    from sqlalchemy import text

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'tblMergedFeatures'
            ORDER BY ORDINAL_POSITION
        """))  
        sql_columns = [row[0] for row in result]

    log(f"Final SQL column count: {len(sql_columns)}")
    log("Final SQL columns:")
    for col in sql_columns:
        log(f"  - {col}")



# ============================================================
# Main
# ============================================================

def main():
    global DRY_RUN

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True

    run_id = setup_logging()
    log(f"Dry-run mode: {DRY_RUN}")

    cfg = load_db_config()
    engine = build_engine(cfg)

    df_prices = load_raw_prices(engine)
    df_ind = load_indicators(engine)
    df_wide = pivot_indicators(df_ind)
    df_merged = merge_features(df_prices, df_wide, run_id)

    print(">>> ENGINEERED FEATURES FUNCTION CALLED <<<")
    df_merged = add_engineered_features(df_merged)
    print(f"Columns after engineered features: {len(df_merged.columns)}")


    df_merged.sort_values(["Symbol", "PriceTimestamp"], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    write_to_sql(engine, df_merged, DRY_RUN)

    log("==== Merge Features Completed Successfully ====")


if __name__ == "__main__":
    main()