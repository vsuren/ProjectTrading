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
from trading_system.features.features_engineered import add_engineered_features


# ============================================================
# 0. SCRIPT-LEVEL DRY RUN FLAG
# ============================================================
DRY_RUN = False


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
# Write to SQL (Dynamic MERGE)
# ============================================================

def write_to_sql(engine, df, dry_run):
    if dry_run:
        log("Dry-run mode: NOT writing to SQL.")
        return

    # Convert BIT columns
    df["IsRegularSession"] = df["IsRegularSession"].astype(int)
    df["IsAfterHours"] = df["IsAfterHours"].astype(int)

    # Convert datetime columns
    df["PriceTimestamp"] = pd.to_datetime(df["PriceTimestamp"], errors="coerce")
    df["RunTimestamp"] = pd.to_datetime(df["RunTimestamp"], errors="coerce")

    # Replace NaN with None
    df = df.where(pd.notnull(df), None)

    # Dynamic column list
    columns = df.columns.tolist()
    col_list = ", ".join(f"[{c}]" for c in columns)

    # Drop temp table
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "IF OBJECT_ID('tempdb..##TempMergeFeatures') IS NOT NULL DROP TABLE ##TempMergeFeatures;"
        )

    # Create temp table with full schema
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"SELECT TOP 0 {col_list} INTO ##TempMergeFeatures FROM dbo.tblMergedFeatures;"
        )

    # Load data
    log(f"Loading {len(df):,} rows into ##TempMergeFeatures via pandas.to_sql...")
    df.to_sql(
        name="##TempMergeFeatures",
        con=engine,
        if_exists="append",
        index=False
    )

    # Dynamic MERGE clause
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

    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE ##TempMergeFeatures;")

    log("UPSERT MERGE completed successfully with dynamic column handling.")


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

    log("Applying engineered features...")
    df_merged = add_engineered_features(df_merged)
    log(f"Columns after engineered features: {len(df_merged.columns)}")

    df_merged.sort_values(["Symbol", "PriceTimestamp"], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    write_to_sql(engine, df_merged, DRY_RUN)

    log("==== Merge Features Completed Successfully ====")


if __name__ == "__main__":
    main()