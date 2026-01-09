import os
import sys
import pyodbc
import yfinance as yf
import pandas as pd
import uuid
from datetime import datetime
import time as time_module
import zoneinfo

# ============================================================
# PROJECT ROOT INJECTION (fix imports)
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# LOGGING SETUP
# ============================================================

from trading_system.engine.logger import log, set_log_prefix
set_log_prefix("ingestion_run")   # <--- ensures filename: ingestion_run_YYYYMMDD_HHMMSS.txt

TORONTO_TZ = zoneinfo.ZoneInfo("America/Toronto")

# ============================================================
# 1. LOAD CONFIG
# ============================================================

from trading_system.config.config_loader import load_db_config

cfg = load_db_config()

server = cfg["SERVER"]
database = cfg["DATABASE"]
username = cfg["USERNAME"]
password = cfg["PASSWORD"]

# ============================================================
# USER PARAMETERS (NEW)
# ============================================================

DRY_RUN = False

USE_DATE_RANGE = False   # <--- True = explicit date range, False = original incremental mode
START_DATE = "2026-01-01"
END_DATE   = "2026-01-03"

# ============================================================
# DB CONNECTION
# ============================================================

conn_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# ============================================================
# 2. RETRY LOGIC
# ============================================================

def retry_download(symbol, interval, period, retries=3):
    for attempt in range(retries):
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if not df.empty:
            return df
        log(f"{symbol}: Retry {attempt+1}/{retries}...")
        time_module.sleep(2)
    return df

# ============================================================
# 2b. DATE RANGE DOWNLOAD (NEW)
# ============================================================

def download_range(symbol, start_date, end_date):
    """
    Download 1-minute data for a specific date range using Yahoo Finance.
    """
    try:
        df = yf.download(
            symbol,
            interval="1m",
            start=start_date,
            end=end_date,
            progress=False
        )
        return df
    except Exception as e:
        log(f"{symbol}: Error in date-range download: {e}")
        return pd.DataFrame()

# ============================================================
# 3. INGESTION LOGIC WITH LOGGING
# ============================================================

def ingest_symbol(symbol, backfill_days):
    start = datetime.now(TORONTO_TZ)
    log(f"=== {symbol}: Starting ingestion ===")
    batch_id = str(uuid.uuid4())

    # Get last exported timestamp
    cursor.execute("""
        SELECT MAX(PriceTimestamp)
        FROM tblRawPrices_Staging
        WHERE Symbol = ?
          AND Exported = 1
    """, (symbol,))
    last_exported_ts = cursor.fetchone()[0]

    # ------------------------------------------------------------
    # OPTION 2: DATE RANGE MODE
    # ------------------------------------------------------------
    if USE_DATE_RANGE:
        log(f"{symbol}: Using explicit date range {START_DATE} -> {END_DATE}")
        df = download_range(symbol, START_DATE, END_DATE)

    # ------------------------------------------------------------
    # ORIGINAL BEHAVIOR (incremental/backfill)
    # ------------------------------------------------------------
    else:
        if last_exported_ts is None:
            log(f"{symbol}: New symbol -> downloading {backfill_days} days of 1m data")
            df = retry_download(symbol, "1m", f"{backfill_days}d")
        else:
            log(f"{symbol}: Incremental ingestion -> downloading 1 day of 1m data")
            df = retry_download(symbol, "1m", "1d")

    if df.empty:
        log(f"{symbol}: No data returned from Yahoo")
        return 0, 0

    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "")
        .str.capitalize()
    )

    # Timezone normalization
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    df.index = df.index.tz_localize(None)

    df.index = df.index.map(lambda ts: ts.replace(second=0, microsecond=0))
    df = df[~df.index.duplicated(keep="last")]

    if (not USE_DATE_RANGE) and (last_exported_ts is not None):
        df = df[df.index > last_exported_ts]

    if df.empty:
        log(f"{symbol}: No new rows after filtering")
        return 0, 0

    log(f"{symbol}: Downloaded rows = {len(df)}")
    log(f"{symbol}: First timestamp = {df.index.min()}")
    log(f"{symbol}: Last timestamp  = {df.index.max()}")

    df["Date"] = df.index.date
    rows_inserted = 0
    days_processed = df["Date"].nunique()

    # Insert rows
    insert_start = datetime.now(TORONTO_TZ)

    for row in df.itertuples():
        ts = row.Index.to_pydatetime()

        if DRY_RUN:
            log(f"[DRY RUN] {symbol}: Would insert {ts}")
            continue

        cursor.execute("""
            INSERT INTO tblRawPrices_Staging
            (Symbol, PriceTimestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume,
             BatchID, LoadTimestamp, Exported)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            ts,
            float(row.Open),
            float(row.High),
            float(row.Low),
            float(row.Close),
            int(row.Volume),
            batch_id,
            datetime.now(),
            0
        ))
        rows_inserted += 1

    conn.commit()

    insert_elapsed = (datetime.now(TORONTO_TZ) - insert_start).total_seconds()
    log(f"{symbol}: Insert step completed in {insert_elapsed:.2f} seconds.")

    # Per-day logging
    log(f"{symbol}: Days processed = {days_processed}")

    for date_val, group in df.groupby("Date"):
        log(f"{symbol}: {date_val} -> {len(group)} rows inserted")

    log(f"{symbol}: TOTAL rows inserted = {rows_inserted}")

    total_elapsed = (datetime.now(TORONTO_TZ) - start).total_seconds()
    log(f"{symbol}: Ingestion completed in {total_elapsed:.2f} seconds.")

    return days_processed, rows_inserted

# ============================================================
# 4. LOAD SYMBOLS FROM EXCEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
symbols_path = os.path.join(BASE_DIR, "..", "ingestion", "symbols.xlsx")

symbols_df = pd.read_excel(symbols_path)

symbols_df["Days"] = (
    symbols_df["Days"]
    .astype(str)
    .str.upper()
    .str.replace(r"[^0-9]", "", regex=True)
    .replace("", "7")
    .astype(int)
)

log("Symbols loaded:")
log(str(symbols_df))

# ============================================================
# 5. RUN INGESTION WITH SUMMARY
# ============================================================

def run_ingestion():
    start = datetime.now(TORONTO_TZ)
    log("=== Starting ingestion ===")

    total_symbols = 0
    total_days = 0
    total_rows = 0

    for _, row in symbols_df.iterrows():
        symbol = row["Symbol"]

        if USE_DATE_RANGE:
            log(f"=== Processing {symbol} (DateRange={START_DATE} -> {END_DATE}) ===")
            days, rows = ingest_symbol(symbol, backfill_days=None)
        else:
            backfill_days = int(row["Days"])
            log(f"=== Processing {symbol} (Backfill={backfill_days} days) ===")
            days, rows = ingest_symbol(symbol, backfill_days)

        total_symbols += 1
        total_days += days
        total_rows += rows

    log("=== INGESTION SUMMARY ===")
    log(f"Total symbols processed: {total_symbols}")
    log(f"Total days processed:    {total_days}")
    log(f"Total rows inserted:     {total_rows}")

    total_elapsed = (datetime.now(TORONTO_TZ) - start).total_seconds()
    log(f"=== INGESTION COMPLETED in {total_elapsed:.2f} seconds ===")

if __name__ == "__main__":
    run_ingestion()