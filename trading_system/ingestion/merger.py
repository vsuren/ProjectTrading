import os
import sys
import pyodbc
from datetime import datetime
import zoneinfo

# ============================================================
# PROJECT ROOT INJECTION
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# LOGGING SETUP
# ============================================================

from trading_system.engine.logger import log, set_log_prefix
set_log_prefix("merger_run")   # Log file: merger_run_YYYYMMDD_HHMMSS.txt

TORONTO_TZ = zoneinfo.ZoneInfo("America/Toronto")

# ============================================================
# LOAD CONFIG
# ============================================================

from trading_system.config.config_loader import load_db_config

cfg = load_db_config()
server = cfg["SERVER"]
database = cfg["DATABASE"]
username = cfg["USERNAME"]
password = cfg["PASSWORD"]

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
# MERGE LOGIC WITH FULL LOGGING
# ============================================================

def run_merger():
    start = datetime.now(TORONTO_TZ)
    log("=== MERGE STARTED: Staging -> RawPrices ===")
    log(f"DB Server: {server}, Database: {database}")

    # --------------------------------------------------------
    # 0. VALIDATE staging timestamps BEFORE MERGE
    # --------------------------------------------------------
    log("Validating staging timestamps...")

    cursor.execute("""
        SELECT COUNT(*)
        FROM tblRawPrices_Staging
        WHERE Exported = 0
          AND (DATEPART(HOUR, PriceTimestamp) NOT BETWEEN 9 AND 16)
    """)
    bad_rows = cursor.fetchone()[0]

    if bad_rows > 0:
        log(f"ERROR: {bad_rows} invalid timestamps detected in staging (outside market hours).")
        log("MERGE aborted. Fix ingestion or clean staging before retrying.")
        return

    log("Timestamp validation passed - no invalid rows.")

    # --------------------------------------------------------
    # 1. SHOW pending rows per symbol BEFORE merge
    # --------------------------------------------------------
    log("Fetching pending rows per symbol...")

    cursor.execute("""
        SELECT Symbol, COUNT(*)
        FROM tblRawPrices_Staging
        WHERE Exported = 0
        GROUP BY Symbol
        ORDER BY Symbol;
    """)

    pending = cursor.fetchall()

    if not pending:
        log("No pending rows in staging. Nothing to merge.")
        return

    log("Pending rows per symbol:")
    for symbol, count in pending:
        log(f"  {symbol}: {count}")

    # --------------------------------------------------------
    # 2. BEGIN TRANSACTION
    # --------------------------------------------------------
    log("Beginning SQL transaction...")
    cursor.execute("BEGIN TRANSACTION;")

    try:
        # --------------------------------------------------------
        # 3. INSERT NEW ROWS ONLY (idempotent + deduped)
        # --------------------------------------------------------
        log("Inserting new rows into tblRawPrices...")

        cursor.execute("""
            ;WITH StagingNormalized AS (
                SELECT
                    s.Symbol,
                    DATEADD(minute, DATEDIFF(minute, 0, s.PriceTimestamp), 0) AS PriceTimestampMinute,
                    s.OpenPrice,
                    s.HighPrice,
                    s.LowPrice,
                    s.ClosePrice,
                    s.Volume,
                    s.BatchID,
                    s.LoadTimestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.Symbol,
                                     DATEADD(minute, DATEDIFF(minute, 0, s.PriceTimestamp), 0)
                        ORDER BY s.LoadTimestamp DESC
                    ) AS rn
                FROM tblRawPrices_Staging s
                WHERE s.Exported = 0
                  AND DATEPART(HOUR, s.PriceTimestamp) BETWEEN 9 AND 16
            ),
            StagingDedup AS (
                SELECT
                    Symbol,
                    PriceTimestampMinute,
                    OpenPrice,
                    HighPrice,
                    LowPrice,
                    ClosePrice,
                    Volume,
                    BatchID,
                    LoadTimestamp
                FROM StagingNormalized
                WHERE rn = 1
            )
            INSERT INTO tblRawPrices
            (Symbol, PriceTimestamp, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume,
             BatchID, LoadTimestamp)
            SELECT 
                d.Symbol,
                d.PriceTimestampMinute,
                d.OpenPrice,
                d.HighPrice,
                d.LowPrice,
                d.ClosePrice,
                d.Volume,
                d.BatchID,
                d.LoadTimestamp
            FROM StagingDedup d
            LEFT JOIN tblRawPrices r
                ON r.Symbol = d.Symbol
               AND DATEADD(minute, DATEDIFF(minute, 0, r.PriceTimestamp), 0)
                   = d.PriceTimestampMinute
            WHERE r.Symbol IS NULL
            ORDER BY d.Symbol, d.PriceTimestampMinute;
        """)

        inserted = cursor.rowcount
        log(f"Inserted {inserted} new rows into tblRawPrices.")

        # --------------------------------------------------------
        # 4. MARK staging rows as exported
        # --------------------------------------------------------
        log("Marking staging rows as Exported...")

        cursor.execute("""
            UPDATE tblRawPrices_Staging
            SET Exported = 1
            WHERE Exported = 0;
        """)

        updated = cursor.rowcount
        log(f"Marked {updated} staging rows as Exported.")

        # --------------------------------------------------------
        # 5. CLEAN DUPLICATES (safety net)
        # --------------------------------------------------------
        log("Cleaning duplicates from tblRawPrices (safety net)...")

        cursor.execute("""
            ;WITH Ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY Symbol, PriceTimestamp
                           ORDER BY LoadTimestamp DESC
                       ) AS rn
                FROM tblRawPrices
            )
            DELETE FROM Ranked WHERE rn > 1;
        """)

        cleaned = cursor.rowcount
        log(f"Duplicate cleanup removed {cleaned} rows.")

        # --------------------------------------------------------
        # 6. COMMIT TRANSACTION
        # --------------------------------------------------------
        log("Committing transaction...")
        cursor.execute("COMMIT;")
        conn.commit()

    except Exception as e:
        cursor.execute("ROLLBACK;")
        conn.commit()
        log("MERGE FAILED - transaction rolled back.")
        log(f"Error: {e}")
        return

    # --------------------------------------------------------
    # 7. SHOW per-symbol merge summary
    # --------------------------------------------------------
    log("Fetching per-symbol merge summary...")

    cursor.execute("""
        SELECT Symbol, COUNT(*)
        FROM tblRawPrices
        WHERE LoadTimestamp >= DATEADD(minute, -10, GETDATE())
        GROUP BY Symbol
        ORDER BY Symbol;
    """)

    merged = cursor.fetchall()

    log("Rows inserted in this MERGE:")
    for symbol, count in merged:
        log(f"  {symbol}: {count}")

    # --------------------------------------------------------
    # 8. FINAL DATABASE SUMMARY
    # --------------------------------------------------------
    cursor.execute("SELECT COUNT(*) FROM tblRawPrices;")
    total = cursor.fetchone()[0]

    log(f"Total rows in RawPrices after merge: {total}")

    total_elapsed = (datetime.now(TORONTO_TZ) - start).total_seconds()
    log(f"=== MERGE COMPLETED in {total_elapsed:.2f} seconds ===")


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    run_merger()