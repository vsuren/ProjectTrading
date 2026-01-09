"""
test_database_integrity.py
Production‑grade database integrity validation suite
"""

# -------------------------------------------------------------------
# UNIVERSAL IMPORT FIX (must be first)
# -------------------------------------------------------------------
import sys
sys.path.insert(0, r"E:\ProjectTrading\trading_system")

# -------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------
import traceback
import pyodbc
import pandas as pd
from engine.logger import log, set_log_prefix
from config.config_loader import load_db_config

# -------------------------------------------------------------------
# DB CONNECTION (Azure SQL)
# -------------------------------------------------------------------
def get_connection():
    cfg = load_db_config()
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={cfg['SERVER']};"
        f"DATABASE={cfg['DATABASE']};"
        f"UID={cfg['USERNAME']};"
        f"PWD={cfg['PASSWORD']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)

# -------------------------------------------------------------------
# GENERIC TEST RUNNER
# -------------------------------------------------------------------
def run_test(name, func, **params):
    log(f"[TEST] --- Testing {name} ---")
    try:
        ok, details = func(**params)

        if ok:
            log(f"[TEST] {name}: PASS")
            if details:
                log(f"[TEST] {name} DETAILS: {details}")
        else:
            log(f"[TEST] {name}: FAIL")
            if details:
                log(f"[TEST] {name} DETAILS: {details}")

        return ok

    except Exception as e:
        log(f"[TEST] {name}: FAIL")
        log(str(e))
        log(traceback.format_exc())
        return False

# -------------------------------------------------------------------
# INTEGRITY CHECKS
# -------------------------------------------------------------------

# 1) TABLE EXISTENCE
def check_table_exists(table_name):
    conn = get_connection()
    query = """
        SELECT 1
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_NAME = ?
    """
    cur = conn.cursor()
    cur.execute(query, (table_name,))
    row = cur.fetchone()
    conn.close()

    if row:
        return True, f"Table [{table_name}] exists."
    else:
        return False, f"Table [{table_name}] is missing."

# 2) COLUMN EXISTENCE
def check_columns_exist(table_name, required_columns):
    conn = get_connection()
    query = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ?
    """
    df = pd.read_sql(query, conn, params=[table_name])
    conn.close()

    existing = set(df["COLUMN_NAME"].tolist())
    missing = [c for c in required_columns if c not in existing]

    if missing:
        return False, f"Missing columns in [{table_name}]: {', '.join(missing)}"
    else:
        return True, f"All required columns exist in [{table_name}]."

# 3) NULL CHECKS
def check_nulls(table_name, critical_columns):
    conn = get_connection()
    cols = ", ".join(f"[{c}]" for c in critical_columns)
    query = f"SELECT {cols} FROM [{table_name}]"
    df = pd.read_sql(query, conn)
    conn.close()

    null_report = {}
    for col in critical_columns:
        cnt = df[col].isna().sum()
        if cnt > 0:
            null_report[col] = cnt

    if null_report:
        details = "; ".join(f"{col}: {cnt} NULLs" for col, cnt in null_report.items())
        return False, f"NULLs detected in [{table_name}] -> {details}"
    else:
        return True, f"No NULLs in critical columns of [{table_name}]."

# 4) DUPLICATE CHECKS
def check_duplicates_rawprices():
    table_name = "tblRawPrices"
    conn = get_connection()
    query = """
        SELECT [Symbol], [PriceTimestamp], COUNT(*) AS Cnt
        FROM [tblRawPrices]
        GROUP BY [Symbol], [PriceTimestamp]
        HAVING COUNT(*) > 1
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        return True, "No duplicate (Symbol, PriceTimestamp) pairs in [tblRawPrices]."
    else:
        return False, f"Found {len(df)} duplicate (Symbol, PriceTimestamp) pairs in [tblRawPrices]."

# 5) RANGE CHECKS (PRICES > 0, VOLUME >= 0)
def check_price_volume_ranges():
    table_name = "tblRawPrices"
    conn = get_connection()
    query = """
        SELECT
            SUM(CASE WHEN [OpenPrice]  <= 0 THEN 1 ELSE 0 END) AS BadOpen,
            SUM(CASE WHEN [HighPrice]  <= 0 THEN 1 ELSE 0 END) AS BadHigh,
            SUM(CASE WHEN [LowPrice]   <= 0 THEN 1 ELSE 0 END) AS BadLow,
            SUM(CASE WHEN [ClosePrice] <= 0 THEN 1 ELSE 0 END) AS BadClose,
            SUM(CASE WHEN [Volume]     <  0 THEN 1 ELSE 0 END) AS BadVolume
        FROM [tblRawPrices]
    """
    df = pd.read_sql(query, conn)
    conn.close()

    row = df.iloc[0]
    issues = []
    if row["BadOpen"] > 0:
        issues.append(f"OpenPrice <= 0: {int(row['BadOpen'])}")
    if row["BadHigh"] > 0:
        issues.append(f"HighPrice <= 0: {int(row['BadHigh'])}")
    if row["BadLow"] > 0:
        issues.append(f"LowPrice <= 0: {int(row['BadLow'])}")
    if row["BadClose"] > 0:
        issues.append(f"ClosePrice <= 0: {int(row['BadClose'])}")
    if row["BadVolume"] > 0:
        issues.append(f"Volume < 0: {int(row['BadVolume'])}")

    if issues:
        return False, "; ".join(issues)
    else:
        return True, "All prices > 0 and Volume >= 0 in [tblRawPrices]."

# 6) ROW COUNT SANITY
def check_rowcount_min(table_name, min_rows):
    conn = get_connection()
    query = f"SELECT COUNT(*) AS Cnt FROM [{table_name}]"
    df = pd.read_sql(query, conn)
    conn.close()

    cnt = int(df["Cnt"].iloc[0])
    if cnt >= min_rows:
        return True, f"[{table_name}] has {cnt} rows (>= {min_rows})."
    else:
        return False, f"[{table_name}] has only {cnt} rows (< {min_rows})."

# 7) BASIC REFERENTIAL SANITY: Indicators vs RawPrices
def check_indicators_reference_rawprices():
    """
    Assumes tblIndicators has Symbol, PriceTimestamp columns
    that should match tblRawPrices.
    Adjust column names if your schema differs.
    """
    conn = get_connection()
    query = """
        SELECT COUNT(*) AS Orphans
        FROM [tblIndicators] i
        LEFT JOIN [tblRawPrices] r
            ON  i.[Symbol] = r.[Symbol]
            AND i.[PriceTimestamp] = r.[PriceTimestamp]
        WHERE r.[Symbol] IS NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()

    orphans = int(df["Orphans"].iloc[0])
    if orphans == 0:
        return True, "No orphan indicator rows (all reference existing raw prices)."
    else:
        return False, f"{orphans} indicator rows do not match any raw price row."

# 8) INDEX EXISTENCE CHECK
def check_index_exists(table_name, indexed_columns):
    """
    indexed_columns = ["Symbol", "PriceTimestamp"] etc.
    """
    conn = get_connection()

    query = """
        SELECT 
            c.name AS ColumnName,
            i.name AS IndexName
        FROM sys.indexes i
        JOIN sys.index_columns ic 
            ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        JOIN sys.columns c 
            ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        WHERE OBJECT_NAME(i.object_id) = ?
    """

    df = pd.read_sql(query, conn, params=[table_name])
    conn.close()

    indexed = set(df["ColumnName"].tolist())
    missing = [col for col in indexed_columns if col not in indexed]

    if missing:
        return False, f"Missing indexes on [{table_name}] columns: {', '.join(missing)}"
    else:
        return True, f"All required indexed columns exist on [{table_name}]."

# -------------------------------------------------------------------
# MAIN TEST SUITE
# -------------------------------------------------------------------
def main():
    set_log_prefix("test_database_integrity")
    log("=== TEST RUN: Database Integrity Validation Suite ===")

    tests = [
        # Table existence
        ("TableExists_tblRawPrices",   check_table_exists, {"table_name": "tblRawPrices"}),
        ("TableExists_tblIndicators",  check_table_exists, {"table_name": "tblIndicators"}),

        # Column existence for tblRawPrices
        ("Columns_tblRawPrices",       check_columns_exist, {
            "table_name": "tblRawPrices",
            "required_columns": [
                "Symbol", "PriceTimestamp",
                "OpenPrice", "HighPrice", "LowPrice", "ClosePrice",
                "Volume"
            ]
        }),

        # Null checks on critical columns
        ("Nulls_tblRawPrices",         check_nulls, {
            "table_name": "tblRawPrices",
            "critical_columns": [
                "Symbol", "PriceTimestamp",
                "OpenPrice", "HighPrice", "LowPrice", "ClosePrice",
                "Volume"
            ]
        }),

        # Duplicates
        ("Duplicates_tblRawPrices",    check_duplicates_rawprices, {}),

        # Range checks
        ("Ranges_tblRawPrices",        check_price_volume_ranges, {}),

        # Row count sanity
        ("RowCount_tblRawPrices",      check_rowcount_min, {
            "table_name": "tblRawPrices",
            "min_rows": 1000
        }),

        # Referential sanity
        ("Indicators_vs_RawPrices",    check_indicators_reference_rawprices, {}),
    ]

    results = []
    for name, func, params in tests:
        ok = run_test(name, func, **params)
        results.append((name, ok))

    log("----- TEST SUMMARY -----")
    for name, ok in results:
        log(f"[TEST] {name}: {'PASS' if ok else 'FAIL'}")

    log("=== TEST RUN COMPLETE ===")

if __name__ == "__main__":
    main()