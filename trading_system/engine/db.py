import pyodbc
import os
from trading_system.engine.logger import log

# ============================================================
# LOAD AZURE SQL CONFIG FROM db_config.txt
# ============================================================

def load_db_config():
    """
    Reads Azure SQL connection settings from db_config.txt.
    Expected format:
        SERVER=xxxx.database.windows.net
        DATABASE=TradingDB
        USERNAME=tradingadmin
        PASSWORD=yourpassword
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "db_config.txt"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"db_config.txt not found at {config_path}")

    config = {}
    with open(config_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key.strip()] = value.strip()

    required = ["SERVER", "DATABASE", "USERNAME", "PASSWORD"]
    for r in required:
        if r not in config:
            raise KeyError(f"Missing {r} in db_config.txt")

    return config


# ============================================================
# DATABASE CONNECTION (AZURE SQL)
# ============================================================

def get_connection():
    """
    Creates a secure Azure SQL connection using ODBC Driver 18.
    """
    cfg = load_db_config()

    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={cfg['SERVER']};"
        f"DATABASE={cfg['DATABASE']};"
        f"UID={cfg['USERNAME']};"
        f"PWD={cfg['PASSWORD']};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    return pyodbc.connect(conn_str)


# ============================================================
# HIGH‑PERFORMANCE BATCH INSERT FOR INDICATORS
# ============================================================

def insert_indicators(insert_sql, data, batch_size=50000):
    """
    Inserts indicator rows into Azure SQL in high‑performance batches.
    Uses fast_executemany for 10x–20x speed improvement.
    """

    total_rows = len(data)
    log(f"Preparing to insert {total_rows} rows into tblIndicators.")

    if total_rows == 0:
        log("No rows to insert.")
        return

    conn = get_connection()
    cursor = conn.cursor()

    # ⚡ MASSIVE SPEED BOOST — enables true bulk insert mode
    cursor.fast_executemany = True

    try:
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = data[start:end]

            log(f"Inserting batch {start//batch_size + 1} ({len(batch)} rows)...")

            cursor.executemany(insert_sql, batch)
            conn.commit()

        log("All batches inserted successfully.")

    except Exception as e:
        log(f"ERROR during batch insert: {e}")
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()
        log("Database connection closed.")


# ============================================================
# SIMPLE QUERY HELPERS
# ============================================================

def fetch_all(sql, params=None):
    """
    Executes a SELECT query and returns all rows.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall()
        return rows

    finally:
        cursor.close()
        conn.close()


def execute_non_query(sql, params=None):
    """
    Executes INSERT/UPDATE/DELETE without returning rows.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql, params or [])
        conn.commit()

    finally:
        cursor.close()
        conn.close()