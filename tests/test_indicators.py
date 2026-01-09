"""
test_indicators.py
Production‑grade indicator validation suite
"""

# -------------------------------------------------------------------
# UNIVERSAL IMPORT FIX (must be first)
# -------------------------------------------------------------------
import sys
sys.path.insert(0, r"E:\ProjectTrading\trading_system")

# -------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------
import pandas as pd
import numpy as np
import traceback
import pyodbc
from engine.logger import log, set_log_prefix
from config.config_loader import load_db_config
from indicators.core_indicators import (
    sma,
    ema,
    rsi,
    macd,
    atr_wilder,
    super_trend,
    adx,
    stochastic,
    bbands,
    cci,
    williams_r,
    vwap
)

# -------------------------------------------------------------------
# WRAPPER FUNCTIONS FOR TESTING
# -------------------------------------------------------------------
def compute_sma(df, period):
    return sma(df["Close"], period)

def compute_ema(df, period):
    return ema(df["Close"], period)

def compute_rsi(df, period):
    return rsi(df["Close"], period)

def compute_macd(df, fast=12, slow=26, signal=9):
    # Simple MACD, ignoring signal for now
    return macd(df["Close"])

def compute_atr(df, period):
    return atr_wilder(df, period)

def compute_supertrend(df, period, multiplier=3):
    return super_trend(df, period, multiplier)

def compute_adx(df, period):
    return adx(df, period)

def compute_stochastic(df, k_period=14, d_period=3):
    return stochastic(df, k_period)[0]  # Assuming it returns (k, d)

def compute_bollinger_bands(df, period=20, stddev=2):
    return bbands(df, period)[1]  # Middle band or something

def compute_cci(df, period):
    return cci(df, period)

def compute_williams_r(df, period):
    return williams_r(df, period)

def compute_vwap(df):
    return vwap(df)

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
# No logger object, use log() function directly

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
# FETCH SAMPLE DATA
# -------------------------------------------------------------------
def load_sample_prices():
    query = """
        SELECT TOP 2000
            [Symbol], [PriceTimestamp], [OpenPrice], [HighPrice], [LowPrice], [ClosePrice], [Volume]
        FROM [tblRawPrices]
        WHERE [Symbol] = 'AAPL'
        ORDER BY [PriceTimestamp] ASC
    """
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    df = df.rename(columns={
        'PriceTimestamp': 'Date',
        'OpenPrice': 'Open',
        'HighPrice': 'High',
        'LowPrice': 'Low',
        'ClosePrice': 'Close'
    })
    return df

# -------------------------------------------------------------------
# GENERIC TEST RUNNER
# -------------------------------------------------------------------
def run_test(name, func, df, **params):
    log(f"[TEST] --- Testing {name} ---")

    try:
        out = func(df.copy(), **params)

        if len(out) != len(df):
            raise ValueError(f"{name}: Output length mismatch")

        if out.iloc[50:].isna().sum().sum() > 0:
            raise ValueError(f"{name}: NaNs detected after warm-up")

        log(f"[TEST] {name}: PASS")
        return True

    except Exception as e:
        log(f"[TEST] {name}: FAIL")
        log(str(e))
        log(traceback.format_exc())
        return False

# -------------------------------------------------------------------
# MAIN TEST SUITE
# -------------------------------------------------------------------
def main():
    set_log_prefix("test_indicators")
    log("=== TEST RUN: Indicator Validation Suite ===")

    df = load_sample_prices()

    tests = [
        ("SMA", compute_sma, {"period": 20}),
        ("EMA", compute_ema, {"period": 20}),
        ("RSI", compute_rsi, {"period": 14}),
        ("MACD", compute_macd, {"fast": 12, "slow": 26, "signal": 9}),
        ("ATR", compute_atr, {"period": 14}),
        ("SuperTrend", compute_supertrend, {"period": 10, "multiplier": 3}),
        ("ADX", compute_adx, {"period": 14}),
        ("Stochastic", compute_stochastic, {"k_period": 14, "d_period": 3}),
        ("BollingerBands", compute_bollinger_bands, {"period": 20, "stddev": 2}),
        ("CCI", compute_cci, {"period": 20}),
        ("WilliamsR", compute_williams_r, {"period": 14}),
        ("VWAP", compute_vwap, {}),
    ]

    results = []
    for name, func, params in tests:
        ok = run_test(name, func, df, **params)
        results.append((name, ok))

    log("----- TEST SUMMARY -----")
    for name, ok in results:
        log(f"[TEST] {name}: {'PASS' if ok else 'FAIL'}")

    log("=== TEST RUN COMPLETE ===")

if __name__ == "__main__":
    main()