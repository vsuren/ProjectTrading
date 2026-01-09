import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import zoneinfo

from trading_system.engine.logger import log
from trading_system.engine.db import (
    insert_indicators,
    fetch_all,
)

from trading_system.indicators.core_indicators import (
    sma,
    ema,
    rsi,
    atr_wilder,
    adx,
    macd,
    obv,
    roc,
    vwap,
    cci,
    bbands,
    stochastic,
    boll_width,
    boll_pctb,
    williams_r,
    super_trend,
)

TORONTO_TZ = zoneinfo.ZoneInfo("America/Toronto")

# ============================================================
# INDICATOR REGISTRY
# ============================================================

INDICATOR_MAP = {
    "SMA": lambda df, p: sma(df["Close"], p),
    "EMA": lambda df, p: ema(df["Close"], p),
    "RSI": lambda df, p: rsi(df["Close"], p),
    "ATR": lambda df, p: atr_wilder(df, p),
    "ADX": lambda df, p: adx(df, p),
    "MACD": lambda df, p: macd(df["Close"]),
    "OBV": lambda df, p: obv(df),
    "ROC": lambda df, p: roc(df["Close"], p),
    "VWAP": lambda df, p: vwap(df),
    "CCI": lambda df, p: cci(df, p),
    "BBANDS_UPPER": lambda df, p: bbands(df, p)[0],
    "BBANDS_MIDDLE": lambda df, p: bbands(df, p)[1],
    "BBANDS_LOWER": lambda df, p: bbands(df, p)[2],
    "STOCH_K": lambda df, p: stochastic(df, p)[0],
    "STOCH_D": lambda df, p: stochastic(df, p)[1],
    "BOLL_WIDTH": lambda df, p: boll_width(df, p),
    "BOLL_PCTB": lambda df, p: boll_pctb(df, p),
    "WILLIAMS_R": lambda df, p: williams_r(df, p),
    "SUPER_TREND": lambda df, p: super_trend(df, p),
    "SUPERTREND": lambda df, p: super_trend(df, p),
}

# ============================================================
# DATABASE LOADERS
# ============================================================

def load_indicators():
    sql = """
        SELECT [IndicatorId], [IndicatorName], [DefaultParams]
        FROM [tblIndicatorDefinitions]
        WHERE [IsActive] = 1
    """
    rows = fetch_all(sql)
    return pd.DataFrame.from_records(
        rows,
        columns=["IndicatorId", "IndicatorName", "DefaultParams"]
    )


def load_prices():
    sql = """
        SELECT 
            [Symbol],
            [PriceTimestamp],
            [OpenPrice],
            [HighPrice],
            [LowPrice],
            [ClosePrice],
            [Volume]
        FROM [tblRawPrices]
        ORDER BY [Symbol], [PriceTimestamp]
    """
    rows = fetch_all(sql)
    return pd.DataFrame.from_records(
        rows,
        columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    )

# ============================================================
# DISPATCHER
# ============================================================

def compute_indicator(name, df_symbol, param, indicator_id):
    df_symbol = df_symbol.sort_values("Date").reset_index(drop=True)
    symbol = df_symbol["Symbol"].iloc[0]
    name_upper = name.upper()

    if name_upper in {
        "SMA", "EMA", "RSI", "ATR", "ADX", "CCI",
        "BBANDS_UPPER", "BBANDS_MIDDLE", "BBANDS_LOWER",
        "STOCH_K", "STOCH_D", "BOLL_WIDTH", "BOLL_PCTB",
        "WILLIAMS_R", "SUPER_TREND", "SUPERTREND"
    }:
        param = int(param)

    func = INDICATOR_MAP.get(name_upper)
    if func is None:
        log(f"Unknown indicator {name}")
        return pd.DataFrame()

    if name_upper in ("SUPER_TREND", "SUPERTREND"):
        log(f"SUPERTREND: computing for symbol={symbol}, period={param}")

    values = func(df_symbol, param)

    return pd.DataFrame({
        "Symbol": symbol,
        "Date": df_symbol["Date"],
        "IndicatorId": indicator_id,
        "IndicatorName": name,
        "Param": param,
        "Value": values
    }).dropna(subset=["Value"])

# ============================================================
# RUNNER
# ============================================================

def run_for_one_indicator(ind_row, df_prices):
    name = ind_row["IndicatorName"]
    param = ind_row["DefaultParams"]
    indicator_id = ind_row["IndicatorId"]

    start = dt.datetime.now(TORONTO_TZ)
    log(f"Starting indicator {name} (IndicatorId={indicator_id}, param={param})")

    results = []
    for symbol, df_sym in df_prices.groupby("Symbol"):
        df_res = compute_indicator(name, df_sym, param, indicator_id)
        if not df_res.empty:
            results.append(df_res)

    df_final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    elapsed = (dt.datetime.now(TORONTO_TZ) - start).total_seconds()
    log(f"Completed indicator {name} with {len(df_final)} rows in {elapsed:.2f} seconds.")

    return df_final


def master_run(dry_run=False, max_workers=8):
    start = dt.datetime.now(TORONTO_TZ)
    log("=== MASTER RUN STARTED ===")
    log(f"DRY_RUN mode: {dry_run}")

    df_ind = load_indicators()
    log(f"Loaded {len(df_ind)} active indicators.")

    df_prices = load_prices()
    log(f"Loaded {len(df_prices)} price rows.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(run_for_one_indicator, row.to_dict(), df_prices)
            for _, row in df_ind.iterrows()
        ]

        for fut in as_completed(futures):
            try:
                df_res = fut.result()
                if not df_res.empty:
                    results.append(df_res)
            except Exception as e:
                log(f"ERROR: {e}")

    df_all = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    log(f"Total computed indicator rows: {len(df_all)}")

    # ============================================================
    # ADD FieldName (required for merge_feature.py)
    # ============================================================

    df_all["FieldName"] = df_all["IndicatorName"] + "_" + df_all["Param"].astype(str)

    # ============================================================
    # INSERT INTO SQL
    # ============================================================

    insert_start = dt.datetime.now(TORONTO_TZ)
    log(f"Preparing to insert {len(df_all)} rows into tblIndicators. DRY_RUN={dry_run}")

    if not dry_run:

        df_all["Timeframe"] = "1D"
        df_all["addDateTime"] = pd.Timestamp.now(tz=TORONTO_TZ)

        df_all = df_all.rename(columns={
            "Date": "PriceTimestamp",
            "Param": "IndicatorParams",
            "Value": "IndicatorValue"
        })

        insert_sql = """
            INSERT INTO [tblIndicators]
            ([Symbol], [PriceTimestamp], [Timeframe],
             [IndicatorId], [IndicatorName], [IndicatorParams],
             [IndicatorValue], [FieldName], [addDateTime])
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        data = list(df_all[[
            "Symbol",
            "PriceTimestamp",
            "Timeframe",
            "IndicatorId",
            "IndicatorName",
            "IndicatorParams",
            "IndicatorValue",
            "FieldName",
            "addDateTime"
        ]].itertuples(index=False, name=None))

        insert_indicators(insert_sql, data)

    else:
        log("DRY_RUN active — no rows inserted.")

    insert_elapsed = (dt.datetime.now(TORONTO_TZ) - insert_start).total_seconds()
    log(f"Insert step completed in {insert_elapsed:.2f} seconds.")

    end = dt.datetime.now(TORONTO_TZ)
    total_elapsed = (end - start).total_seconds()
    log(f"=== MASTER RUN COMPLETED in {total_elapsed:.2f} seconds ===")