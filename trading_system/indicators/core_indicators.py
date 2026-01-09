import pandas as pd
import numpy as np

# ============================================================
# BASIC INDICATORS
# ============================================================

def sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr_wilder(df, period):
    df = df.sort_values("Date").reset_index(drop=True)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr

def adx(df, period):
    df = df.sort_values("Date").reset_index(drop=True)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    tr_smooth = tr.rolling(window=period, min_periods=period).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum()

    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.rolling(window=period, min_periods=period).mean()

    return adx_val

def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def obv(df):
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()

def roc(series, period=12):
    return series.pct_change(periods=period) * 100

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (tp * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)

def cci(df, period):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = (tp - sma_tp).abs().rolling(window=period, min_periods=period).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

def bbands(df, period):
    close = df["Close"]
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower

def stochastic(df, period):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=3, min_periods=3).mean()

    return k, d

def boll_width(df, period):
    upper, mid, lower = bbands(df, period)
    return (upper - lower) / mid.replace(0, np.nan)

def boll_pctb(df, period):
    upper, mid, lower = bbands(df, period)
    close = df["Close"]
    return (close - lower) / (upper - lower).replace(0, np.nan)

def williams_r(df, period):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return wr

# ============================================================
# SUPER TREND (no logging here – pure math)
# ============================================================

def super_trend(df: pd.DataFrame, period: int, multiplier: float = 3.0) -> pd.Series:
    df = df.sort_values("Date").reset_index(drop=True)

    high = df["High"].reset_index(drop=True)
    low = df["Low"].reset_index(drop=True)
    close = df["Close"].reset_index(drop=True)

    atr = atr_wilder(df, period).reset_index(drop=True)

    hl2 = (high + low) / 2

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        final_upper.iloc[i] = (
            basic_upper.iloc[i]
            if (basic_upper.iloc[i] < final_upper.iloc[i - 1]) or (close.iloc[i - 1] > final_upper.iloc[i - 1])
            else final_upper.iloc[i - 1]
        )

        final_lower.iloc[i] = (
            basic_lower.iloc[i]
            if (basic_lower.iloc[i] > final_lower.iloc[i - 1]) or (close.iloc[i - 1] < final_lower.iloc[i - 1])
            else final_lower.iloc[i - 1]
        )

    trend = np.ones(len(df), dtype=int)
    supertrend = pd.Series(index=df.index, dtype=float)

    supertrend.iloc[0] = final_lower.iloc[0]

    for i in range(1, len(df)):
        if close.iloc[i] > final_upper.iloc[i]:
            trend[i] = 1
        elif close.iloc[i] < final_lower.iloc[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        supertrend.iloc[i] = final_lower.iloc[i] if trend[i] == 1 else final_upper.iloc[i]

    return supertrend