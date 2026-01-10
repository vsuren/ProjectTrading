import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    print(">>> ENGINEERED FEATURES FUNCTION CALLED <<<")

    """
    Adds all engineered features required for FeatureSet v1.2.
    Assumes df already contains:
        - OHLCV
        - ATR_14
        - EMA_9, EMA_20, EMA_50, EMA_200
        - SMA_20, SMA_200
        - BBANDS_UPPER_20, BBANDS_MIDDLE_20, BBANDS_LOWER_20
        - WILLIAMS_R_14
        - SUPERTREND_10
    """

    # --- 1. Volatility-normalized features ---
    df["Return_t_norm"] = (df["ClosePrice"] - df["ClosePrice"].shift(1)) / df["ATR_14"]
    df["CandleBody_norm"] = (df["ClosePrice"] - df["OpenPrice"]) / df["ATR_14"]
    df["Range_norm"] = (df["HighPrice"] - df["LowPrice"]) / df["ATR_14"]

    df["UpperWick_norm"] = (
        df["HighPrice"] - df[["OpenPrice", "ClosePrice"]].max(axis=1)
    ) / df["ATR_14"]

    df["LowerWick_norm"] = (
        df[["OpenPrice", "ClosePrice"]].min(axis=1) - df["LowPrice"]
    ) / df["ATR_14"]

    # --- 2. Trend strength features ---
    df["EMA20_minus_EMA200"] = df["EMA_20"] - df["EMA_200"]
    df["EMA9_minus_EMA20"] = df["EMA_9"] - df["EMA_20"]
    df["SMA20_minus_SMA200"] = df["SMA_20"] - df["SMA_200"]
    df["Close_minus_EMA20"] = df["ClosePrice"] - df["EMA_20"]
    df["Close_minus_EMA200"] = df["ClosePrice"] - df["EMA_200"]

    # --- 3. Regime features ---
    df["Volatility_regime"] = df["ATR_14"] / df["ATR_14"].rolling(100).mean()
    df["Trend_regime"] = (df["EMA_20"] > df["EMA_200"]).astype(int)

    df["HourOfDay"] = df["PriceTimestamp"].dt.hour
    df["MinuteOfHour"] = df["PriceTimestamp"].dt.minute

    df["SessionType"] = np.select(
        [
            (df["HourOfDay"] < 9) | (df["HourOfDay"] >= 16),
            (df["HourOfDay"] == 9),
            (df["HourOfDay"] >= 10) & (df["HourOfDay"] <= 14),
            (df["HourOfDay"] >= 15) & (df["HourOfDay"] < 16),
        ],
        [0, 1, 2, 3],
        default=2,
    )

    # --- 4. Compression / expansion ---
    df["BBWidth"] = df["BBANDS_UPPER_20"] - df["BBANDS_LOWER_20"]
    df["BBWidth_norm"] = df["BBWidth"] / df["ATR_14"]
    df["Range_vs_ATR"] = (df["HighPrice"] - df["LowPrice"]) / df["ATR_14"]

    df["RollingRangeCompression"] = (
        df["Range_norm"] / df["Range_norm"].rolling(50).mean()
    )

    # --- 5. Momentum / reversal ---
    df["Return_zscore"] = (
        df["ClosePrice"].pct_change().fillna(0).rolling(50).apply(
            lambda x: np.nan if len(x) < 50 else (x.iloc[-1] - x.mean()) / (x.std() + 1e-9),
            raw=False
        )
    )

    df["WilliamsR_smoothed"] = df["WILLIAMS_R_14"].rolling(5).mean()

    # --- 6. Volume microstructure ---
    df["Volume_rel"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Volume_spike"] = (df["Volume_rel"] > 2).astype(int)

    df["Volume_percentile"] = (
        df["Volume"]
        .rolling(100)
        .apply(
            lambda x: np.nan if len(x) < 100 else pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )
    )

    # --- 7. MACD (12, 26, 9) ---
    def compute_macd(group):
        group["EMA_12"] = group["ClosePrice"].ewm(span=12, adjust=False).mean()
        group["EMA_26"] = group["ClosePrice"].ewm(span=26, adjust=False).mean()

        group["MACD_line"] = group["EMA_12"] - group["EMA_26"]
        group["MACD_signal"] = group["MACD_line"].ewm(span=9, adjust=False).mean()
        group["MACD_histogram"] = group["MACD_line"] - group["MACD_signal"]

        return group

    df = df.groupby("Symbol", group_keys=False).apply(compute_macd)

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df