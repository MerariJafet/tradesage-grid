from typing import Tuple, Union

import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower


def engineer_features(
    df_ohlcv: pd.DataFrame,
    imbalance: Union[pd.Series, float, int],
    funding_rate: Union[pd.Series, float, int],
) -> pd.DataFrame:
    df = df_ohlcv.copy()
    if not isinstance(imbalance, pd.Series):
        imbalance = pd.Series(imbalance, index=df.index)
    if not isinstance(funding_rate, pd.Series):
        funding_rate = pd.Series(funding_rate, index=df.index)
    imbalance = imbalance.reindex(df.index)
    funding_rate = funding_rate.reindex(df.index)
    df["atr"] = calculate_atr(df)
    df["rsi"] = calculate_rsi(df["close"])
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    df["funding_rate"] = funding_rate
    df["imbalance"] = imbalance
    df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"])
    df["bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["close"].rolling(20).mean()
    df["adx"] = df["atr"].rolling(14).std() * 100
    df["target"] = np.where(df["close"].shift(-60) > df["close"], 1, 0)
    return df.dropna()
