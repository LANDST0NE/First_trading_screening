from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class TFIndicators:
    tf: str
    ema20: float
    ema50: float
    atr14: float
    rsi14: float
    poc80: float
    swing_low_20: float
    swing_high_20: float
    close: float
    candles_count: int


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Экспоненциальная скользящая средняя (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_wilder_rma(series: pd.Series, period: int = 14) -> pd.Series:
    """Сглаживание по методу Уайлдера (RMA / Wilder's Smoothing)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR по методу сглаживания Уайлдера (RMA от True Range)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Первое значение можно заполнить high - low
    tr.iloc[0] = tr1.iloc[0]
    atr = calculate_wilder_rma(tr, period=period)
    return atr


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI(14) по методу Уайлдера."""
    close = df["close"]
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = calculate_wilder_rma(gain, period=period)
    avg_loss = calculate_wilder_rma(loss, period=period)

    # Предотвращение деления на ноль
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Если avg_loss == 0 и avg_gain == 0 -> 50, если avg_loss == 0 и avg_gain > 0 -> 100
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)
    return pd.Series(rsi, index=df.index)


def calculate_vpvr_poc(df: pd.DataFrame, lookback: int = 80, num_bins: int = 60) -> float:
    """
    Расчёт POC (Point of Control) — уровень с максимальным проторгованным
    объёмом в профиле VPVR за последние lookback свечей, разбив диапазон
    [min_low, max_high] на num_bins корзин.
    """
    subset = df.iloc[-lookback:].copy()
    if subset.empty:
        return 0.0

    min_low = float(subset["low"].min())
    max_high = float(subset["high"].max())

    if max_high <= min_low:
        return min_low

    bin_width = (max_high - min_low) / num_bins
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    bin_edges = [min_low + i * bin_width for i in range(num_bins + 1)]

    for _, row in subset.iterrows():
        c_low = float(row["low"])
        c_high = float(row["high"])
        c_vol = float(row["volume"])

        if c_vol <= 0:
            continue

        if c_high <= c_low:
            # Свеча с нулевым диапазоном — определяем корзину
            idx = int((c_low - min_low) / bin_width)
            idx = min(max(0, idx), num_bins - 1)
            bin_volumes[idx] += c_vol
            continue

        c_range = c_high - c_low

        # Распределяем объём свечи по перекрывающимся корзинам
        for b in range(num_bins):
            b_left = bin_edges[b]
            b_right = bin_edges[b + 1]

            # Пересечение диапазона свечи [c_low, c_high] и корзины [b_left, b_right]
            overlap_left = max(c_low, b_left)
            overlap_right = min(c_high, b_right)

            if overlap_right > overlap_left:
                overlap_fraction = (overlap_right - overlap_left) / c_range
                bin_volumes[b] += c_vol * overlap_fraction

    # Находим корзину с максимальным объёмом
    max_bin_idx = int(np.argmax(bin_volumes))
    poc_level = min_low + (max_bin_idx + 0.5) * bin_width
    return float(poc_level)


def calculate_tf_indicators(df: pd.DataFrame, tf_name: str) -> Optional[TFIndicators]:
    """Рассчитывает полный пакет индикаторов для одного таймфрейма."""
    if df.empty or len(df) < 50:
        return None

    df_calc = df.copy()
    df_calc["ema20"] = calculate_ema(df_calc["close"], span=20)
    df_calc["ema50"] = calculate_ema(df_calc["close"], span=50)
    df_calc["atr14"] = calculate_atr(df_calc, period=14)
    df_calc["rsi14"] = calculate_rsi(df_calc, period=14)

    last_row = df_calc.iloc[-1]
    last_close = float(last_row["close"])

    poc80 = calculate_vpvr_poc(df_calc, lookback=80, num_bins=60)

    # Локальные экстремумы за последние 20 свечей
    subset20 = df_calc.iloc[-20:]
    swing_low_20 = float(subset20["low"].min())
    swing_high_20 = float(subset20["high"].max())

    return TFIndicators(
        tf=tf_name,
        ema20=float(last_row["ema20"]),
        ema50=float(last_row["ema50"]),
        atr14=float(last_row["atr14"]),
        rsi14=float(last_row["rsi14"]),
        poc80=poc80,
        swing_low_20=swing_low_20,
        swing_high_20=swing_high_20,
        close=last_close,
        candles_count=len(df_calc)
    )


def calculate_all_indicators(candles_dict: Dict[str, pd.DataFrame]) -> Dict[str, TFIndicators]:
    """Рассчитывает индикаторы для 1H, 4H, 1D."""
    indicators = {}
    for tf in ["1H", "4H", "1D"]:
        if tf in candles_dict and not candles_dict[tf].empty:
            ind = calculate_tf_indicators(candles_dict[tf], tf)
            if ind:
                indicators[tf] = ind
    return indicators
