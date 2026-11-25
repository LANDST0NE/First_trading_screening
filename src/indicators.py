from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


@dataclass
class PivotPoint:
    price: float
    formation_idx: int
    confirmation_idx: int
    formation_time: Any
    confirmation_time: Any
    is_high: bool


@dataclass
class TFIndicators:
    tf: str
    ema20: float
    ema50: float
    atr14: float
    rsi14: float
    poc80: float
    val80: float
    vah80: float
    swing_low_20: float
    swing_high_20: float
    last_pivot_high: Optional[PivotPoint]
    last_pivot_low: Optional[PivotPoint]
    market_structure: str
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

    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)
    return pd.Series(rsi, index=df.index)


def calculate_vpvr(
    df: pd.DataFrame, lookback: int = 80, num_bins: int = 60, va_pct: float = 0.70
) -> Tuple[float, float, float]:
    """
    Расчёт eVPVR (Estimated Volume Profile) за последние lookback свечей.
    Оценочный объёмный профиль: объём OHLCV-свечей распределяется по корзинам диапазона.
    - POC (Point of Control) — корзина с максимальным объёмом.
    - VAL (Value Area Low) — нижняя граница 70% проторгованного объёма.
    - VAH (Value Area High) — верхняя граница 70% проторгованного объёма.
    """
    subset = df.iloc[-lookback:].copy()
    if subset.empty:
        return 0.0, 0.0, 0.0

    min_low = float(subset["low"].min())
    max_high = float(subset["high"].max())

    if max_high <= min_low:
        return min_low, min_low, max_high

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
            idx = int((c_low - min_low) / bin_width)
            idx = min(max(0, idx), num_bins - 1)
            bin_volumes[idx] += c_vol
            continue

        c_range = c_high - c_low

        for b in range(num_bins):
            b_left = bin_edges[b]
            b_right = bin_edges[b + 1]

            overlap_left = max(c_low, b_left)
            overlap_right = min(c_high, b_right)

            if overlap_right > overlap_left:
                overlap_fraction = (overlap_right - overlap_left) / c_range
                bin_volumes[b] += c_vol * overlap_fraction

    total_vol = float(np.sum(bin_volumes))
    if total_vol <= 0:
        mid = 0.5 * (min_low + max_high)
        return mid, min_low, max_high

    poc_idx = int(np.argmax(bin_volumes))
    poc_level = min_low + (poc_idx + 0.5) * bin_width

    # Value Area 70%
    target_vol = total_vol * va_pct
    accumulated_vol = float(bin_volumes[poc_idx])
    upper_idx = poc_idx
    lower_idx = poc_idx

    while accumulated_vol < target_vol and (upper_idx < num_bins - 1 or lower_idx > 0):
        next_up = float(bin_volumes[upper_idx + 1]) if upper_idx + 1 < num_bins else -1.0
        next_down = float(bin_volumes[lower_idx - 1]) if lower_idx - 1 >= 0 else -1.0

        if next_up >= next_down and next_up >= 0:
            upper_idx += 1
            accumulated_vol += next_up
        elif next_down > next_up and next_down >= 0:
            lower_idx -= 1
            accumulated_vol += next_down
        else:
            break

    val_level = float(min_low + lower_idx * bin_width)
    vah_level = float(min_low + (upper_idx + 1) * bin_width)

    return poc_level, val_level, vah_level


def calculate_vpvr_poc(df: pd.DataFrame, lookback: int = 80, num_bins: int = 60) -> float:
    """Для обратной совместимости: возвращает только POC."""
    poc, _, _ = calculate_vpvr(df, lookback, num_bins)
    return poc


def find_confirmed_pivots(
    df: pd.DataFrame, left: int = 3, right: int = 2
) -> Tuple[List[PivotPoint], List[PivotPoint]]:
    """
    Поиск подтверждённых точек перелома структуры (Pivot High / Pivot Low).
    Параметры:
    - left = 3 (число свечей слева)
    - right = 2 (число свечей справа)
    
    СТРОГОЕ ПРАВИЛО (БЕЗ LOOK-AHEAD BIAS):
    Пивот на свече i подтверждается ТОЛЬКО после закрытия свечи i + right.
    Если общее число закрытых свечей N, то последняя свеча, которая может быть
    подтверждённым пивотом — это candle N - 1 - right.
    """
    n = len(df)
    pivot_highs: List[PivotPoint] = []
    pivot_lows: List[PivotPoint] = []

    if n < left + right + 1:
        return pivot_highs, pivot_lows

    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values

    # Проверяем все свечи, для которых закрыты right свечей справа
    # Последняя допустимая свеча: n - 1 - right
    for i in range(left, n - right):
        # 1. Проверка Pivot High
        is_pivot_high = True
        h_i = highs[i]
        for l_offset in range(1, left + 1):
            if highs[i - l_offset] >= h_i:
                is_pivot_high = False
                break
        if is_pivot_high:
            for r_offset in range(1, right + 1):
                if highs[i + r_offset] >= h_i:
                    is_pivot_high = False
                    break

        if is_pivot_high:
            conf_idx = i + right
            pivot_highs.append(PivotPoint(
                price=float(h_i),
                formation_idx=i,
                confirmation_idx=conf_idx,
                formation_time=times[i],
                confirmation_time=times[conf_idx],
                is_high=True
            ))

        # 2. Проверка Pivot Low
        is_pivot_low = True
        l_i = lows[i]
        for l_offset in range(1, left + 1):
            if lows[i - l_offset] <= l_i:
                is_pivot_low = False
                break
        if is_pivot_low:
            for r_offset in range(1, right + 1):
                if lows[i + r_offset] <= l_i:
                    is_pivot_low = False
                    break

        if is_pivot_low:
            conf_idx = i + right
            pivot_lows.append(PivotPoint(
                price=float(l_i),
                formation_idx=i,
                confirmation_idx=conf_idx,
                formation_time=times[i],
                confirmation_time=times[conf_idx],
                is_high=False
            ))

    return pivot_highs, pivot_lows


def evaluate_market_structure(
    df: pd.DataFrame,
    pivot_highs: List[PivotPoint],
    pivot_lows: List[PivotPoint],
    atr: float
) -> str:
    """
    Оценивает характер рыночной структуры по последним подтверждённым пивотам.
    """
    if not pivot_highs or not pivot_lows:
        return "Not Enough Pivots"

    last_ph = pivot_highs[-1]
    last_pl = pivot_lows[-1]
    last_close = float(df["close"].iloc[-1])

    # Проверка пробоя структуры на последней закрытой свече
    if last_close > last_ph.price + 0.1 * atr:
        return f"Bullish Breakout (выше {last_ph.price:.2f})"
    elif last_close < last_pl.price - 0.1 * atr:
        return f"Bearish Breakdown (ниже {last_pl.price:.2f})"

    # Сравнение последних двух пивотов
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        prev_ph = pivot_highs[-2]
        prev_pl = pivot_lows[-2]

        is_higher_high = last_ph.price > prev_ph.price
        is_higher_low = last_pl.price > prev_pl.price
        is_lower_high = last_ph.price < prev_ph.price
        is_lower_low = last_pl.price < prev_pl.price

        if is_higher_high and is_higher_low:
            return "Uptrend (Higher Highs / Higher Lows)"
        elif is_lower_high and is_lower_low:
            return "Downtrend (Lower Highs / Lower Lows)"
        elif is_lower_high and is_higher_low:
            return "Consolidation (Сжатие диапазона)"
        elif is_higher_high and is_lower_low:
            return "Expanding (Расширение диапазона)"

    # Если пивотов мало
    if last_close > last_ph.price:
        return "Breakout Testing"
    elif last_close < last_pl.price:
        return "Breakdown Testing"

    return "Range / Neutral"


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
    atr14 = float(last_row["atr14"])

    poc80, val80, vah80 = calculate_vpvr(df_calc, lookback=80, num_bins=60, va_pct=0.70)

    # Локальные экстремумы за последние 20 свечей (скользящие)
    subset20 = df_calc.iloc[-20:]
    swing_low_20 = float(subset20["low"].min())
    swing_high_20 = float(subset20["high"].max())

    # Поиск подтверждённых Pivot Points (L=3, R=2)
    pivot_highs, pivot_lows = find_confirmed_pivots(df_calc, left=3, right=2)
    last_ph = pivot_highs[-1] if pivot_highs else None
    last_pl = pivot_lows[-1] if pivot_lows else None

    # Оценка структуры рынка
    market_structure = evaluate_market_structure(df_calc, pivot_highs, pivot_lows, atr14)

    return TFIndicators(
        tf=tf_name,
        ema20=float(last_row["ema20"]),
        ema50=float(last_row["ema50"]),
        atr14=atr14,
        rsi14=float(last_row["rsi14"]),
        poc80=poc80,
        val80=val80,
        vah80=vah80,
        swing_low_20=swing_low_20,
        swing_high_20=swing_high_20,
        last_pivot_high=last_ph,
        last_pivot_low=last_pl,
        market_structure=market_structure,
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
