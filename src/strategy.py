from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from src.indicators import TFIndicators, PivotPoint
from src.client import InstrumentInfo
from src.utils import round_to_step
from src.config import (
    COMMISSION,
    DAILY_RISK_LIMIT,
    RISK_STANDARD,
    RISK_STRONG,
    MIN_RR_THRESHOLD,
    STRONG_RR_THRESHOLD
)


@dataclass
class SetupResult:
    ticker: str
    direction: str             # "LONG" или "SHORT"
    status: str                # "ACTIVE (Strong)", "ACTIVE (Standard)", "SKIP (...)"
    trade_type: str            # "Trend Following", "Confirmed Reversal", "Reversal Candidate", "Counter Trend", "MIXED / NO TRADE"
    current_price: float
    senior_trend: str
    market_structure_1h: str   # Описание часовой структуры (Pivots)
    last_pivot_high: Optional[float]
    last_pivot_low: Optional[float]
    confirmation_trigger: str  # Что должно произойти для подтверждения
    invalidation_trigger: str  # Условие отмены сценария
    anchor_stop: float
    target_1: float
    target_2: float
    entry_1: float
    entry_2: float
    entry_2_source: str        # Источник второго входа: "Cluster (VAL + Pivot Low)", "VAL 70%", etc.
    entry_avg: float
    stop: float
    take_1: float
    take_2: float
    risk_per_share: float
    reward_t1_per_share: float
    reward_t2_per_share: float
    reward_avg_per_share: float
    reward_per_share: float
    rr: float
    risk_rub: float
    ideal_entry: Optional[float]
    total_shares: int
    total_lots: int
    lots_entry_1: int
    lots_entry_2: int
    lots_take_1: int
    lots_take_2: int
    actual_loss_rub: float
    profit_take_1_rub: float
    profit_take_2_rub: float
    total_profit_rub: float
    step: float
    lot_size: int
    skip_reason: Optional[str] = None
    warning: Optional[str] = None


def get_senior_trend(indicators: Dict[str, TFIndicators]) -> str:
    """Определяет господствующий тренд на основе 4H_EMA50 и 1D_EMA20."""
    ind_1h = indicators.get("1H")
    ind_4h = indicators.get("4H")
    ind_1d = indicators.get("1D")
    if not ind_1h or not ind_4h or not ind_1d:
        return "UNKNOWN"

    price = ind_1h.close
    if price < ind_4h.ema50 and price < ind_1d.ema20:
        return "BEARISH"
    elif price > ind_4h.ema50 and price > ind_1d.ema20:
        return "BULLISH"
    return "MIXED"


def classify_trade_type(
    direction: str,
    indicators: Dict[str, TFIndicators],
    senior_trend: str
) -> Tuple[str, str, str]:
    """
    Классифицирует сделку по строгой 4-уровневой иерархии приоритета:
    1. Trend Following (по старшему тренду)
    2. Confirmed Reversal (подтверждённый слом структуры закрытием за пивот)
    3. Reversal Candidate (локальное восстановление выше/ниже EMA50 и POC)
    4. Counter Trend (контртрендовый отскок)
    или MIXED / NO TRADE при противоречиях.
    
    Возвращает:
    (trade_type, confirmation_trigger, invalidation_trigger_desc)
    """
    ind_1h = indicators.get("1H")
    if not ind_1h:
        return "Counter Trend", "Недостаточно данных 1H", "Пробой стоп-лосса"

    price = ind_1h.close
    atr_1h = ind_1h.atr14
    last_ph = ind_1h.last_pivot_high
    last_pl = ind_1h.last_pivot_low

    ph_price_str = f"{last_ph.price:.2f} ₽" if last_ph else "N/A"
    pl_price_str = f"{last_pl.price:.2f} ₽" if last_pl else "N/A"

    if direction == "LONG":
        # 1. Приоритет: Trend Following
        if senior_trend == "BULLISH":
            if price >= ind_1h.ema50 or ind_1h.rsi14 >= 40.0:
                trade_type = "Trend Following"
                conf_trigger = f"Удержание выше 1H EMA20/EMA50 ({ind_1h.ema20:.2f} ₽) и закрытие текущего часа"
            else:
                trade_type = "Counter Trend"
                conf_trigger = f"Глубокий откат; возврат и закрепление выше 1H EMA50 ({ind_1h.ema50:.2f} ₽)"

        # 2. Приоритет: Confirmed Reversal (при медвежьем или смешанном старшем тренде)
        elif last_ph and (price > last_ph.price + 0.10 * atr_1h):
            trade_type = "Confirmed Reversal"
            conf_trigger = f"Подтверждено: закрытие 1H выше последнего Pivot High ({ph_price_str}) с удержанием"

        # 3. Приоритет: Reversal Candidate
        elif (price > ind_1h.ema50) and (price >= ind_1h.poc80) and (ind_1h.rsi14 >= 45.0):
            trade_type = "Reversal Candidate"
            conf_trigger = f"Пробой и закрытие часа выше подтверждённого 1H Pivot High ({ph_price_str})"

        # 4. Приоритет: MIXED / NO TRADE
        elif senior_trend == "MIXED" and abs(price - ind_1h.ema50) < 0.4 * atr_1h and (45.0 <= ind_1h.rsi14 <= 55.0):
            trade_type = "MIXED / NO TRADE"
            conf_trigger = "Ожидание выхода из распила и формирования направленного импульса"

        # 5. Приоритет: Counter Trend
        else:
            trade_type = "Counter Trend"
            conf_trigger = f"Спекулятивный подбор; импульсный возврат выше 1H EMA50 ({ind_1h.ema50:.2f} ₽) и POC ({ind_1h.poc80:.2f} ₽)"

    else:  # SHORT
        # 1. Приоритет: Trend Following
        if senior_trend == "BEARISH":
            if price <= ind_1h.ema50 or ind_1h.rsi14 <= 60.0:
                trade_type = "Trend Following"
                conf_trigger = f"Удержание под 1H EMA20/EMA50 ({ind_1h.ema20:.2f} ₽) и продолжение нисходящей структуры"
            else:
                trade_type = "Counter Trend"
                conf_trigger = f"Глубокий откат вверх; уход под 1H EMA50 ({ind_1h.ema50:.2f} ₽)"

        # 2. Приоритет: Confirmed Reversal (при бычьем старшем тренде)
        elif last_pl and (price < last_pl.price - 0.10 * atr_1h):
            trade_type = "Confirmed Reversal"
            conf_trigger = f"Подтверждено: закрытие 1H ниже последнего Pivot Low ({pl_price_str}) с удержанием"

        # 3. Приоритет: Reversal Candidate
        elif (price < ind_1h.ema50) and (price <= ind_1h.poc80) and (ind_1h.rsi14 <= 55.0):
            trade_type = "Reversal Candidate"
            conf_trigger = f"Пробой и закрытие часа ниже подтверждённого 1H Pivot Low ({pl_price_str})"

        # 4. Приоритет: MIXED / NO TRADE
        elif senior_trend == "MIXED" and abs(price - ind_1h.ema50) < 0.4 * atr_1h and (45.0 <= ind_1h.rsi14 <= 55.0):
            trade_type = "MIXED / NO TRADE"
            conf_trigger = "Ожидание выхода из распила и формирования направленного импульса"

        # 5. Приоритет: Counter Trend
        else:
            trade_type = "Counter Trend"
            conf_trigger = f"Спекулятивный шорт на отскоке; импульсный уход под 1H EMA50 ({ind_1h.ema50:.2f} ₽) и POC ({ind_1h.poc80:.2f} ₽)"

    inval_trigger = "Закрытие 1H свечи за уровнем Stop-Loss"
    return trade_type, conf_trigger, inval_trigger


def calculate_structural_entry_2(
    direction: str,
    entry_1: float,
    anchor_stop: float,
    atr_1h: float,
    indicators: Dict[str, TFIndicators]
) -> Tuple[float, str]:
    """
    Рассчитывает Entry_2 на основе структурного рейтинга и конфлюэнции (кластеризации) уровней:
    - Для LONG: ищет подтверждённые поддержки между anchor_stop и entry_1 (Pivot Low, VAL, EMA50, POC).
    - Для SHORT: ищет сопротивления между entry_1 и anchor_stop (Pivot High, VAH, EMA50, POC).
    - При совпадении нескольких уровней (кластер) выбирается центр масс кластера.
    - При отсутствии структурных кандидатов срабатывает fallback (геометрическая середина).
    """
    ind_1h = indicators.get("1H")
    ind_4h = indicators.get("4H")
    fallback_val = 0.5 * (entry_1 + anchor_stop)
    fallback_desc = "Fallback (Midpoint)"

    if not ind_1h:
        return fallback_val, fallback_desc

    candidates: List[Tuple[float, str, float]] = []

    if direction == "LONG":
        lower_bound = anchor_stop
        upper_bound = entry_1 - 0.20 * atr_1h

        if upper_bound <= lower_bound:
            return fallback_val, fallback_desc

        raw_candidates = [
            (ind_1h.val80, "VAL 70%", 2.5),
            (ind_1h.swing_low_20, "1H Swing Low", 2.0),
            (ind_1h.ema50, "1H EMA50", 2.0),
            (ind_1h.poc80, "1H POC", 1.5),
        ]
        if ind_1h.last_pivot_low:
            raw_candidates.append((ind_1h.last_pivot_low.price, "1H Pivot Low", 2.5))
        if ind_4h:
            raw_candidates.append((ind_4h.ema20, "4H EMA20", 1.5))

        for price_lvl, name, weight in raw_candidates:
            if lower_bound < price_lvl <= upper_bound:
                candidates.append((price_lvl, name, weight))

    else:  # SHORT
        lower_bound = entry_1 + 0.20 * atr_1h
        upper_bound = anchor_stop

        if upper_bound <= lower_bound:
            return fallback_val, fallback_desc

        raw_candidates = [
            (ind_1h.vah80, "VAH 70%", 2.5),
            (ind_1h.swing_high_20, "1H Swing High", 2.0),
            (ind_1h.ema50, "1H EMA50", 2.0),
            (ind_1h.poc80, "1H POC", 1.5),
        ]
        if ind_1h.last_pivot_high:
            raw_candidates.append((ind_1h.last_pivot_high.price, "1H Pivot High", 2.5))
        if ind_4h:
            raw_candidates.append((ind_4h.ema20, "4H EMA20", 1.5))

        for price_lvl, name, weight in raw_candidates:
            if lower_bound <= price_lvl < upper_bound:
                candidates.append((price_lvl, name, weight))

    if not candidates:
        return fallback_val, fallback_desc

    cluster_radius = 0.35 * atr_1h
    best_score = -1.0
    best_price = fallback_val
    best_desc = fallback_desc

    for i, (p_i, name_i, w_i) in enumerate(candidates):
        in_cluster = [c for c in candidates if abs(c[0] - p_i) <= cluster_radius]
        total_weight = sum(c[2] for c in in_cluster)
        cluster_price = sum(c[0] * c[2] for c in in_cluster) / total_weight

        if total_weight > best_score:
            best_score = total_weight
            best_price = cluster_price
            if len(in_cluster) > 1:
                names = " + ".join(list(dict.fromkeys(c[1] for c in in_cluster)))
                best_desc = f"Cluster ({names})"
            else:
                best_desc = in_cluster[0][1]

    return best_price, best_desc


def calculate_setup(
    instrument: InstrumentInfo,
    indicators: Dict[str, TFIndicators],
    direction: str
) -> SetupResult:
    """
    Рассчитывает сетап для заданного инструмента и направления (LONG или SHORT).
    """
    direction = direction.upper().strip()
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"Неизвестное направление: {direction}")

    ind_1h = indicators.get("1H")
    ind_4h = indicators.get("4H")
    ind_1d = indicators.get("1D")

    if not ind_1h or not ind_4h or not ind_1d:
        return _make_skip_result(
            instrument,
            direction,
            current_price=ind_1h.close if ind_1h else 0.0,
            senior_trend="UNKNOWN",
            reason="Недостаточно исторических данных по всем трём таймфреймам (1H, 4H, 1D)"
        )

    current_price = ind_1h.close
    atr_1h = ind_1h.atr14
    atr_1d = ind_1d.atr14
    step = instrument.min_price_increment
    lot = instrument.lot

    senior_trend = get_senior_trend(indicators)
    trade_type, conf_trigger, inval_trigger_desc = classify_trade_type(direction, indicators, senior_trend)

    last_ph_val = ind_1h.last_pivot_high.price if ind_1h.last_pivot_high else None
    last_pl_val = ind_1h.last_pivot_low.price if ind_1h.last_pivot_low else None
    market_structure_1h = ind_1h.market_structure

    # Предупреждения
    warnings = []
    if trade_type == "Counter Trend":
        warnings.append("ВНИМАНИЕ: КОНТРТРЕНД ПРОТИВ 1D/4H")
    elif trade_type == "Reversal Candidate":
        warnings.append("ИНФО: КАНДИДАТ В РАЗВОРОТ (Reversal Candidate 1H — требуется подтверждение)")
    elif trade_type == "Confirmed Reversal":
        warnings.append("СИЛЬНЫЙ СИГНАЛ: ПОДТВЕРЖДЁННЫЙ СЛОМ СТРУКТУРЫ (Confirmed Reversal 1H)")
    elif trade_type == "MIXED / NO TRADE":
        warnings.append("НЕОПРЕДЕЛЁННОСТЬ: РЫНОК В СМЕШАННОМ РЕЖИМЕ (Mixed Structure)")

    if direction == "SHORT" and not instrument.short_enabled_flag:
        warnings.append("ВНИМАНИЕ: Шорт по данной бумаге запрещён брокером (short_enabled_flag == False)!")

    warning = " | ".join(warnings) if warnings else None

    # -------------------------------------------------------------
    # А. НАПРАВЛЕНИЕ SHORT
    # -------------------------------------------------------------
    if direction == "SHORT":
        entry_1 = max(current_price, min(ind_1h.ema20, ind_1h.poc80))

        if current_price < ind_4h.ema50 - 0.5 * atr_1h:
            anchor_stop = ind_4h.ema50
        else:
            anchor_stop = max(ind_1d.poc80, ind_1d.ema20)

        if anchor_stop <= entry_1:
            fallback = max(ind_4h.ema50, ind_1d.poc80, ind_1d.ema20, ind_1d.swing_high_20)
            if fallback > entry_1:
                anchor_stop = fallback
            else:
                alt = max(ind_1h.swing_high_20, ind_4h.swing_high_20)
                if alt > entry_1:
                    anchor_stop = alt
                else:
                    anchor_stop = entry_1 + 1.5 * atr_1h

        entry_2, entry_2_source = calculate_structural_entry_2(
            direction="SHORT",
            entry_1=entry_1,
            anchor_stop=anchor_stop,
            atr_1h=atr_1h,
            indicators=indicators
        )

        stop = anchor_stop + 0.6 * atr_1h

        entry_1_r = round_to_step(entry_1, step)
        entry_2_r = round_to_step(entry_2, step)
        entry_avg_r = round_to_step(0.5 * entry_1_r + 0.5 * entry_2_r, step)
        stop_r = round_to_step(stop, step)

        risk_per_share_pre = abs(stop_r - entry_avg_r) + COMMISSION * entry_avg_r
        if risk_per_share_pre <= 0:
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)",
                trade_type=trade_type,
                last_ph=last_ph_val, last_pl=last_pl_val, market_structure_1h=market_structure_1h,
                conf_trigger=conf_trigger, inval_trigger=f"Закрытие 1H выше {stop_r:.2f} ₽"
            )

        levels_pool = [
            ind_1h.poc80, ind_1h.val80, ind_1h.ema20, ind_1h.ema50, ind_1h.swing_low_20,
            ind_4h.ema20, ind_4h.poc80, ind_4h.val80, ind_4h.swing_low_20
        ]
        if ind_1h.last_pivot_low:
            levels_pool.append(ind_1h.last_pivot_low.price)

        upper_bound = min(current_price, entry_1_r)
        sorted_supports = sorted(list(set(lvl for lvl in levels_pool if lvl < upper_bound)), reverse=True)

        min_drop_t1 = risk_per_share_pre + COMMISSION * entry_avg_r
        max_t1_price = entry_avg_r - min_drop_t1
        max_drop_t1 = max(2.2 * risk_per_share_pre, 3.0 * atr_1h)
        min_t1_price = entry_avg_r - max_drop_t1

        target_1 = None
        take_1 = None

        for idx, lvl in enumerate(sorted_supports):
            t1_cand = lvl + 0.2 * atr_1h
            if min_t1_price <= t1_cand <= max_t1_price and t1_cand < entry_1_r:
                target_1 = lvl
                take_1 = t1_cand
                break

        if target_1 is None:
            take_1 = entry_avg_r - max(min_drop_t1, 1.2 * risk_per_share_pre)
            target_1 = take_1 - 0.2 * atr_1h

        min_drop_t2 = max(2.0 * risk_per_share_pre, (entry_avg_r - take_1) + 0.8 * atr_1h)
        max_drop_t2 = min(4.5 * risk_per_share_pre, max(5.0 * atr_1h, 1.5 * atr_1d))
        if max_drop_t2 < min_drop_t2:
            max_drop_t2 = min_drop_t2 + 1.0 * atr_1h

        t2_window_high = entry_avg_r - min_drop_t2
        t2_window_low = entry_avg_r - max_drop_t2

        target_2 = None
        take_2 = None

        remaining_supports = [lvl for lvl in sorted_supports if lvl < target_1]
        for lvl in remaining_supports:
            t2_cand = lvl + 0.2 * atr_1h
            if t2_window_low <= t2_cand <= t2_window_high:
                target_2 = lvl
                take_2 = t2_cand
                break

        if target_2 is None:
            take_2 = entry_avg_r - min_drop_t2
            target_2 = take_2 - 0.2 * atr_1h

    # -------------------------------------------------------------
    # Б. НАПРАВЛЕНИЕ LONG
    # -------------------------------------------------------------
    else:
        entry_1 = min(current_price, max(ind_1h.ema20, ind_1h.poc80))

        if current_price > ind_4h.ema50 + 0.5 * atr_1h:
            anchor_stop = ind_4h.ema50
        else:
            anchor_stop = min(ind_1d.poc80, ind_1d.ema20)

        if anchor_stop >= entry_1:
            fallback = min(ind_4h.ema50, ind_1d.poc80, ind_1d.ema20, ind_1d.swing_low_20)
            if fallback < entry_1:
                anchor_stop = fallback
            else:
                alt = min(ind_1h.swing_low_20, ind_4h.swing_low_20)
                if alt < entry_1:
                    anchor_stop = alt
                else:
                    anchor_stop = entry_1 - 1.5 * atr_1h

        entry_2, entry_2_source = calculate_structural_entry_2(
            direction="LONG",
            entry_1=entry_1,
            anchor_stop=anchor_stop,
            atr_1h=atr_1h,
            indicators=indicators
        )

        stop = anchor_stop - 0.6 * atr_1h

        entry_1_r = round_to_step(entry_1, step)
        entry_2_r = round_to_step(entry_2, step)
        entry_avg_r = round_to_step(0.5 * entry_1_r + 0.5 * entry_2_r, step)
        stop_r = round_to_step(stop, step)

        risk_per_share_pre = abs(stop_r - entry_avg_r) + COMMISSION * entry_avg_r
        if risk_per_share_pre <= 0:
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)",
                trade_type=trade_type,
                last_ph=last_ph_val, last_pl=last_pl_val, market_structure_1h=market_structure_1h,
                conf_trigger=conf_trigger, inval_trigger=f"Закрытие 1H ниже {stop_r:.2f} ₽"
            )

        levels_pool = [
            ind_1h.poc80, ind_1h.vah80, ind_1h.ema20, ind_1h.ema50, ind_1h.swing_high_20,
            ind_4h.ema20, ind_4h.poc80, ind_4h.vah80, ind_4h.swing_high_20
        ]
        if ind_1h.last_pivot_high:
            levels_pool.append(ind_1h.last_pivot_high.price)

        lower_bound = max(current_price, entry_1_r)
        sorted_resistances = sorted(list(set(lvl for lvl in levels_pool if lvl > lower_bound)))

        min_rise_t1 = risk_per_share_pre + COMMISSION * entry_avg_r
        min_t1_price = entry_avg_r + min_rise_t1
        max_rise_t1 = max(2.2 * risk_per_share_pre, 3.0 * atr_1h)
        max_t1_price = entry_avg_r + max_rise_t1

        target_1 = None
        take_1 = None

        for idx, lvl in enumerate(sorted_resistances):
            t1_cand = lvl - 0.2 * atr_1h
            if min_t1_price <= t1_cand <= max_t1_price and t1_cand > entry_1_r:
                target_1 = lvl
                take_1 = t1_cand
                break

        if target_1 is None:
            take_1 = entry_avg_r + max(min_rise_t1, 1.2 * risk_per_share_pre)
            target_1 = take_1 + 0.2 * atr_1h

        min_rise_t2 = max(2.0 * risk_per_share_pre, (take_1 - entry_avg_r) + 0.8 * atr_1h)
        max_rise_t2 = min(4.5 * risk_per_share_pre, max(5.0 * atr_1h, 1.5 * atr_1d))
        if max_rise_t2 < min_rise_t2:
            max_rise_t2 = min_rise_t2 + 1.0 * atr_1h

        t2_window_low = entry_avg_r + min_rise_t2
        t2_window_high = entry_avg_r + max_rise_t2

        target_2 = None
        take_2 = None

        remaining_resistances = [lvl for lvl in sorted_resistances if lvl > target_1]
        for lvl in remaining_resistances:
            t2_cand = lvl - 0.2 * atr_1h
            if t2_window_low <= t2_cand <= t2_window_high:
                target_2 = lvl
                take_2 = t2_cand
                break

        if target_2 is None:
            take_2 = entry_avg_r + min_rise_t2
            target_2 = take_2 + 0.2 * atr_1h

    # -------------------------------------------------------------
    # В. ОКРУГЛЕНИЕ И ПРОВЕРКА КОРРЕКТНОСТИ СТРУКТУРЫ УРОВНЕЙ
    # -------------------------------------------------------------
    entry_1 = round_to_step(entry_1, step)
    entry_2 = round_to_step(entry_2, step)
    anchor_stop = round_to_step(anchor_stop, step)
    stop = round_to_step(stop, step)
    target_1 = round_to_step(target_1, step)
    target_2 = round_to_step(target_2, step)
    take_1 = round_to_step(take_1, step)
    take_2 = round_to_step(take_2, step)

    entry_avg = round_to_step(0.5 * entry_1 + 0.5 * entry_2, step)

    invalidation_trigger = f"Закрытие 1H бара ниже {stop:.2f} ₽" if direction == "LONG" else f"Закрытие 1H бара выше {stop:.2f} ₽"

    if direction == "SHORT":
        if not (stop > entry_avg > take_2 and take_1 > take_2 and entry_1 > take_1):
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason=f"Некорректная структура уровней SHORT (Stop={stop}, Entry_avg={entry_avg}, Take1={take_1}, Take2={take_2})",
                trade_type=trade_type,
                last_ph=last_ph_val, last_pl=last_pl_val, market_structure_1h=market_structure_1h,
                conf_trigger=conf_trigger, inval_trigger=invalidation_trigger
            )
    else:
        if not (stop < entry_avg < take_2 and take_1 < take_2 and entry_1 < take_1):
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason=f"Некорректная структура уровней LONG (Stop={stop}, Entry_avg={entry_avg}, Take1={take_1}, Take2={take_2})",
                trade_type=trade_type,
                last_ph=last_ph_val, last_pl=last_pl_val, market_structure_1h=market_structure_1h,
                conf_trigger=conf_trigger, inval_trigger=invalidation_trigger
            )

    # -------------------------------------------------------------
    # Г. РАСЧЁТ РИСКА, REWARD И СРЕДНЕВЗВЕШЕННОГО R/R
    # -------------------------------------------------------------
    risk_per_share = abs(stop - entry_avg) + COMMISSION * entry_avg
    reward_t1_per_share = abs(take_1 - entry_avg) - COMMISSION * entry_avg
    reward_t2_per_share = abs(take_2 - entry_avg) - COMMISSION * entry_avg

    reward_avg_per_share = 0.5 * reward_t1_per_share + 0.5 * reward_t2_per_share

    if risk_per_share <= 0:
        return _make_skip_result(
            instrument, direction, current_price, senior_trend,
            reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)",
            trade_type=trade_type,
            last_ph=last_ph_val, last_pl=last_pl_val, market_structure_1h=market_structure_1h,
            conf_trigger=conf_trigger, inval_trigger=invalidation_trigger
        )

    rr = reward_avg_per_share / risk_per_share

    take_avg = 0.5 * (take_1 + take_2)
    if direction == "SHORT":
        ideal_entry = (1.8 * stop + take_avg) / 2.7986
    else:
        ideal_entry = (1.8 * stop + take_avg) / 2.8014
    ideal_entry = round_to_step(ideal_entry, step)

    # -------------------------------------------------------------
    # Д. СТАТУС И АДАПТИВНЫЙ ЛИМИТ РИСКА
    # -------------------------------------------------------------
    skip_reason = None
    if rr < MIN_RR_THRESHOLD:
        status = "SKIP (Недостаточный R/R)"
        risk_rub = 0.0
        skip_reason = f"Средневзвешенное соотношение R/R ({rr:.2f}) ниже минимального порога 1.80"
    elif trade_type == "MIXED / NO TRADE":
        status = "SKIP (Рынок в распиле)"
        risk_rub = 0.0
        skip_reason = "Смешанная структура рынка (MIXED / NO TRADE) — нет направленного преимущества"
    elif rr < STRONG_RR_THRESHOLD:
        status = "ACTIVE (Standard)"
        risk_rub = RISK_STANDARD
    else:
        status = "ACTIVE (Strong)"
        risk_rub = RISK_STRONG

    # -------------------------------------------------------------
    # Е. РАСЧЁТ ЛОТНОСТИ И ФИНАНСОВОГО РЕЗУЛЬТАТА В РУБЛЯХ
    # -------------------------------------------------------------
    if risk_rub > 0 and risk_per_share > 0:
        total_shares = int(risk_rub // risk_per_share)
        total_lots = total_shares // lot
    else:
        total_shares = 0
        total_lots = 0

    if total_lots == 0:
        lots_entry_1 = 0
        lots_entry_2 = 0
        lots_take_1 = 0
        lots_take_2 = 0
        actual_loss_rub = 0.0
        profit_take_1_rub = 0.0
        profit_take_2_rub = 0.0
        total_profit_rub = 0.0
    else:
        lots_entry_1 = total_lots // 2
        lots_entry_2 = total_lots - lots_entry_1

        if total_lots == 1:
            lots_take_1 = 0
            lots_take_2 = 1
        else:
            lots_take_1 = total_lots // 2
            lots_take_2 = total_lots - lots_take_1

        shares_take_1 = lots_take_1 * lot
        shares_take_2 = lots_take_2 * lot
        actual_total_shares = total_lots * lot

        actual_loss_rub = actual_total_shares * risk_per_share

        profit_take_1_rub = shares_take_1 * max(0.0, reward_t1_per_share)
        profit_take_2_rub = shares_take_2 * max(0.0, reward_t2_per_share)
        total_profit_rub = profit_take_1_rub + profit_take_2_rub

    return SetupResult(
        ticker=instrument.ticker,
        direction=direction,
        status=status,
        trade_type=trade_type,
        current_price=current_price,
        senior_trend=senior_trend,
        market_structure_1h=market_structure_1h,
        last_pivot_high=last_ph_val,
        last_pivot_low=last_pl_val,
        confirmation_trigger=conf_trigger,
        invalidation_trigger=invalidation_trigger,
        anchor_stop=anchor_stop,
        target_1=target_1,
        target_2=target_2,
        entry_1=entry_1,
        entry_2=entry_2,
        entry_2_source=entry_2_source,
        entry_avg=entry_avg,
        stop=stop,
        take_1=take_1,
        take_2=take_2,
        risk_per_share=risk_per_share,
        reward_t1_per_share=reward_t1_per_share,
        reward_t2_per_share=reward_t2_per_share,
        reward_avg_per_share=reward_avg_per_share,
        reward_per_share=reward_avg_per_share,
        rr=rr,
        risk_rub=risk_rub,
        ideal_entry=ideal_entry,
        total_shares=total_shares,
        total_lots=total_lots,
        lots_entry_1=lots_entry_1,
        lots_entry_2=lots_entry_2,
        lots_take_1=lots_take_1,
        lots_take_2=lots_take_2,
        actual_loss_rub=actual_loss_rub,
        profit_take_1_rub=profit_take_1_rub,
        profit_take_2_rub=profit_take_2_rub,
        total_profit_rub=total_profit_rub,
        step=step,
        lot_size=lot,
        skip_reason=skip_reason,
        warning=warning
    )


def _make_skip_result(
    instrument: InstrumentInfo,
    direction: str,
    current_price: float,
    senior_trend: str,
    reason: str,
    trade_type: str = "Counter Trend",
    last_ph: Optional[float] = None,
    last_pl: Optional[float] = None,
    market_structure_1h: str = "Undefined",
    conf_trigger: str = "—",
    inval_trigger: str = "—"
) -> SetupResult:
    """Создаёт результат со статусом SKIP при невозможности построения сетапа."""
    return SetupResult(
        ticker=instrument.ticker,
        direction=direction,
        status="SKIP",
        trade_type=trade_type,
        current_price=current_price,
        senior_trend=senior_trend,
        market_structure_1h=market_structure_1h,
        last_pivot_high=last_ph,
        last_pivot_low=last_pl,
        confirmation_trigger=conf_trigger,
        invalidation_trigger=inval_trigger,
        anchor_stop=0.0,
        target_1=0.0,
        target_2=0.0,
        entry_1=0.0,
        entry_2=0.0,
        entry_2_source="None",
        entry_avg=0.0,
        stop=0.0,
        take_1=0.0,
        take_2=0.0,
        risk_per_share=0.0,
        reward_t1_per_share=0.0,
        reward_t2_per_share=0.0,
        reward_avg_per_share=0.0,
        reward_per_share=0.0,
        rr=0.0,
        risk_rub=0.0,
        ideal_entry=None,
        total_shares=0,
        total_lots=0,
        lots_entry_1=0,
        lots_entry_2=0,
        lots_take_1=0,
        lots_take_2=0,
        actual_loss_rub=0.0,
        profit_take_1_rub=0.0,
        profit_take_2_rub=0.0,
        total_profit_rub=0.0,
        step=instrument.min_price_increment,
        lot_size=instrument.lot,
        skip_reason=reason
    )
