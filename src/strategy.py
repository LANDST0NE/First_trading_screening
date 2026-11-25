from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from src.indicators import TFIndicators
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
    direction: str       # "LONG" или "SHORT"
    status: str          # "ACTIVE (Strong)", "ACTIVE (Standard)", "SKIP (...)"
    trade_type: str      # "Trend Following", "Trend Reversal", "Counter Trend"
    current_price: float
    senior_trend: str
    anchor_stop: float
    target_1: float
    target_2: float
    entry_1: float
    entry_2: float
    entry_2_source: str  # Источник второго входа: "Cluster (VAL + EMA50)", "Swing Low", etc.
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
    return "NEUTRAL"


def classify_trade_type(direction: str, indicators: Dict[str, TFIndicators], senior_trend: str) -> str:
    """
    Классифицирует сделку по трендовому типу:
    - Trend Following: по направлению старшего тренда 1D/4H.
    - Trend Reversal: против старшего тренда, но с подтверждённым сломом структуры на 1H.
    - Counter Trend: чистый контртренд без подтверждения смены структуры.
    """
    ind_1h = indicators.get("1H")
    if not ind_1h:
        return "Counter Trend" if senior_trend in ("BEARISH", "BULLISH") else "Trend Following"

    price = ind_1h.close

    if direction == "LONG":
        if senior_trend == "BULLISH":
            return "Trend Following"
        elif senior_trend == "BEARISH":
            # Слом структуры вверх на 1H: цена выше 1H EMA50 и выше 1H POC, RSI не в экстремальной перепроданности
            if price > ind_1h.ema50 and price >= ind_1h.poc80 and ind_1h.rsi14 >= 45.0:
                return "Trend Reversal"
            return "Counter Trend"
        else:
            return "Trend Following" if price >= ind_1h.ema50 else "Counter Trend"

    else:  # SHORT
        if senior_trend == "BEARISH":
            return "Trend Following"
        elif senior_trend == "BULLISH":
            # Слом структуры вниз на 1H: цена ниже 1H EMA50 и ниже 1H POC
            if price < ind_1h.ema50 and price <= ind_1h.poc80 and ind_1h.rsi14 <= 55.0:
                return "Trend Reversal"
            return "Counter Trend"
        else:
            return "Trend Following" if price <= ind_1h.ema50 else "Counter Trend"


def calculate_structural_entry_2(
    direction: str,
    entry_1: float,
    anchor_stop: float,
    atr_1h: float,
    indicators: Dict[str, TFIndicators]
) -> Tuple[float, str]:
    """
    Рассчитывает Entry_2 на основе структурного рейтинга и конфлюэнции (кластеризации) уровней:
    - Для LONG: ищет подтверждённые поддержки между anchor_stop и entry_1 (VAL, Swing Low, EMA50, POC).
    - Для SHORT: ищет сопротивления между entry_1 и anchor_stop (VAH, Swing High, EMA50, POC).
    - При совпадении нескольких уровней (кластер) выбирается центр кластера.
    - При отсутствии структурных кандидатов срабатывает fallback (геометрическая середина).
    """
    ind_1h = indicators.get("1H")
    ind_4h = indicators.get("4H")
    fallback_val = 0.5 * (entry_1 + anchor_stop)
    fallback_desc = "Fallback (Midpoint)"

    if not ind_1h:
        return fallback_val, fallback_desc

    candidates: List[Tuple[float, str, float]] = []  # (цена, название, вес)

    if direction == "LONG":
        lower_bound = anchor_stop
        # Отступаем от entry_1 минимум на 0.2 ATR, чтобы уровни не слипались
        upper_bound = entry_1 - 0.20 * atr_1h

        if upper_bound <= lower_bound:
            return fallback_val, fallback_desc

        raw_candidates = [
            (ind_1h.val80, "VAL 70%", 2.5),
            (ind_1h.swing_low_20, "1H Swing Low", 2.0),
            (ind_1h.ema50, "1H EMA50", 2.0),
            (ind_1h.poc80, "1H POC", 1.5),
        ]
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
        if ind_4h:
            raw_candidates.append((ind_4h.ema20, "4H EMA20", 1.5))

        for price_lvl, name, weight in raw_candidates:
            if lower_bound <= price_lvl < upper_bound:
                candidates.append((price_lvl, name, weight))

    if not candidates:
        return fallback_val, fallback_desc

    # Поиск кластеров (конфлюэнция уровней в радиусе 0.35 * ATR)
    cluster_radius = 0.35 * atr_1h
    best_score = -1.0
    best_price = fallback_val
    best_desc = fallback_desc

    for i, (p_i, name_i, w_i) in enumerate(candidates):
        # Собираем все уровни, попадающие в окрестность p_i
        in_cluster = [c for c in candidates if abs(c[0] - p_i) <= cluster_radius]
        total_weight = sum(c[2] for c in in_cluster)
        # Центр масс кластера
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
    trade_type = classify_trade_type(direction, indicators, senior_trend)

    # Предупреждения: контртренд и шорт брокера
    warnings = []
    if trade_type == "Counter Trend":
        warnings.append("ВНИМАНИЕ: КОНТРТРЕНД ПРОТИВ 1D/4H")
    elif trade_type == "Trend Reversal":
        warnings.append("ИНФО: ПОТЕНЦИАЛЬНЫЙ РАЗВОРОТ СТРУКТУРЫ (Trend Reversal 1H)")

    if direction == "SHORT" and not instrument.short_enabled_flag:
        warnings.append("ВНИМАНИЕ: Шорт по данной бумаге запрещён брокером (short_enabled_flag == False)!")

    warning = " | ".join(warnings) if warnings else None

    # -------------------------------------------------------------
    # А. НАПРАВЛЕНИЕ SHORT
    # -------------------------------------------------------------
    if direction == "SHORT":
        # 1. Entry_1 (50%): max(Current_Price, min(1H_EMA20, 1H_POC))
        entry_1 = max(current_price, min(ind_1h.ema20, ind_1h.poc80))

        # 2. Anchor_Stop:
        if current_price < ind_4h.ema50 - 0.5 * atr_1h:
            anchor_stop = ind_4h.ema50
        else:
            anchor_stop = max(ind_1d.poc80, ind_1d.ema20)

        # Гарантия: Anchor_Stop строго выше Entry_1
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

        # 3. Структурный Entry_2 (50%): рейтинг уровней (VAL, Swing, EMA)
        entry_2, entry_2_source = calculate_structural_entry_2(
            direction="SHORT",
            entry_1=entry_1,
            anchor_stop=anchor_stop,
            atr_1h=atr_1h,
            indicators=indicators
        )

        # Stop
        stop = anchor_stop + 0.6 * atr_1h

        # Округляем промежуточные значения для точной оценки риска
        entry_1_r = round_to_step(entry_1, step)
        entry_2_r = round_to_step(entry_2, step)
        entry_avg_r = round_to_step(0.5 * entry_1_r + 0.5 * entry_2_r, step)
        stop_r = round_to_step(stop, step)

        risk_per_share_pre = abs(stop_r - entry_avg_r) + COMMISSION * entry_avg_r
        if risk_per_share_pre <= 0:
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)",
                trade_type=trade_type
            )

        # Пул локальных уровней поддержки для SHORT (1H и 4H):
        levels_pool = [
            ind_1h.poc80, ind_1h.val80, ind_1h.ema20, ind_1h.ema50, ind_1h.swing_low_20,
            ind_4h.ema20, ind_4h.poc80, ind_4h.val80, ind_4h.swing_low_20
        ]
        upper_bound = min(current_price, entry_1_r)
        sorted_supports = sorted(list(set(lvl for lvl in levels_pool if lvl < upper_bound)), reverse=True)

        # Реалистичный коридор для Take 1:
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

        # Реалистичный коридор для Take 2:
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
        # 1. Entry_1 (50%): min(Current_Price, max(1H_EMA20, 1H_POC))
        entry_1 = min(current_price, max(ind_1h.ema20, ind_1h.poc80))

        # 2. Anchor_Stop:
        if current_price > ind_4h.ema50 + 0.5 * atr_1h:
            anchor_stop = ind_4h.ema50
        else:
            anchor_stop = min(ind_1d.poc80, ind_1d.ema20)

        # Гарантия: Anchor_Stop строго ниже Entry_1
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

        # 3. Структурный Entry_2 (50%): рейтинг уровней (VAL, Swing, EMA)
        entry_2, entry_2_source = calculate_structural_entry_2(
            direction="LONG",
            entry_1=entry_1,
            anchor_stop=anchor_stop,
            atr_1h=atr_1h,
            indicators=indicators
        )

        # Stop
        stop = anchor_stop - 0.6 * atr_1h

        # Округляем промежуточные значения для точной оценки риска
        entry_1_r = round_to_step(entry_1, step)
        entry_2_r = round_to_step(entry_2, step)
        entry_avg_r = round_to_step(0.5 * entry_1_r + 0.5 * entry_2_r, step)
        stop_r = round_to_step(stop, step)

        risk_per_share_pre = abs(stop_r - entry_avg_r) + COMMISSION * entry_avg_r
        if risk_per_share_pre <= 0:
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)",
                trade_type=trade_type
            )

        # Пул локальных уровней сопротивления для LONG (1H и 4H):
        levels_pool = [
            ind_1h.poc80, ind_1h.vah80, ind_1h.ema20, ind_1h.ema50, ind_1h.swing_high_20,
            ind_4h.ema20, ind_4h.poc80, ind_4h.vah80, ind_4h.swing_high_20
        ]
        lower_bound = max(current_price, entry_1_r)
        sorted_resistances = sorted(list(set(lvl for lvl in levels_pool if lvl > lower_bound)))

        # Реалистичный коридор для Take 1:
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

        # Реалистичный коридор для Take 2:
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

    if direction == "SHORT":
        if not (stop > entry_avg > take_2 and take_1 > take_2 and entry_1 > take_1):
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason=f"Некорректная структура уровней SHORT (Stop={stop}, Entry_avg={entry_avg}, Take1={take_1}, Take2={take_2})",
                trade_type=trade_type
            )
    else:
        if not (stop < entry_avg < take_2 and take_1 < take_2 and entry_1 < take_1):
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason=f"Некорректная структура уровней LONG (Stop={stop}, Entry_avg={entry_avg}, Take1={take_1}, Take2={take_2})",
                trade_type=trade_type
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
            trade_type=trade_type
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
    trade_type: str = "Counter Trend"
) -> SetupResult:
    """Создаёт результат со статусом SKIP при невозможности построения сетапа."""
    return SetupResult(
        ticker=instrument.ticker,
        direction=direction,
        status="SKIP",
        trade_type=trade_type,
        current_price=current_price,
        senior_trend=senior_trend,
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
