from dataclasses import dataclass
from typing import Optional, Dict, List
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
    direction: str  # "LONG" или "SHORT"
    status: str     # "ACTIVE (Strong)", "ACTIVE (Standard)", "SKIP (...)"
    current_price: float
    senior_trend: str
    anchor_stop: float
    target_1: float
    target_2: float
    entry_1: float
    entry_2: float
    entry_avg: float
    stop: float
    take_1: float
    take_2: float
    risk_per_share: float
    reward_t1_per_share: float
    reward_t2_per_share: float
    reward_avg_per_share: float
    reward_per_share: float  # Для обратной совместимости = reward_avg_per_share
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
    """
    Определяет господствующий тренд на основе 4H_EMA50 и 1D_EMA20:
    - Current_Price < 4H_EMA50 и Current_Price < 1D_EMA20 -> BEARISH
    - Current_Price > 4H_EMA50 и Current_Price > 1D_EMA20 -> BULLISH
    - Иначе -> NEUTRAL
    """
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

    # Предупреждения: контртренд и шорт брокера
    warnings = []
    if direction == "LONG" and senior_trend == "BEARISH":
        warnings.append("ВНИМАНИЕ: КОНТРТРЕНД ПРОТИВ 1D/4H")
    elif direction == "SHORT" and senior_trend == "BULLISH":
        warnings.append("ВНИМАНИЕ: КОНТРТРЕНД ПРОТИВ 1D/4H")

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
        # Старший уровень сопротивления строго выше текущей цены:
        # 4H_EMA50 (если Current_Price < 4H_EMA50 - 0.5 * ATR_1H)
        # либо max(1D_POC, 1D_EMA20)
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

        # Entry_2 (50%): 0.5 * (Entry_1 + Anchor_Stop) (строго ниже Stop)
        entry_2 = 0.5 * (entry_1 + anchor_stop)

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
                reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)"
            )

        # Пул локальных уровней поддержки для SHORT (1H и 4H):
        levels_pool = [
            ind_1h.poc80, ind_1h.ema20, ind_1h.ema50, ind_1h.swing_low_20,
            ind_4h.ema20, ind_4h.poc80, ind_4h.swing_low_20
        ]
        upper_bound = min(current_price, entry_1_r)
        sorted_supports = sorted(list(set(lvl for lvl in levels_pool if lvl < upper_bound)), reverse=True)

        # Реалистичный коридор для Take 1:
        # Минимальный ход: Reward_T1 >= 1.0 * Risk_per_share
        min_drop_t1 = risk_per_share_pre + COMMISSION * entry_avg_r
        max_t1_price = entry_avg_r - min_drop_t1
        # Максимальный ход для Take 1: до 2.2 * Risk или 3.0 * ATR_1H
        max_drop_t1 = max(2.2 * risk_per_share_pre, 3.0 * atr_1h)
        min_t1_price = entry_avg_r - max_drop_t1

        target_1 = None
        take_1 = None
        target_1_idx = -1

        for idx, lvl in enumerate(sorted_supports):
            t1_cand = lvl + 0.2 * atr_1h
            if min_t1_price <= t1_cand <= max_t1_price and t1_cand < entry_1_r:
                target_1 = lvl
                take_1 = t1_cand
                target_1_idx = idx
                break

        if target_1 is None:
            # Если нет локального уровня в коридоре, ставим Take 1 на 1.2 * Risk
            take_1 = entry_avg_r - max(min_drop_t1, 1.2 * risk_per_share_pre)
            target_1 = take_1 - 0.2 * atr_1h

        # Реалистичный коридор для Take 2:
        # Должен быть ниже Take 1, в диапазоне 2.0 * Risk ... 4.2 * Risk (не более 1.5 * ATR_1D / 5.5 * ATR_1H)
        min_drop_t2 = max(2.0 * risk_per_share_pre, (entry_avg_r - take_1) + 0.8 * atr_1h)
        max_drop_t2 = min(4.5 * risk_per_share_pre, max(5.0 * atr_1h, 1.5 * atr_1d))
        if max_drop_t2 < min_drop_t2:
            max_drop_t2 = min_drop_t2 + 1.0 * atr_1h

        t2_window_high = entry_avg_r - min_drop_t2  # верхняя планка цены для T2 (< take_1)
        t2_window_low = entry_avg_r - max_drop_t2   # нижняя реалистичная граница

        target_2 = None
        take_2 = None

        # Ищем уровень поддержки ниже target_1, попадающий в реалистичный коридор T2
        remaining_supports = [lvl for lvl in sorted_supports if lvl < target_1]
        for lvl in remaining_supports:
            t2_cand = lvl + 0.2 * atr_1h
            if t2_window_low <= t2_cand <= t2_window_high:
                target_2 = lvl
                take_2 = t2_cand
                break

        if target_2 is None:
            # Устанавливаем сбалансированный свинговый Тейк 2 (2.5 * Risk)
            take_2 = entry_avg_r - min_drop_t2
            target_2 = take_2 - 0.2 * atr_1h

    # -------------------------------------------------------------
    # Б. НАПРАВЛЕНИЕ LONG
    # -------------------------------------------------------------
    else:
        # 1. Entry_1 (50%): min(Current_Price, max(1H_EMA20, 1H_POC))
        entry_1 = min(current_price, max(ind_1h.ema20, ind_1h.poc80))

        # 2. Anchor_Stop:
        # Старший уровень поддержки строго ниже текущей цены:
        # 4H_EMA50 (если Current_Price > 4H_EMA50 + 0.5 * ATR_1H)
        # либо min(1D_POC, 1D_EMA20)
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

        # Entry_2 (50%): 0.5 * (Entry_1 + Anchor_Stop) (строго выше Stop)
        entry_2 = 0.5 * (entry_1 + anchor_stop)

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
                reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)"
            )

        # Пул локальных уровней сопротивления для LONG (1H и 4H):
        levels_pool = [
            ind_1h.poc80, ind_1h.ema20, ind_1h.ema50, ind_1h.swing_high_20,
            ind_4h.ema20, ind_4h.poc80, ind_4h.swing_high_20
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
        target_1_idx = -1

        for idx, lvl in enumerate(sorted_resistances):
            t1_cand = lvl - 0.2 * atr_1h
            if min_t1_price <= t1_cand <= max_t1_price and t1_cand > entry_1_r:
                target_1 = lvl
                take_1 = t1_cand
                target_1_idx = idx
                break

        if target_1 is None:
            take_1 = entry_avg_r + max(min_rise_t1, 1.2 * risk_per_share_pre)
            target_1 = take_1 + 0.2 * atr_1h

        # Реалистичный коридор для Take 2:
        min_rise_t2 = max(2.0 * risk_per_share_pre, (take_1 - entry_avg_r) + 0.8 * atr_1h)
        max_rise_t2 = min(4.5 * risk_per_share_pre, max(5.0 * atr_1h, 1.5 * atr_1d))
        if max_rise_t2 < min_rise_t2:
            max_rise_t2 = min_rise_t2 + 1.0 * atr_1h

        t2_window_low = entry_avg_r + min_rise_t2   # нижняя планка для T2 (> take_1)
        t2_window_high = entry_avg_r + max_rise_t2  # верхняя реалистичная граница

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

    # Проверка геометрической корректности структуры
    if direction == "SHORT":
        if not (stop > entry_avg > take_2 and take_1 > take_2 and entry_1 > take_1):
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason=f"Некорректная структура уровней SHORT (Stop={stop}, Entry_avg={entry_avg}, Take1={take_1}, Take2={take_2})"
            )
    else:
        if not (stop < entry_avg < take_2 and take_1 < take_2 and entry_1 < take_1):
            return _make_skip_result(
                instrument, direction, current_price, senior_trend,
                reason=f"Некорректная структура уровней LONG (Stop={stop}, Entry_avg={entry_avg}, Take1={take_1}, Take2={take_2})"
            )

    # -------------------------------------------------------------
    # Г. РАСЧЁТ РИСКА, REWARD И СРЕДНЕВЗВЕШЕННОГО R/R
    # -------------------------------------------------------------
    risk_per_share = abs(stop - entry_avg) + COMMISSION * entry_avg
    reward_t1_per_share = abs(take_1 - entry_avg) - COMMISSION * entry_avg
    reward_t2_per_share = abs(take_2 - entry_avg) - COMMISSION * entry_avg

    # Средневзвешенная прибыль двух тейков (50% + 50%)
    reward_avg_per_share = 0.5 * reward_t1_per_share + 0.5 * reward_t2_per_share

    if risk_per_share <= 0:
        return _make_skip_result(
            instrument, direction, current_price, senior_trend,
            reason="Риск на акцию <= 0 (стоп совпадает со средней ценой входа)"
        )

    # Итоговый R/R считается по средневзвешенной прибыли
    rr = reward_avg_per_share / risk_per_share

    # Расчёт Ideal_Entry при RR = 1.8 по средневзвешенной цене тейков Take_avg = 0.5 * (Take_1 + Take_2)
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

        # Поправка ТЗ: если 1 лот, фиксируем 100% на Take 2 с безубытком на Take 1
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
        current_price=current_price,
        senior_trend=senior_trend,
        anchor_stop=anchor_stop,
        target_1=target_1,
        target_2=target_2,
        entry_1=entry_1,
        entry_2=entry_2,
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
    reason: str
) -> SetupResult:
    """Создаёт результат со статусом SKIP при невозможности построения сетапа."""
    return SetupResult(
        ticker=instrument.ticker,
        direction=direction,
        status="SKIP",
        current_price=current_price,
        senior_trend=senior_trend,
        anchor_stop=0.0,
        target_1=0.0,
        target_2=0.0,
        entry_1=0.0,
        entry_2=0.0,
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
