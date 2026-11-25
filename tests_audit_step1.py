"""
Комплексный автоматический аудит и тестирование Шага 1 после внесения исправлений:
1. Pivot Points (L=3, R=2, walk-forward без look-ahead bias).
2. Классификация сетапов (Trend Following, Retest Confirmed, Breakout Confirmed, Reversal Candidate, Counter Trend, MIXED / NO TRADE).
3. Разделение Breakout Confirmed и Retest Confirmed (окно 3 бара, проверка зоны и закрытия, отсутствие ретеста на свече пробоя).
4. Проверка фильтрации касания, тени и ложного пробоя.
5. Защитные буферы Entry 2 (0.20 ATR от Stop и 0.20 ATR от Entry 1, блокировка при узком диапазоне).
6. Работа только с закрытыми свечами (фильтрация is_complete=False).
7. Синхронизация расчётных порогов и отчётов.
"""

import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from src.config import (
    BREAKOUT_BUFFER_ATR_MULT,
    MIN_DISTANCE_STOP_ATR_MULT,
    MIN_DISTANCE_ENTRY_ATR_MULT,
    RETEST_WINDOW_BARS,
    RETEST_ZONE_PENETRATION_MULT
)
from src.indicators import (
    PivotPoint,
    BreakoutRetestInfo,
    find_confirmed_pivots,
    evaluate_market_structure,
    check_breakout_and_retest,
    calculate_tf_indicators,
    TFIndicators,
)
from src.strategy import (
    classify_trade_type,
    calculate_structural_entry_2,
    calculate_setup,
    get_senior_trend,
)
from src.client import InstrumentInfo


def make_dummy_indicators(
    close: float,
    ema50_1h: float,
    ema20_1h: float,
    poc80_1h: float,
    rsi14_1h: float,
    atr14_1h: float,
    pivot_high: Optional[float] = None,
    pivot_low: Optional[float] = None,
    market_structure: str = "Range / Neutral",
    val80_1h: float = 0.0,
    vah80_1h: float = 0.0,
    ema50_4h: float = 100.0,
    ema20_1d: float = 100.0,
    bo_long: Optional[BreakoutRetestInfo] = None,
    bo_short: Optional[BreakoutRetestInfo] = None,
) -> Dict[str, TFIndicators]:
    last_ph = PivotPoint(price=pivot_high, formation_idx=5, confirmation_idx=7, formation_time=None, confirmation_time=None, is_high=True) if pivot_high else None
    last_pl = PivotPoint(price=pivot_low, formation_idx=10, confirmation_idx=12, formation_time=None, confirmation_time=None, is_high=False) if pivot_low else None

    ind_1h = TFIndicators(
        tf="1H",
        ema20=ema20_1h,
        ema50=ema50_1h,
        atr14=atr14_1h,
        rsi14=rsi14_1h,
        poc80=poc80_1h,
        val80=val80_1h,
        vah80=vah80_1h,
        swing_low_20=pivot_low or (close - 2.0),
        swing_high_20=pivot_high or (close + 2.0),
        last_pivot_high=last_ph,
        last_pivot_low=last_pl,
        market_structure=market_structure,
        close=close,
        candles_count=100,
        last_closed_close=close,
        is_last_candle_closed=True,
        breakout_info_long=bo_long,
        breakout_info_short=bo_short
    )

    ind_4h = TFIndicators(
        tf="4H",
        ema20=ema50_4h,
        ema50=ema50_4h,
        atr14=atr14_1h * 1.5,
        rsi14=50.0,
        poc80=close,
        val80=close - 5,
        vah80=close + 5,
        swing_low_20=close - 5,
        swing_high_20=close + 5,
        last_pivot_high=None,
        last_pivot_low=None,
        market_structure="Neutral",
        close=close,
        candles_count=100
    )

    ind_1d = TFIndicators(
        tf="1D",
        ema20=ema20_1d,
        ema50=ema20_1d,
        atr14=atr14_1h * 2.5,
        rsi14=50.0,
        poc80=close,
        val80=close - 10,
        vah80=close + 10,
        swing_low_20=close - 10,
        swing_high_20=close + 10,
        last_pivot_high=None,
        last_pivot_low=None,
        market_structure="Neutral",
        close=close,
        candles_count=100
    )

    return {"1H": ind_1h, "4H": ind_4h, "1D": ind_1d}


def test_pivots_and_lookahead():
    print("=" * 80)
    print("ТЕСТ 1: PIVOT POINTS (L=3, R=2, ZERO LOOK-AHEAD BIAS)")
    print("=" * 80)

    prices = [
        {"time": f"2026-01-01 {i:02d}:00", "open": 100, "high": 100 + i, "low": 98, "close": 99 + i, "volume": 1000}
        for i in range(5)
    ]
    # свеча 5 - пик 115.0
    prices.append({"time": "2026-01-01 05:00", "open": 104, "high": 115.0, "low": 103, "close": 110.0, "volume": 1500})
    prices.append({"time": "2026-01-01 06:00", "open": 110, "high": 111.0, "low": 102, "close": 105.0, "volume": 1200})
    prices.append({"time": "2026-01-01 07:00", "open": 105, "high": 106.0, "low": 98, "close": 100.0, "volume": 1100})
    prices.append({"time": "2026-01-01 08:00", "open": 100, "high": 101.0, "low": 95, "close": 96.0, "volume": 1000})
    prices.append({"time": "2026-01-01 09:00", "open": 96, "high": 97.0, "low": 92, "close": 93.0, "volume": 900})
    # свеча 10 - дно 88.0
    prices.append({"time": "2026-01-01 10:00", "open": 93, "high": 94.0, "low": 88.0, "close": 90.0, "volume": 2000})
    prices.append({"time": "2026-01-01 11:00", "open": 90, "high": 96.0, "low": 89.5, "close": 95.0, "volume": 1300})
    prices.append({"time": "2026-01-01 12:00", "open": 95, "high": 99.0, "low": 94.0, "close": 98.0, "volume": 1400})
    for i, idx in enumerate(range(13, 18)):
        prices.append({"time": f"2026-01-01 {idx:02d}:00", "open": 98 + i*2, "high": 102 + i*2, "low": 97 + i*2, "close": 101 + i*2, "volume": 1000})
    # свеча 18 - пик 130.0 (неподтверждённый)
    prices.append({"time": "2026-01-01 18:00", "open": 108, "high": 130.0, "low": 107, "close": 125.0, "volume": 3000})
    # свеча 19 - последняя закрытая свеча
    prices.append({"time": "2026-01-01 19:00", "open": 125, "high": 126.0, "low": 120, "close": 122.0, "volume": 2000})

    df = pd.DataFrame(prices)
    p_highs, p_lows = find_confirmed_pivots(df, left=3, right=2)

    assert len(p_highs) == 1 and p_highs[0].price == 115.0 and p_highs[0].formation_idx == 5 and p_highs[0].confirmation_idx == 7
    assert len(p_lows) == 1 and p_lows[0].price == 88.0 and p_lows[0].formation_idx == 10 and p_lows[0].confirmation_idx == 12
    assert not any(p.formation_idx == 18 for p in p_highs), "Неподтверждённый пик свечи 18 не должен быть подтверждён!"
    print("  [PASS] Пивоты подтверждаются строго на свече i+2. Неподтверждённые пивоты исключены.")


def test_unclosed_candle_filtering():
    print("\n" + "=" * 80)
    print("ТЕСТ 2: ФИЛЬТРАЦИЯ НЕЗАКРЫТЫХ СВЕЧЕЙ (is_complete = False)")
    print("=" * 80)

    # Создаём датафрейм из 60 закрытых свечей + 1 незакрытой с экстремальным ложным скачком
    candles = [
        {"time": f"2026-01-01 {i:02d}:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000, "is_complete": True}
        for i in range(60)
    ]
    # 61-я свеча (незакрытая, в процессе формирования)
    candles.append({
        "time": "2026-01-03 12:00", "open": 100.0, "high": 150.0, "low": 99.0, "close": 145.0, "volume": 5000, "is_complete": False
    })
    df = pd.DataFrame(candles)

    ind = calculate_tf_indicators(df, "1H")
    assert ind is not None
    assert ind.is_last_candle_closed == False, "Флаг is_last_candle_closed должен быть False!"
    assert ind.last_closed_close == 100.0, f"last_closed_close должен быть 100.0, получено {ind.last_closed_close}"
    assert ind.close == 145.0, "Live close должен отражать текущую цену"
    # Пивоты не должны включать свечу 60
    assert ind.last_pivot_high is None or ind.last_pivot_high.price < 140.0
    print("  [PASS] Незакрытая свеча корректно отделена от закрытой истории индикаторов.")


def test_breakout_vs_retest():
    print("\n" + "=" * 80)
    print("ТЕСТ 3: РАЗДЕЛЕНИЕ BREAKOUT CONFIRMED И RETEST CONFIRMED")
    print("=" * 80)

    atr = 2.0
    buffer_val = BREAKOUT_BUFFER_ATR_MULT * atr # 0.20
    # Пивот High = 100.0. Порог пробоя = 100.20
    ph = PivotPoint(price=100.0, formation_idx=5, confirmation_idx=7, formation_time=None, confirmation_time=None, is_high=True)

    # 1. Сценарий: Пробой на самой последней закрытой свече (bars_since = 0)
    # Свечи: 0..7 (формирование пивота), 8 (Close = 101.0 > 100.20)
    candles_bo_now = [
        {"high": 99.0, "low": 97.0, "close": 98.0} for _ in range(8)
    ]
    candles_bo_now.append({"high": 101.5, "low": 98.5, "close": 101.0}) # свеча 8: пробой!
    df_bo_now = pd.DataFrame(candles_bo_now)

    bo_long_now, _ = check_breakout_and_retest(df_bo_now, [ph], [], atr)
    print(f"Сценарий 3.1 (Пробой на последней свече): status='{bo_long_now.status}'")
    assert bo_long_now.status == "Breakout Confirmed", f"Ожидался Breakout Confirmed, получено {bo_long_now.status}"
    assert bo_long_now.bars_since_breakout == 0
    assert "Ретест ещё не сформирован" in bo_long_now.description

    # Проверка независимости trade_type и breakout_status
    ind_bo = make_dummy_indicators(close=101.0, ema50_1h=95.0, ema20_1h=98.0, poc80_1h=96.0, rsi14_1h=60.0,
                                   atr14_1h=atr, pivot_high=100.0, ema50_4h=110.0, ema20_1d=115.0, bo_long=bo_long_now)
    tt_bo, _, _, _, bo_stat, _ = classify_trade_type("LONG", ind_bo, "BEARISH")
    assert tt_bo == "Confirmed Reversal", f"trade_type должен быть 'Confirmed Reversal', получено '{tt_bo}'"
    assert bo_stat == "Breakout Confirmed", f"breakout_status должен быть 'Breakout Confirmed', получено '{bo_stat}'"
    print("  [PASS] На свече самого пробоя: trade_type='Confirmed Reversal', breakout_status='Breakout Confirmed'.")

    # 2. Сценарий: Пробой произошёл 1 свечу назад, и следующая свеча совершила РЕТЕСТ
    # Свеча 8: Пробой (Close 101.0)
    # Свеча 9: Тест сверху (Low = 100.05 в зону [99.80, 100.40], Close = 100.80 > 100.0)
    candles_retest = list(candles_bo_now)
    candles_retest.append({"high": 101.0, "low": 100.05, "close": 100.80}) # свеча 9: ретест!
    df_retest = pd.DataFrame(candles_retest)

    bo_retest, _ = check_breakout_and_retest(df_retest, [ph], [], atr)
    print(f"Сценарий 3.2 (Пробой + ретест на свече 9): status='{bo_retest.status}'")
    assert bo_retest.status == "Retest Confirmed", f"Ожидался Retest Confirmed, получено {bo_retest.status}"
    assert bo_retest.retest_candle_idx == 9
    print("  [PASS] Retest Confirmed успешно зафиксирован при возврате и удержании!")

    # 3. Сценарий: Пробой произошёл, но за 4 свечи ретеста не было (окно истекло)
    candles_no_retest = list(candles_bo_now)
    for _ in range(4): # 4 свечи улетели вверх без ретеста уровня
        candles_no_retest.append({"high": 106.0, "low": 103.0, "close": 105.0})
    df_no_retest = pd.DataFrame(candles_no_retest)

    bo_no_retest, _ = check_breakout_and_retest(df_no_retest, [ph], [], atr)
    print(f"Сценарий 3.3 (Пробой + уход без ретеста, 4 свечи): status='{bo_no_retest.status}'")
    assert bo_no_retest.status == "Breakout Confirmed", f"Ожидался Breakout Confirmed, получено {bo_no_retest.status}"
    print("  [PASS] Без ретеста статус остаётся Breakout Confirmed (не повышается).")

    # 4. Сценарий: Ложный пробой с инвалидацией (возврат глубоко под пивот)
    candles_fake = list(candles_bo_now)
    candles_fake.append({"high": 100.5, "low": 98.0, "close": 98.5}) # возврат под 100.0 - 0.2*2 = 99.6
    df_fake = pd.DataFrame(candles_fake)

    bo_fake, _ = check_breakout_and_retest(df_fake, [ph], [], atr)
    print(f"Сценарий 3.4 (Ложный пробой с возвратом под пивот): status='{bo_fake.status}'")
    assert bo_fake.status == "None"
    assert "Ложный пробой" in bo_fake.description
    print("  [PASS] Возврат под пивот корректно инвалидирует пробой.")


def test_mixed_and_consolidation():
    print("\n" + "=" * 80)
    print("ТЕСТ 4: РАЗДЕЛЕНИЕ КОНСОЛИДАЦИИ, КОНФЛИКТА 1D/4H И REVERSAL CANDIDATE")
    print("=" * 80)

    # 1. Узкая консолидация около EMA50 (флэт): Close=100.1, EMA50=100.0, RSI=50.0
    ind_flat = make_dummy_indicators(
        close=100.1, ema50_1h=100.0, ema20_1h=100.0, poc80_1h=100.0, rsi14_1h=50.0, atr14_1h=2.0,
        pivot_high=105.0, pivot_low=95.0, ema50_4h=95.0, ema20_1d=105.0 # MIXED senior trend
    )
    senior_flat = get_senior_trend(ind_flat)
    tt_flat, conf_flat, _, _, _, _ = classify_trade_type("LONG", ind_flat, senior_flat)
    print(f"Сценарий 4.1 (Узкая консолидация / флэт около EMA50): trade_type='{tt_flat}'")
    assert tt_flat == "MIXED / NO TRADE", f"Ожидался MIXED / NO TRADE, получено '{tt_flat}'"
    print("  [PASS] Флэт корректно блокируется как MIXED / NO TRADE.")

    # 2. Конфликт 1D/4H без направленного импульса
    ind_conflict = make_dummy_indicators(
        close=102.0, ema50_1h=100.0, ema20_1h=101.0, poc80_1h=101.0, rsi14_1h=46.0, atr14_1h=2.0,
        pivot_high=108.0, pivot_low=95.0, ema50_4h=98.0, ema20_1d=105.0 # MIXED
    )
    senior_conf = get_senior_trend(ind_conflict)
    tt_conf, _, _, _, _, _ = classify_trade_type("LONG", ind_conflict, senior_conf)
    print(f"Сценарий 4.2 (Конфликт 1D/4H со слабым RSI=46): trade_type='{tt_conf}'")
    assert tt_conf == "MIXED / NO TRADE"
    print("  [PASS] Конфликт 1D/4H без импульса корректно блокируется.")

    # 3. Честный Reversal Candidate: Старший тренд BEARISH, восстановление выше EMA50 и POC, импульс RSI=55, не флэт
    ind_cand = make_dummy_indicators(
        close=103.0, ema50_1h=100.0, ema20_1h=102.0, poc80_1h=101.0, rsi14_1h=55.0, atr14_1h=2.0,
        pivot_high=106.0, pivot_low=95.0, ema50_4h=110.0, ema20_1d=115.0 # BEARISH senior
    )
    senior_cand = get_senior_trend(ind_cand)
    tt_cand, conf_cand, _, diag_cand, _, th_cand = classify_trade_type("LONG", ind_cand, senior_cand)
    print(f"Сценарий 4.3 (Медвежий старший тренд + восстановление): trade_type='{tt_cand}'")
    assert tt_cand == "Reversal Candidate"
    assert diag_cand is not None
    print(f"  -> Диагностика в отчёте: {diag_cand}")
    print(f"  -> Порог пробоя: {th_cand:.2f} ₽")
    print("  [PASS] Reversal Candidate присвоен обоснованно с детальной диагностикой.")


def test_entry2_buffers_and_invalidation():
    print("\n" + "=" * 80)
    print("ТЕСТ 5: ЗАЩИТНЫЕ БУФЕРЫ ENTRY 2 И БЛОКИРОВКА УЗКОГО ДИАПАЗОНА")
    print("=" * 80)

    atr = 2.0
    min_stop_buf = MIN_DISTANCE_STOP_ATR_MULT * atr   # 0.40
    min_entry_buf = MIN_DISTANCE_ENTRY_ATR_MULT * atr # 0.40

    # 1. Нормальный широкий диапазон: Entry 1 = 100.0, Stop = 90.0 (дельта 10.0 > 0.80)
    # Уровень поддержки на 90.20 (в 0.20 от стопа) ДОЛЖЕН БЫТЬ ОТСЕЧЁН нижним буфером!
    # Допустимый диапазон: [90.0 + 0.40 = 90.40, 100.0 - 0.40 = 99.60]
    ind_wide = make_dummy_indicators(
        close=100.0, ema50_1h=85.0, ema20_1h=99.0, poc80_1h=88.0, rsi14_1h=50.0, atr14_1h=atr,
        pivot_low=90.20, val80_1h=86.0 # оба вне диапазона 90.40..99.60
    )
    e2_val, e2_desc = calculate_structural_entry_2("LONG", entry_1=100.0, anchor_stop=90.0, atr_1h=atr, indicators=ind_wide)
    print(f"Сценарий 5.1 (Широкий диапазон, уровень 90.20 отсечён буфером стопа 90.40):")
    print(f"  -> Entry 2 = {e2_val:.2f}, Опора = '{e2_desc}'")
    assert e2_val == 95.0, f"Ожидался fallback midpoint 95.0, получено {e2_val}"
    assert e2_desc == "Fallback (Midpoint)"
    print("  [PASS] Нижний буфер 0.20 ATR корректно защитил от размещения входа около стопа.")

    # 2. Узкий диапазон: Entry 1 = 100.0, Stop = 99.50 (расстояние 0.50 < 0.80 ATR)
    # Безопасное размещение Entry 2 невозможно!
    ind_narrow = make_dummy_indicators(
        close=100.0, ema50_1h=85.0, ema20_1h=99.0, poc80_1h=88.0, rsi14_1h=50.0, atr14_1h=atr
    )
    e2_narr, e2_desc_narr = calculate_structural_entry_2("LONG", entry_1=100.0, anchor_stop=99.50, atr_1h=atr, indicators=ind_narrow)
    print(f"Сценарий 5.2 (Узкий диапазон 0.50 < 0.80 ATR):")
    print(f"  -> Entry 2 = {e2_narr:.2f}, Опора = '{e2_desc_narr}'")
    assert e2_narr == 0.0 and e2_desc_narr == "INVALID_ENTRY_RANGE"
    print("  [PASS] Узкий диапазон корректно заблокирован со статусом INVALID_ENTRY_RANGE.")


def test_mirror_short_logic():
    print("\n" + "=" * 80)
    print("ТЕСТ 6: ЗЕРКАЛЬНАЯ ЛОГИКА ДЛЯ SHORT (ПОРОГИ, ПРОБОЙ, РЕТЕСТ)")
    print("=" * 80)

    atr = 2.0
    buffer_val = BREAKOUT_BUFFER_ATR_MULT * atr # 0.20
    # Pivot Low = 90.0. Порог пробоя вниз = 90.0 - 0.20 = 89.80
    pl = PivotPoint(price=90.0, formation_idx=5, confirmation_idx=7, formation_time=None, confirmation_time=None, is_high=False)

    # 1. Пробой шорта на последней свече (Close = 89.50 < 89.80)
    candles_s = [{"high": 93.0, "low": 91.0, "close": 92.0} for _ in range(8)]
    candles_s.append({"high": 91.5, "low": 89.0, "close": 89.50})
    df_s = pd.DataFrame(candles_s)

    _, bo_short = check_breakout_and_retest(df_s, [], [pl], atr)
    print(f"Сценарий 6.1 (Шорт пробой на последней свече): status='{bo_short.status}'")
    assert bo_short.status == "Breakout Confirmed"
    assert bo_short.threshold == 89.80
    assert bo_short.bars_since_breakout == 0

    # 2. Шорт ретест (тест снизу в зону [89.60, 90.20], закрытие ниже 90.0)
    candles_s_retest = list(candles_s)
    candles_s_retest.append({"high": 89.95, "low": 88.5, "close": 89.20})
    df_s_retest = pd.DataFrame(candles_s_retest)

    _, bo_short_retest = check_breakout_and_retest(df_s_retest, [], [pl], atr)
    print(f"Сценарий 6.2 (Шорт ретест снизу): status='{bo_short_retest.status}'")
    assert bo_short_retest.status == "Retest Confirmed"
    print("  [PASS] Зеркальная логика для SHORT работает безупречно.")


def main():
    print("ЗАПУСК РАСШИРЕННОГО АВТОМАТИЧЕСКОГО АУДИТА ШАГА 1 ПОСЛЕ ИСПРАВЛЕНИЙ\n")
    test_pivots_and_lookahead()
    test_unclosed_candle_filtering()
    test_breakout_vs_retest()
    test_mixed_and_consolidation()
    test_entry2_buffers_and_invalidation()
    test_mirror_short_logic()
    print("\n" + "=" * 80)
    print("ВСЕ 6 КОМПЛЕКСНЫХ ТЕСТОВ УСПЕШНО ПРОЙДЕНЫ!")
    print("=" * 80)


if __name__ == "__main__":
    main()
