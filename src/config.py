import os
from dotenv import load_dotenv

load_dotenv()

INVEST_TOKEN = os.getenv("INVEST_TOKEN", "").strip()

# Базовый список ликвидных акций Мосбиржи
DEFAULT_TICKERS = [
    "YDEX", "CHMF", "MTLR", "NVTK", "GAZP", "SBER", "GMKN", "T",
    "LKOH", "ROSN", "AFLT", "ALRS", "MAGN", "NLMK", "PLZL", "MOEX"
]

# FIGI индекса IMOEX
IMOEX_FIGI = "BBG00KDWPPW2"

# Торговые и риск-параметры
COMMISSION = 0.0005  # 0.05% за круг (открытие + закрытие)
DAILY_RISK_LIMIT = 2000.0  # Базовый дневной лимит убытка в рублях
RISK_STANDARD = 1000.0     # Риск при 1.8 <= RR < 2.5
RISK_STRONG = 1750.0       # Риск при RR >= 2.5

MIN_RR_THRESHOLD = 1.8
STRONG_RR_THRESHOLD = 2.5

# Структурный анализ: параметры пробоя, ретеста и буферов Entry 2
BREAKOUT_BUFFER_ATR_MULT = 0.10      # Буфер пробоя (0.10 * ATR_1H)
MIN_DISTANCE_STOP_ATR_MULT = 0.20    # Минимальное расстояние Entry 2 до Anchor Stop (0.20 * ATR_1H)
MIN_DISTANCE_ENTRY_ATR_MULT = 0.20   # Минимальное расстояние Entry 2 до Entry 1 (0.20 * ATR_1H)

RETEST_WINDOW_BARS = 3               # Окно наблюдения ретеста (3 закрытые часовые свечи после пробоя)
RETEST_ZONE_PENETRATION_MULT = 0.10  # Допустимое проникновение за уровень при ретесте (0.10 * ATR_1H)

# Минимальные требования к числу свечей
MIN_CANDLES_1H = 150
MIN_CANDLES_4H = 120
MIN_CANDLES_1D = 100
