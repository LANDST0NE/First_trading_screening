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

# Минимальные требования к числу свечей
MIN_CANDLES_1H = 150
MIN_CANDLES_4H = 120
MIN_CANDLES_1D = 100
