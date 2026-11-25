import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd

# Поддержка обоих пакетов SDK по требованию ТЗ
try:
    from t_tech.invest import Client, CandleInterval
    from t_tech.invest.utils import quotation_to_decimal, now
except ImportError:
    from tinkoff.invest import Client, CandleInterval
    from tinkoff.invest.utils import quotation_to_decimal, now

from src.config import INVEST_TOKEN, IMOEX_FIGI, MIN_CANDLES_1H, MIN_CANDLES_4H, MIN_CANDLES_1D


@dataclass
class InstrumentInfo:
    ticker: str
    figi: str
    uid: str
    name: str
    lot: int
    min_price_increment: float
    short_enabled_flag: bool
    currency: str = "rub"


class TInvestClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or INVEST_TOKEN
        if not self.token or self.token == "t.your_token_here":
            raise ValueError(
                "INVEST_TOKEN не найден в .env или не заполнен! "
                "Укажите валидный токен Т-Инвестиций в файле .env."
            )
        self._shares_cache: Optional[List[Any]] = None

    def _get_all_shares(self, client: Client) -> List[Any]:
        if self._shares_cache is None:
            res = client.instruments.shares()
            self._shares_cache = res.instruments
        return self._shares_cache

    def get_instrument_by_ticker(self, ticker: str) -> Optional[InstrumentInfo]:
        """Находит акцию Мосбиржи (класс TQBR) по тикеру."""
        ticker = ticker.upper().strip()
        with Client(self.token) as client:
            shares = self._get_all_shares(client)
            match = next(
                (s for s in shares if s.ticker.upper() == ticker and getattr(s, "class_code", "") == "TQBR"),
                None
            )
            # Если по TQBR напрямую не найдено, пробуем найти среди любых акций с таким тикером
            if not match:
                match = next((s for s in shares if s.ticker.upper() == ticker), None)

            if not match:
                return None

            step = float(quotation_to_decimal(match.min_price_increment))
            return InstrumentInfo(
                ticker=match.ticker,
                figi=match.figi,
                uid=getattr(match, "uid", match.figi),
                name=match.name,
                lot=int(match.lot),
                min_price_increment=step,
                short_enabled_flag=bool(match.short_enabled_flag),
                currency=getattr(match, "currency", "rub")
            )

    def get_candles(self, figi: str, interval_type: str) -> pd.DataFrame:
        """
        Загружает исторические свечи через get_all_candles для заданного таймфрейма.
        Поддерживаемые interval_type: '1H', '4H', '1D'.
        """
        current_time = now()

        if interval_type == "1H":
            interval = CandleInterval.CANDLE_INTERVAL_HOUR
            min_count = MIN_CANDLES_1H
            # Мосбиржа торгуется ~14 часов в день, 5 дней в неделю (~70 часов в неделю).
            # 45 дней с запасом дают 200+ свечей.
            from_time = current_time - datetime.timedelta(days=45)
        elif interval_type == "4H":
            interval = CandleInterval.CANDLE_INTERVAL_4_HOUR
            min_count = MIN_CANDLES_4H
            from_time = current_time - datetime.timedelta(days=150)
        elif interval_type == "1D":
            interval = CandleInterval.CANDLE_INTERVAL_DAY
            min_count = MIN_CANDLES_1D
            from_time = current_time - datetime.timedelta(days=365)
        else:
            raise ValueError(f"Неизвестный таймфрейм: {interval_type}")

        with Client(self.token) as client:
            raw_candles = list(client.get_all_candles(
                figi=figi,
                from_=from_time,
                to=current_time,
                interval=interval
            ))

        if not raw_candles:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        rows = []
        for c in raw_candles:
            rows.append({
                "time": c.time,
                "open": float(quotation_to_decimal(c.open)),
                "high": float(quotation_to_decimal(c.high)),
                "low": float(quotation_to_decimal(c.low)),
                "close": float(quotation_to_decimal(c.close)),
                "volume": int(c.volume),
                "is_complete": bool(getattr(c, "is_complete", True))
            })

        df = pd.DataFrame(rows)
        df.sort_values(by="time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_multi_tf_candles(self, figi: str) -> Dict[str, pd.DataFrame]:
        """Загружает свечи по трём таймфреймам: 1H, 4H, 1D."""
        return {
            "1H": self.get_candles(figi, "1H"),
            "4H": self.get_candles(figi, "4H"),
            "1D": self.get_candles(figi, "1D")
        }

    def get_imoex_context(self) -> Dict[str, Any]:
        """Получает текущую динамику индекса МосБиржи (IMOEX)."""
        try:
            with Client(self.token) as client:
                candles = list(client.get_all_candles(
                    figi=IMOEX_FIGI,
                    from_=now() - datetime.timedelta(days=10),
                    to=now(),
                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                ))
            if len(candles) >= 2:
                prev_close = float(quotation_to_decimal(candles[-2].close))
                last_close = float(quotation_to_decimal(candles[-1].close))
                change_pct = ((last_close - prev_close) / prev_close) * 100.0
                return {
                    "price": last_close,
                    "change_pct": change_pct,
                    "trend": "BULLISH" if change_pct > 0.3 else ("BEARISH" if change_pct < -0.3 else "NEUTRAL")
                }
            elif len(candles) == 1:
                last_close = float(quotation_to_decimal(candles[0].close))
                return {"price": last_close, "change_pct": 0.0, "trend": "NEUTRAL"}
        except Exception:
            pass
        return {"price": 0.0, "change_pct": 0.0, "trend": "UNKNOWN"}
