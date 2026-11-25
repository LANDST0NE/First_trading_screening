"""
Модуль исторического тестирования (Chronological Walk-Forward Backtester).

Реализует:
1. Хронологический прогон свечей строго от прошлого к будущему (No Lookahead Bias).
2. Мульти-таймфреймовую синхронизацию (1H, 4H, 1D): на часе t видны только закрытые к моменту t бары 4H и 1D.
3. Точку входа в стратегию (calculate_setup) строго на закрытых барах.
4. Исполнение лимитных и защитных ордеров через ExecutionSimulator strictly на будущих барах (t+1..).
5. Локальное кэширование исторических данных для мгновенных повторных запусков.
6. Вывод подробных институциональных метрик через MetricsCalculator.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.client import TInvestClient, InstrumentInfo
from src.indicators import calculate_all_indicators
from src.strategy import calculate_setup, get_senior_trend, SetupResult
from src.backtest_config import BacktestConfig, DEFAULT_BACKTEST_CONFIG
from src.execution import (
    ExecutionSimulator,
    TradePlan,
    PlanState,
    OrderSide,
    Position
)
from src.metrics import MetricsCalculator, BacktestMetrics

console = Console()
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")


class Backtester:
    """
    Движок хронологического исторического тестирования торговой системы.
    """

    def __init__(
        self,
        client: Optional[TInvestClient] = None,
        config: Optional[BacktestConfig] = None
    ):
        self.client = client
        self.cfg = config or DEFAULT_BACKTEST_CONFIG
        self.simulator = ExecutionSimulator(config=self.cfg)
        os.makedirs(CACHE_DIR, exist_ok=True)

    def load_historical_data(
        self,
        ticker: str,
        days_1h: int = 120,
        days_4h: int = 240,
        days_1d: int = 500,
        force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Загружает исторические свечи для тикера по таймфреймам 1H, 4H, 1D.
        Использует локальный дисковый кэш parquet/csv для ускорения.
        """
        ticker = ticker.upper().strip()
        cache_file = os.path.join(CACHE_DIR, f"{ticker}_history.csv.gz")

        if not force_reload and os.path.exists(cache_file):
            try:
                df_all = pd.read_csv(cache_file)
                df_all["time"] = pd.to_datetime(df_all["time"])
                # Разделяем по таймфреймам
                res = {}
                for tf in ["1H", "4H", "1D"]:
                    sub = df_all[df_all["tf"] == tf].copy()
                    if not sub.empty:
                        sub = sub.sort_values("time").reset_index(drop=True)
                        res[tf] = sub
                if len(res) == 3:
                    return res
            except Exception as e:
                console.print(f"[dim yellow]Ошибка чтения кэша {cache_file}: {e}. Перезагрузка через API...[/dim yellow]")

        if not self.client:
            raise ValueError(f"Нет клиента API и локального кэша для {ticker}")

        inst = self.client.get_instrument_by_ticker(ticker)
        if not inst:
            raise ValueError(f"Инструмент {ticker} не найден в T-Bank API")

        # Загрузка через клиент с расширенными горизонтами
        candles_1h = self._fetch_candles_period(inst.figi, "1H", days=days_1h)
        candles_4h = self._fetch_candles_period(inst.figi, "4H", days=days_4h)
        candles_1d = self._fetch_candles_period(inst.figi, "1D", days=days_1d)

        # Сохранение в кэш
        candles_1h["tf"] = "1H"
        candles_4h["tf"] = "4H"
        candles_1d["tf"] = "1D"

        combined = pd.concat([candles_1h, candles_4h, candles_1d], ignore_index=True)
        try:
            combined.to_csv(cache_file, index=False, compression="gzip")
        except Exception:
            pass

        return {
            "1H": candles_1h.sort_values("time").reset_index(drop=True),
            "4H": candles_4h.sort_values("time").reset_index(drop=True),
            "1D": candles_1d.sort_values("time").reset_index(drop=True)
        }

    def _fetch_candles_period(self, figi: str, interval_type: str, days: int) -> pd.DataFrame:
        """Загружает свечи за указанное количество дней через клиент."""
        from datetime import datetime, timezone
        try:
            from t_tech.invest import Client, CandleInterval
            from t_tech.invest.utils import quotation_to_decimal
        except ImportError:
            from tinkoff.invest import Client, CandleInterval
            from tinkoff.invest.utils import quotation_to_decimal

        tf_map = {
            "1H": CandleInterval.CANDLE_INTERVAL_HOUR,
            "4H": CandleInterval.CANDLE_INTERVAL_4_HOUR,
            "1D": CandleInterval.CANDLE_INTERVAL_DAY,
        }
        interval = tf_map[interval_type]
        to_time = datetime.now(timezone.utc)
        from_time = to_time - timedelta(days=days)

        with Client(self.client.token) as client:
            raw = list(client.get_all_candles(
                figi=figi,
                from_=from_time,
                to=to_time,
                interval=interval
            ))

        data = []
        for c in raw:
            data.append({
                "time": pd.to_datetime(c.time),
                "open": float(quotation_to_decimal(c.open)),
                "high": float(quotation_to_decimal(c.high)),
                "low": float(quotation_to_decimal(c.low)),
                "close": float(quotation_to_decimal(c.close)),
                "volume": int(c.volume),
                "is_complete": bool(c.is_complete) if hasattr(c, "is_complete") else True
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df = df[df["is_complete"] == True].copy()
        return df

    def run_ticker(
        self,
        ticker: str,
        candles_dict: Dict[str, pd.DataFrame],
        instrument: InstrumentInfo,
        direction_pref: str = "auto",
        warmup_bars_1h: int = 100
    ) -> BacktestMetrics:
        """
        Запускает хронологический бэктест по одному тикеру.
        """
        df_1h = candles_dict["1H"]
        df_4h = candles_dict["4H"]
        df_1d = candles_dict["1D"]

        total_1h_bars = len(df_1h)
        if total_1h_bars <= warmup_bars_1h:
            raise ValueError(f"Недостаточно часовых свечей ({total_1h_bars}) для прогрева индикаторов ({warmup_bars_1h})")

        plans: List[TradePlan] = []
        active_plan: Optional[TradePlan] = None
        equity_curve: List[Dict[str, Any]] = []

        curr_capital = self.cfg.initial_capital
        equity_curve.append({
            "time": df_1h.iloc[warmup_bars_1h]["time"],
            "equity": curr_capital
        })

        # Хронологический шаг по каждой часовой свече
        for t in range(warmup_bars_1h, total_1h_bars):
            curr_candle_row = df_1h.iloc[t]
            curr_time = curr_candle_row["time"]
            curr_candle = {
                "time": curr_time,
                "open": float(curr_candle_row["open"]),
                "high": float(curr_candle_row["high"]),
                "low": float(curr_candle_row["low"]),
                "close": float(curr_candle_row["close"]),
                "volume": int(curr_candle_row["volume"])
            }

            # -------------------------------------------------------------
            # ШАГ 1: Обработка активного или ожидающего торгового плана
            # -------------------------------------------------------------
            if active_plan is not None:
                prev_pnl = active_plan.position.realized_pnl
                prev_comm = active_plan.position.commission_paid

                self.simulator.process_candle(active_plan, t, curr_candle)

                new_pnl = active_plan.position.realized_pnl
                new_comm = active_plan.position.commission_paid

                # Учитываем изменение финансового результата
                delta_rub = (new_pnl - prev_pnl) - (new_comm - prev_comm)
                curr_capital += delta_rub

                # Если план завершён (стоп, полный тейк, отмена или экспирация)
                if active_plan.state in (
                    PlanState.CLOSED_FULL_PROFIT,
                    PlanState.CLOSED_STOP_LOSS,
                    PlanState.EXPIRED_NO_FILL,
                    PlanState.CANCELLED_HIT_STOP_ON_ENTRY,
                    PlanState.CANCELLED_BEFORE_ENTRY
                ):
                    active_plan = None

            equity_curve.append({
                "time": curr_time,
                "equity": curr_capital
            })

            # -------------------------------------------------------------
            # ШАГ 2: Генерация нового сигнала (только если нет активного плана)
            # -------------------------------------------------------------
            if active_plan is None:
                # Срез данных СТРОГО до закрытой свечи t включительно (No Lookahead Bias)
                sub_1h = df_1h.iloc[: t + 1]

                # Для 4H и 1D берем только свечи, время закрытия которых <= curr_time
                sub_4h = df_4h[df_4h["time"] <= curr_time]
                sub_1d = df_1d[df_1d["time"] <= curr_time]

                # Требуется минимальная глубина для EMA и ATR
                if len(sub_1h) >= 60 and len(sub_4h) >= 30 and len(sub_1d) >= 20:
                    indicators = calculate_all_indicators({
                        "1H": sub_1h,
                        "4H": sub_4h,
                        "1D": sub_1d
                    })

                    if len(indicators) == 3:
                        senior_trend = get_senior_trend(indicators)

                        # Выбор направления сделки
                        eval_direction: Optional[str] = None
                        if direction_pref == "long":
                            eval_direction = "LONG"
                        elif direction_pref == "short":
                            eval_direction = "SHORT"
                        else:  # auto
                            if senior_trend == "BULLISH":
                                eval_direction = "LONG"
                            elif senior_trend == "BEARISH":
                                eval_direction = "SHORT"
                            else:
                                # В нейтральном/смешанном тренде оцениваем оба сетапа
                                setup_l = calculate_setup(instrument, indicators, "LONG")
                                setup_s = calculate_setup(instrument, indicators, "SHORT")
                                is_l = setup_l.status.startswith("ACTIVE")
                                is_s = setup_s.status.startswith("ACTIVE")
                                if is_l and is_s:
                                    eval_direction = "LONG" if setup_l.rr >= setup_s.rr else "SHORT"
                                elif is_l:
                                    eval_direction = "LONG"
                                elif is_s:
                                    eval_direction = "SHORT"

                        if eval_direction:
                            setup = calculate_setup(instrument, indicators, eval_direction)
                            if setup.status.startswith("ACTIVE"):
                                # Формируем новый план
                                plan_id = f"{ticker}_{t}_{eval_direction}"
                                new_plan = TradePlan(
                                    plan_id=plan_id,
                                    ticker=ticker,
                                    direction=eval_direction,
                                    step=instrument.min_price_increment,
                                    lot_size=instrument.lot,
                                    created_bar_idx=t,
                                    entry_1_price=setup.entry_1,
                                    entry_1_shares=setup.lots_entry_1 * instrument.lot,
                                    entry_2_price=setup.entry_2,
                                    entry_2_shares=setup.lots_entry_2 * instrument.lot,
                                    stop_loss_price=setup.stop,
                                    take_1_price=setup.take_1,
                                    take_1_shares=setup.lots_take_1 * instrument.lot,
                                    take_2_price=setup.take_2,
                                    take_2_shares=setup.lots_take_2 * instrument.lot,
                                    trade_type=setup.trade_type,
                                    metadata={
                                        "senior_trend": senior_trend,
                                        "rr": setup.rr,
                                        "status": setup.status
                                    }
                                )
                                plans.append(new_plan)
                                active_plan = new_plan

        # Если на конец бэктеста осталась открытая позиция, закрываем её по последнему Close
        if active_plan is not None and active_plan.position.is_open:
            last_candle = df_1h.iloc[-1]
            c_dict = {
                "time": last_candle["time"],
                "open": float(last_candle["close"]),
                "high": float(last_candle["close"]),
                "low": float(last_candle["close"]),
                "close": float(last_candle["close"]),
                "volume": 0
            }
            # Закрываем остаток позиции
            self.simulator._close_position_by_stop(active_plan, total_1h_bars - 1, c_dict, [])
            active_plan.close_reason = "END_OF_BACKTEST"

        metrics = MetricsCalculator.calculate(
            plans=plans,
            initial_capital=self.cfg.initial_capital,
            equity_curve=equity_curve
        )
        return metrics


def print_backtest_report(metrics: BacktestMetrics, ticker: str = "PORTFOLIO"):
    """
    Печатает профессиональный форматированный отчёт бэктеста в консоль.
    """
    console.print(f"\n[bold green]====================================================================[/bold green]")
    console.print(f"[bold cyan] РЕЗУЛЬТАТЫ ИСТОРИЧЕСКОГО ТЕСТИРОВАНИЯ (БЭКТЕСТ): [yellow]{ticker}[/yellow][/bold cyan]")
    console.print(f"[bold green]====================================================================[/bold green]\n")

    # 1. Основные финансовые показатели
    t_fin = Table(title="Финансовые показатели", header_style="bold magenta")
    t_fin.add_column("Метрика", style="cyan")
    t_fin.add_column("Значение", justify="right", style="bold")

    pnl_color = "green" if metrics.net_profit >= 0 else "red"
    t_fin.add_row("Стартовый депозит", f"{metrics.initial_capital:,.2f} ₽")
    t_fin.add_row("Итоговый депозит", f"{metrics.final_capital:,.2f} ₽")
    t_fin.add_row("Чистая прибыль (Net PnL)", f"[{pnl_color}]{metrics.net_profit:+,.2f} ₽ ({metrics.net_profit_pct:+.2f}%)[/{pnl_color}]")
    t_fin.add_row("Валовая прибыль (Gross Profit)", f"+{metrics.gross_profit:,.2f} ₽")
    t_fin.add_row("Валовый убыток (Gross Loss)", f"-{metrics.gross_loss:,.2f} ₽")
    t_fin.add_row("Уплаченные комиссии", f"{metrics.total_commissions:,.2f} ₽")
    t_fin.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
    t_fin.add_row("Максимальная просадка (Max DD)", f"[red]{metrics.max_drawdown_rub:,.2f} ₽ ({metrics.max_drawdown_pct:.2f}%)[/red]")
    t_fin.add_row("Recovery Factor", f"{metrics.recovery_factor:.2f}")
    console.print(t_fin)

    # 2. Торговая статистика
    t_stat = Table(title="Торговая статистика и исполнение", header_style="bold blue")
    t_stat.add_column("Показатель", style="cyan")
    t_stat.add_column("Количество / Значение", justify="right", style="bold")

    t_stat.add_row("Всего торговых сигналов", str(metrics.total_plans))
    t_stat.add_row("Исполненных сделок", str(metrics.executed_plans))
    t_stat.add_row("Истёкших без входа (TTL Expired)", f"{metrics.expired_plans} ({metrics.expired_plans/max(1, metrics.total_plans)*100:.1f}%)")
    t_stat.add_row("Отменённых до входа (Cancelled)", str(metrics.cancelled_plans))
    t_stat.add_row("Прибыльных сделок (Wins)", f"[green]{metrics.winning_plans} ({metrics.win_rate_pct:.1f}%)[/green]")
    t_stat.add_row("Убыточных сделок (Losses)", f"[red]{metrics.losing_plans} ({metrics.loss_rate_pct:.1f}%)[/red]")
    t_stat.add_row("Средняя прибыль (Avg Win)", f"+{metrics.avg_win_rub:,.2f} ₽")
    t_stat.add_row("Средний убыток (Avg Loss)", f"-{metrics.avg_loss_rub:,.2f} ₽")
    t_stat.add_row("Payoff Ratio (Win/Loss)", f"{metrics.payoff_ratio:.2f}")
    t_stat.add_row("Матожидание на сделку (Expectancy)", f"[{pnl_color}]{metrics.expectancy_rub:+,.2f} ₽[/{pnl_color}]")
    t_stat.add_row("Входы: только Entry 1 (50%)", f"{metrics.entry1_only_count} ({metrics.entry1_only_pct:.1f}%)")
    t_stat.add_row("Входы: оба Entry 1 + Entry 2 (100%)", f"{metrics.both_entries_count} ({metrics.both_entries_pct:.1f}%)")
    t_stat.add_row("Выходы: Stop-Loss", f"[red]{metrics.stop_loss_exits}[/red]")
    t_stat.add_row("Выходы: Take Profit 1 (частичные)", f"[yellow]{metrics.tp1_only_exits}[/yellow]")
    t_stat.add_row("Выходы: Take Profit 2 (полный профит)", f"[green]{metrics.tp2_full_exits}[/green]")
    t_stat.add_row("Среднее время в сделке", f"{metrics.avg_bars_in_trade:.1f} часовых свечей")
    console.print(t_stat)

    # 3. Детализация по типам сетапов (Trade Type Breakdown)
    if metrics.by_trade_type:
        t_types = Table(title="Разбивка по типам сделок (Trade Types)", header_style="bold yellow")
        t_types.add_column("Тип сделки", style="cyan")
        t_types.add_column("Сигналы", justify="right")
        t_types.add_column("Сделки", justify="right")
        t_types.add_column("Win Rate", justify="right")
        t_types.add_column("Profit Factor", justify="right")
        t_types.add_column("Чистый PnL", justify="right")
        t_types.add_column("Матожидание", justify="right")

        for name, sm in metrics.by_trade_type.items():
            c = "green" if sm.net_profit >= 0 else "red"
            t_types.add_row(
                name,
                str(sm.total_plans),
                str(sm.executed_plans),
                f"{sm.win_rate_pct:.1f}%",
                f"{sm.profit_factor:.2f}" if sm.profit_factor < 100 else "N/A",
                f"[{c}]{sm.net_profit:+,.2f} ₽[/{c}]",
                f"[{c}]{sm.expectancy:+,.2f} ₽[/{c}]"
            )
        console.print(t_types)
