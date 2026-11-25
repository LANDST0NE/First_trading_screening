"""
Модуль расчёта метрик эффективности и торговой статистики (Performance Metrics).

Принимает результаты бэктеста (завершённые торговые планы TradePlan, сделки Trade и кривую капитала Equity),
рассчитывает классические институциональные метрики и формирует детализированные срезы по:
- Типам сигналов (Trade Type)
- Тикерам (Instruments)
- Качеству исполнения (1-Entry vs 2-Entry, % экспираций, среднее время в позиции)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import numpy as np

from src.execution import TradePlan, PlanState, Trade


@dataclass
class SegmentMetrics:
    """Метрики для отдельного сегмента (по типу сигнала или тикеру)."""
    name: str
    total_plans: int = 0
    executed_plans: int = 0
    winning_plans: int = 0
    losing_plans: int = 0
    breakeven_plans: int = 0
    win_rate_pct: float = 0.0
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    commissions: float = 0.0


@dataclass
class BacktestMetrics:
    """Комплексный отчёт по результатам исторического тестирования."""
    # 1. Общие финансовые результаты
    initial_capital: float = 0.0
    final_capital: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_commissions: float = 0.0

    # 2. Относительные показатели эффективности
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    max_drawdown_rub: float = 0.0
    max_drawdown_pct: float = 0.0

    # 3. Торговая статистика (по планам/позициям)
    total_plans: int = 0
    executed_plans: int = 0
    expired_plans: int = 0
    cancelled_plans: int = 0
    winning_plans: int = 0
    losing_plans: int = 0
    breakeven_plans: int = 0

    win_rate_pct: float = 0.0
    loss_rate_pct: float = 0.0
    avg_win_rub: float = 0.0
    avg_loss_rub: float = 0.0
    payoff_ratio: float = 0.0
    expectancy_rub: float = 0.0

    # 4. Анализ качества исполнения (Execution Stats)
    entry1_only_count: int = 0
    entry1_only_pct: float = 0.0
    both_entries_count: int = 0
    both_entries_pct: float = 0.0

    tp1_only_exits: int = 0
    tp2_full_exits: int = 0
    stop_loss_exits: int = 0

    # 5. Временные характеристики
    avg_bars_in_trade: float = 0.0

    # 6. Сегментация (Breakdowns)
    by_trade_type: Dict[str, SegmentMetrics] = field(default_factory=dict)
    by_ticker: Dict[str, SegmentMetrics] = field(default_factory=dict)


class MetricsCalculator:
    """
    Калькулятор метрик бэктеста.
    """

    @staticmethod
    def calculate(
        plans: List[TradePlan],
        initial_capital: float,
        equity_curve: Optional[List[Dict[str, Any]]] = None
    ) -> BacktestMetrics:
        """
        Вычисляет полный набор метрик по завершённым торговым планам и кривой капитала.
        """
        m = BacktestMetrics(initial_capital=initial_capital)
        m.total_plans = len(plans)

        if not plans:
            m.final_capital = initial_capital
            return m

        # Разделение планов по статусам
        executed_plans: List[TradePlan] = []
        for p in plans:
            if p.state in (
                PlanState.CLOSED_FULL_PROFIT,
                PlanState.CLOSED_STOP_LOSS,
                PlanState.ACTIVE_PARTIAL_TP1,  # Если бэктест завершился, а часть позиции еще открыта
                PlanState.ACTIVE_E1,
                PlanState.ACTIVE_E1_E2
            ) or (p.position.trades and not p.position.is_open):
                executed_plans.append(p)
            elif p.state == PlanState.EXPIRED_NO_FILL:
                m.expired_plans += 1
            elif p.state in (PlanState.CANCELLED_HIT_STOP_ON_ENTRY, PlanState.CANCELLED_BEFORE_ENTRY):
                m.cancelled_plans += 1

        m.executed_plans = len(executed_plans)

        # Подсчёт PnL и сделок
        profits: List[float] = []
        losses: List[float] = []
        holding_bars: List[int] = []

        total_net_pnl = 0.0
        total_comm = 0.0

        for p in executed_plans:
            pos = p.position
            pnl = pos.realized_pnl
            comm = pos.commission_paid

            total_net_pnl += pnl
            total_comm += comm

            # Длительность удержания позиции (в барах)
            if pos.opened_bar_idx is not None and pos.closed_bar_idx is not None:
                bars = max(1, pos.closed_bar_idx - pos.opened_bar_idx)
                holding_bars.append(bars)

            # Анализ исполнения входов
            e1_filled = any(t.reason == "ENTRY_1" for t in pos.trades)
            e2_filled = any(t.reason == "ENTRY_2" for t in pos.trades)
            if e1_filled and e2_filled:
                m.both_entries_count += 1
            elif e1_filled:
                m.entry1_only_count += 1

            # Анализ выходов
            if p.state == PlanState.CLOSED_FULL_PROFIT or p.close_reason == "TAKE_2":
                m.tp2_full_exits += 1
            elif p.state == PlanState.CLOSED_STOP_LOSS or p.close_reason == "STOP_LOSS":
                m.stop_loss_exits += 1
            elif p.close_reason == "TAKE_1_FULL" or any(t.reason == "TAKE_1" for t in pos.trades):
                m.tp1_only_exits += 1

            if pnl > 0.01:
                m.winning_plans += 1
                profits.append(pnl)
            elif pnl < -0.01:
                m.losing_plans += 1
                losses.append(pnl)
            else:
                m.breakeven_plans += 1

        # Общие финансовые метрики
        m.net_profit = total_net_pnl
        m.final_capital = initial_capital + total_net_pnl
        m.net_profit_pct = (total_net_pnl / initial_capital * 100.0) if initial_capital > 0 else 0.0
        m.total_commissions = total_comm
        m.gross_profit = sum(profits)
        m.gross_loss = abs(sum(losses))

        # Win Rate и Payoff
        if m.executed_plans > 0:
            m.win_rate_pct = (m.winning_plans / m.executed_plans) * 100.0
            m.loss_rate_pct = (m.losing_plans / m.executed_plans) * 100.0
            m.entry1_only_pct = (m.entry1_only_count / m.executed_plans) * 100.0
            m.both_entries_pct = (m.both_entries_count / m.executed_plans) * 100.0

        if profits:
            m.avg_win_rub = sum(profits) / len(profits)
        if losses:
            m.avg_loss_rub = abs(sum(losses) / len(losses))

        if m.avg_loss_rub > 0:
            m.payoff_ratio = m.avg_win_rub / m.avg_loss_rub
        else:
            m.payoff_ratio = float("inf") if m.avg_win_rub > 0 else 0.0

        if m.gross_loss > 0:
            m.profit_factor = m.gross_profit / m.gross_loss
        else:
            m.profit_factor = float("inf") if m.gross_profit > 0 else 0.0

        # Математическое ожидание на одну сделку (в рублях)
        wr = m.win_rate_pct / 100.0
        lr = m.loss_rate_pct / 100.0
        m.expectancy_rub = (wr * m.avg_win_rub) - (lr * m.avg_loss_rub)

        if holding_bars:
            m.avg_bars_in_trade = sum(holding_bars) / len(holding_bars)

        # Расчёт максимальной просадки (Drawdown) по кривой капитала
        if equity_curve and len(equity_curve) > 1:
            peak = initial_capital
            max_dd_rub = 0.0
            max_dd_pct = 0.0
            for pt in equity_curve:
                eq = pt["equity"]
                if eq > peak:
                    peak = eq
                dd_rub = peak - eq
                dd_pct = (dd_rub / peak * 100.0) if peak > 0 else 0.0
                if dd_rub > max_dd_rub:
                    max_dd_rub = dd_rub
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            m.max_drawdown_rub = max_dd_rub
            m.max_drawdown_pct = max_dd_pct
        else:
            # Если кривой нет, оцениваем кумулятивно по сделкам
            peak = initial_capital
            curr = initial_capital
            max_dd_rub = 0.0
            max_dd_pct = 0.0
            for p in executed_plans:
                curr += p.position.realized_pnl
                if curr > peak:
                    peak = curr
                dd_rub = peak - curr
                dd_pct = (dd_rub / peak * 100.0) if peak > 0 else 0.0
                if dd_rub > max_dd_rub:
                    max_dd_rub = dd_rub
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            m.max_drawdown_rub = max_dd_rub
            m.max_drawdown_pct = max_dd_pct

        if m.max_drawdown_rub > 0:
            m.recovery_factor = m.net_profit / m.max_drawdown_rub
        else:
            m.recovery_factor = float("inf") if m.net_profit > 0 else 0.0

        # Сегментация по Trade Type
        m.by_trade_type = MetricsCalculator._segment_by(plans, key_fn=lambda p: p.trade_type)

        # Сегментация по Ticker
        m.by_ticker = MetricsCalculator._segment_by(plans, key_fn=lambda p: p.ticker)

        return m

    @staticmethod
    def _segment_by(plans: List[TradePlan], key_fn) -> Dict[str, SegmentMetrics]:
        """Группирует планы по ключу и считает локальные метрики."""
        groups: Dict[str, List[TradePlan]] = {}
        for p in plans:
            k = key_fn(p) or "Unknown"
            groups.setdefault(k, []).append(p)

        result: Dict[str, SegmentMetrics] = {}
        for name, group_plans in groups.items():
            sm = SegmentMetrics(name=name)
            sm.total_plans = len(group_plans)

            exec_plans = [p for p in group_plans if p.position.trades and not p.position.is_open]
            sm.executed_plans = len(exec_plans)

            profits: List[float] = []
            losses: List[float] = []

            for p in exec_plans:
                pnl = p.position.realized_pnl
                sm.net_profit += pnl
                sm.commissions += p.position.commission_paid

                if pnl > 0.01:
                    sm.winning_plans += 1
                    profits.append(pnl)
                elif pnl < -0.01:
                    sm.losing_plans += 1
                    losses.append(pnl)
                else:
                    sm.breakeven_plans += 1

            sm.gross_profit = sum(profits)
            sm.gross_loss = abs(sum(losses))

            if sm.executed_plans > 0:
                sm.win_rate_pct = (sm.winning_plans / sm.executed_plans) * 100.0

            if profits:
                sm.avg_win = sum(profits) / len(profits)
            if losses:
                sm.avg_loss = abs(sum(losses) / len(losses))

            if sm.avg_loss > 0:
                sm.payoff_ratio = sm.avg_win / sm.avg_loss
            else:
                sm.payoff_ratio = float("inf") if sm.avg_win > 0 else 0.0

            if sm.gross_loss > 0:
                sm.profit_factor = sm.gross_profit / sm.gross_loss
            else:
                sm.profit_factor = float("inf") if sm.gross_profit > 0 else 0.0

            wr = sm.win_rate_pct / 100.0
            lr = (sm.losing_plans / sm.executed_plans) if sm.executed_plans > 0 else 0.0
            sm.expectancy = (wr * sm.avg_win) - (lr * sm.avg_loss)

            result[name] = sm

        return result
