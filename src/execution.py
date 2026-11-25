"""
Модуль исполнения торговых заявок (Execution Simulator).
Полная изоляция от торговой стратегии: оперирует исключительно заявками, позициями,
ценовыми свечами и консервативными правилами биржевого стакана.

Реализует:
- Консервативную проверку проникновения за уровень (Penetration Rule).
- Учёт ценовых разрывов (Gaps) и отдельное моделирование проскальзывания.
- Разрешение конфликтов внутри одной OHLCV-свечи (Stop vs Entry, Entry 2 vs TP1, Stop vs TP).
- Двухэтапный набор позиции (Entry 1 / Entry 2) с динамическим средневзвешенным входом.
- Частичную фиксацию прибыли (Take 1 и Take 2) и отмену добора при взятии первой цели.
- Контроль времени жизни лимитных заявок (TTL).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

from src.backtest_config import BacktestConfig, DEFAULT_BACKTEST_CONFIG


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class PlanState(str, Enum):
    WAITING_ENTRY = "WAITING_ENTRY"                     # Ожидание первого входа
    ACTIVE_E1 = "ACTIVE_E1"                             # Исполнен Entry 1 (50%), Entry 2 ожидает
    ACTIVE_E1_E2 = "ACTIVE_E1_E2"                       # Исполнены оба входа (100% позиции)
    ACTIVE_PARTIAL_TP1 = "ACTIVE_PARTIAL_TP1"           # Взят TP1, остаток удерживается до TP2
    CLOSED_FULL_PROFIT = "CLOSED_FULL_PROFIT"           # Успешно закрыт по TP2 (полный профит)
    CLOSED_STOP_LOSS = "CLOSED_STOP_LOSS"               # Закрыт по Stop-Loss
    CLOSED_END_OF_BACKTEST = "CLOSED_END_OF_BACKTEST"   # Закрыт по рыночной цене в конце истории
    EXPIRED_NO_FILL = "EXPIRED_NO_FILL"                 # Истёк TTL без исполнения входов
    CANCELLED_HIT_STOP_ON_ENTRY = "CANCELLED_HIT_STOP_ON_ENTRY"  # На свече входа был задет и Стоп (отмена)
    CANCELLED_BEFORE_ENTRY = "CANCELLED_BEFORE_ENTRY"   # Стоп пробит до первого входа


@dataclass
class Trade:
    trade_id: str
    ticker: str
    side: OrderSide
    shares: int
    price: float
    commission: float
    timestamp: Any
    reason: str                        # "ENTRY_1", "ENTRY_2", "TAKE_1", "TAKE_2", "STOP_LOSS"
    slippage: float = 0.0


@dataclass
class Position:
    ticker: str
    side: str                          # "LONG" или "SHORT"
    total_shares: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0          # Валовый PnL (Gross PnL) без вычета комиссий
    commission_paid: float = 0.0       # Суммарная уплаченная комиссия (входы + выходы)
    is_open: bool = False
    trades: List[Trade] = field(default_factory=list)
    opened_bar_idx: Optional[int] = None
    closed_bar_idx: Optional[int] = None

    @property
    def net_pnl(self) -> float:
        """Чистый PnL с учётом всех уплаченных комиссий."""
        return self.realized_pnl - self.commission_paid


@dataclass
class TradePlan:
    plan_id: str
    ticker: str
    direction: str                     # "LONG" или "SHORT"
    step: float                        # min_price_increment
    lot_size: int
    created_bar_idx: int               # Индекс закрытой свечи сигнала (t)

    # Параметры входов
    entry_1_price: float
    entry_1_shares: int
    entry_2_price: float
    entry_2_shares: int

    # Параметры выходов
    stop_loss_price: float
    take_1_price: float
    take_1_shares: int
    take_2_price: float
    take_2_shares: int

    # Метаданные стратегии (для аналитики)
    trade_type: str = "Trend Following"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Текущее состояние
    state: PlanState = PlanState.WAITING_ENTRY
    position: Position = field(default_factory=lambda: Position(ticker="", side=""))
    close_reason: Optional[str] = None
    cancel_reason: Optional[str] = None

    def __post_init__(self):
        self.position.ticker = self.ticker
        self.position.side = self.direction


class ExecutionSimulator:
    """
    Симулятор исполнения ордеров для одного торгового плана.
    Обрабатывает каждую последующую свечу хронологически и детерминированно.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg = config or DEFAULT_BACKTEST_CONFIG

    def process_candle(
        self,
        plan: TradePlan,
        candle_idx: int,
        candle: Dict[str, Any]
    ) -> List[str]:
        """
        Обрабатывает ценовую свечу (bar) для заданного торгового плана.
        Возвращает список зафиксированных событий (логов исполнения).
        """
        events: List[str] = []

        # Если план уже завершён или отменён — ничего не делаем
        if plan.state in (
            PlanState.CLOSED_FULL_PROFIT,
            PlanState.CLOSED_STOP_LOSS,
            PlanState.CLOSED_END_OF_BACKTEST,
            PlanState.EXPIRED_NO_FILL,
            PlanState.CANCELLED_HIT_STOP_ON_ENTRY,
            PlanState.CANCELLED_BEFORE_ENTRY
        ):
            return events

        # Строгое правило: заявки активируются не ранее следующей свечи после сигнала
        if candle_idx <= plan.created_bar_idx:
            return events

        c_time = candle.get("time")
        c_open = float(candle["open"])
        c_high = float(candle["high"])
        c_low = float(candle["low"])
        c_close = float(candle["close"])
        step = plan.step

        # -----------------------------------------------------------------
        # 1. СОСТОЯНИЕ: WAITING_ENTRY (Ожидание первого входа)
        # -----------------------------------------------------------------
        if plan.state == PlanState.WAITING_ENTRY:
            # А. Проверка TTL (срок действия лимитных заявок)
            if candle_idx > plan.created_bar_idx + self.cfg.limit_expiry_bars:
                plan.state = PlanState.EXPIRED_NO_FILL
                plan.cancel_reason = f"Истёк TTL ({self.cfg.limit_expiry_bars} бара) без исполнения входа"
                events.append(f"[{c_time}] {plan.ticker} план ИСТЁК (TTL {self.cfg.limit_expiry_bars} б.)")
                return events

            # Б. Проверка достижения Stop-Loss до входа
            stop_reachable = self._is_price_reached(plan.direction, is_stop=True, price=plan.stop_loss_price, candle=candle)
            entry1_reachable = self._is_entry_reached(plan.direction, plan.entry_1_price, candle, step)

            # Консервативное правило: если на одной свече достижимы и Вход, и Стоп -> Стоп сработал первым!
            if stop_reachable and entry1_reachable:
                if self.cfg.assume_stop_before_entry_on_same_bar:
                    plan.state = PlanState.CANCELLED_HIT_STOP_ON_ENTRY
                    plan.cancel_reason = "На свече входа был одновременно задет Stop-Loss. Сделка отменена."
                    events.append(f"[{c_time}] {plan.ticker} ОТМЕНА: одновременное касание Entry 1 и Stop на свече")
                    return events

            # Если стоп достигнут, а вход нет — сетап инвалидирован
            if stop_reachable and not entry1_reachable:
                plan.state = PlanState.CANCELLED_BEFORE_ENTRY
                plan.cancel_reason = "Цена выбила уровень Stop-Loss до срабатывания лимитного входа"
                events.append(f"[{c_time}] {plan.ticker} ОТМЕНА: Stop-Loss пробит до входа в позицию")
                return events

            # В. Исполнение Entry 1
            if entry1_reachable:
                fill_price, slip = self._get_entry_fill_price(plan.direction, plan.entry_1_price, candle)
                comm = fill_price * plan.entry_1_shares * self.cfg.commission_entry_rate
                trade1 = Trade(
                    trade_id=f"{plan.plan_id}_E1",
                    ticker=plan.ticker,
                    side=OrderSide.BUY if plan.direction == "LONG" else OrderSide.SELL,
                    shares=plan.entry_1_shares,
                    price=fill_price,
                    commission=comm,
                    timestamp=c_time,
                    reason="ENTRY_1",
                    slippage=slip
                )
                self._apply_entry_trade(plan.position, trade1, candle_idx)
                plan.state = PlanState.ACTIVE_E1
                events.append(f"[{c_time}] {plan.ticker} ВХОД 1 исполнен: {trade1.shares} шт. по {fill_price:.3f} ₽ (ком.: {comm:.2f} ₽)")

                # Проверяем, исполнился ли Entry 2 на этой же свече
                entry2_reachable = self._is_entry_reached(plan.direction, plan.entry_2_price, candle, step)
                tp1_reachable = self._is_price_reached(plan.direction, is_stop=False, price=plan.take_1_price, candle=candle)

                # Консервативное правило: если на свече достигнуты и Entry 2, и TP1 -> TP1 взят первым, Entry 2 отменяется!
                if entry2_reachable and tp1_reachable and self.cfg.assume_tp_before_entry2_on_same_bar:
                    events.append(f"[{c_time}] {plan.ticker} Конфликт Entry 2 и TP1 на одной свече: Entry 2 отменён в пользу TP1")
                    entry2_reachable = False

                if entry2_reachable:
                    fill2_price, slip2 = self._get_entry_fill_price(plan.direction, plan.entry_2_price, candle)
                    comm2 = fill2_price * plan.entry_2_shares * self.cfg.commission_entry_rate
                    trade2 = Trade(
                        trade_id=f"{plan.plan_id}_E2",
                        ticker=plan.ticker,
                        side=OrderSide.BUY if plan.direction == "LONG" else OrderSide.SELL,
                        shares=plan.entry_2_shares,
                        price=fill2_price,
                        commission=comm2,
                        timestamp=c_time,
                        reason="ENTRY_2",
                        slippage=slip2
                    )
                    self._apply_entry_trade(plan.position, trade2, candle_idx)
                    plan.state = PlanState.ACTIVE_E1_E2
                    events.append(f"[{c_time}] {plan.ticker} ВХОД 2 исполнен на той же свече: {trade2.shares} шт. по {fill2_price:.3f} ₽")

                # Если TP1 был достигнут на той же свече (даже если Entry 2 не исполнился)
                if tp1_reachable:
                    self._execute_take_profit_1(plan, candle_idx, candle, events)

                return events

        # -----------------------------------------------------------------
        # 2. СОСТОЯНИЕ: ACTIVE_E1 (Исполнен только Вход 1, Вход 2 активен)
        # -----------------------------------------------------------------
        elif plan.state == PlanState.ACTIVE_E1:
            # А. Проверка Stop-Loss и TP1
            stop_reachable = self._is_price_reached(plan.direction, is_stop=True, price=plan.stop_loss_price, candle=candle)
            tp1_reachable = self._is_price_reached(plan.direction, is_stop=False, price=plan.take_1_price, candle=candle)

            # Консервативное правило: Стоп vs Тейк на одной свече -> СТОП сработал первым!
            if stop_reachable and tp1_reachable and self.cfg.assume_stop_before_tp_on_same_bar:
                self._close_position_by_stop(plan, candle_idx, candle, events)
                return events

            if stop_reachable:
                self._close_position_by_stop(plan, candle_idx, candle, events)
                return events

            # Б. Проверка конфликта Entry 2 и TP1 на этой свече
            entry2_reachable = self._is_entry_reached(plan.direction, plan.entry_2_price, candle, step)

            # Если истёк срок жизни Entry 2
            if candle_idx > plan.created_bar_idx + self.cfg.limit_expiry_bars:
                if entry2_reachable:
                    events.append(f"[{c_time}] {plan.ticker} Лимит Entry 2 отменён по истечении TTL")
                entry2_reachable = False

            if entry2_reachable and tp1_reachable and self.cfg.assume_tp_before_entry2_on_same_bar:
                events.append(f"[{c_time}] {plan.ticker} Entry 2 отменён: на свече зафиксирован Take 1")
                entry2_reachable = False

            # В. Исполнение добора Entry 2
            if entry2_reachable:
                fill2_price, slip2 = self._get_entry_fill_price(plan.direction, plan.entry_2_price, candle)
                comm2 = fill2_price * plan.entry_2_shares * self.cfg.commission_entry_rate
                trade2 = Trade(
                    trade_id=f"{plan.plan_id}_E2",
                    ticker=plan.ticker,
                    side=OrderSide.BUY if plan.direction == "LONG" else OrderSide.SELL,
                    shares=plan.entry_2_shares,
                    price=fill2_price,
                    commission=comm2,
                    timestamp=c_time,
                    reason="ENTRY_2",
                    slippage=slip2
                )
                self._apply_entry_trade(plan.position, trade2, candle_idx)
                plan.state = PlanState.ACTIVE_E1_E2
                events.append(f"[{c_time}] {plan.ticker} ВХОД 2 добран: {trade2.shares} шт. по {fill2_price:.3f} ₽ (Средняя: {plan.position.avg_price:.3f} ₽)")

            # Г. Исполнение Take 1 (для объема, открытого по Entry 1)
            if tp1_reachable:
                self._execute_take_profit_1(plan, candle_idx, candle, events)
                return events

        # -----------------------------------------------------------------
        # 3. СОСТОЯНИЕ: ACTIVE_E1_E2 (Оба входа исполнены, 100% позиции)
        # -----------------------------------------------------------------
        elif plan.state == PlanState.ACTIVE_E1_E2:
            stop_reachable = self._is_price_reached(plan.direction, is_stop=True, price=plan.stop_loss_price, candle=candle)
            tp1_reachable = self._is_price_reached(plan.direction, is_stop=False, price=plan.take_1_price, candle=candle)

            if stop_reachable and tp1_reachable and self.cfg.assume_stop_before_tp_on_same_bar:
                self._close_position_by_stop(plan, candle_idx, candle, events)
                return events

            if stop_reachable:
                self._close_position_by_stop(plan, candle_idx, candle, events)
                return events

            if tp1_reachable:
                self._execute_take_profit_1(plan, candle_idx, candle, events)
                return events

        # -----------------------------------------------------------------
        # 4. СОСТОЯНИЕ: ACTIVE_PARTIAL_TP1 (Take 1 зафиксирован, держим остаток до Take 2)
        # -----------------------------------------------------------------
        elif plan.state == PlanState.ACTIVE_PARTIAL_TP1:
            stop_reachable = self._is_price_reached(plan.direction, is_stop=True, price=plan.stop_loss_price, candle=candle)
            tp2_reachable = self._is_price_reached(plan.direction, is_stop=False, price=plan.take_2_price, candle=candle)

            if stop_reachable and tp2_reachable and self.cfg.assume_stop_before_tp_on_same_bar:
                self._close_position_by_stop(plan, candle_idx, candle, events)
                return events

            if stop_reachable:
                self._close_position_by_stop(plan, candle_idx, candle, events)
                return events

            if tp2_reachable:
                self._execute_take_profit_2(plan, candle_idx, candle, events)
                return events

        return events

    # =====================================================================
    # Внутренние методы логики исполнения
    # =====================================================================

    def _is_entry_reached(
        self,
        direction: str,
        limit_price: float,
        candle: Dict[str, Any],
        step: float
    ) -> bool:
        """
        Консервативная проверка исполнения лимитной заявки (Penetration Rule):
        - Для LONG: Low < limit_price - (ticks - 1)*step, либо Low == limit_price и Close <= limit_price.
        - Для SHORT: High > limit_price + (ticks - 1)*step, либо High == limit_price и Close >= limit_price.
        """
        c_low = float(candle["low"])
        c_high = float(candle["high"])
        c_close = float(candle["close"])
        req_ticks = self.cfg.min_penetration_ticks

        if direction == "LONG":
            threshold = limit_price - max(0, req_ticks - 1) * step
            if c_low < threshold:
                return True
            if abs(c_low - limit_price) < 1e-6 and c_close <= limit_price:
                return True
            return False
        else:  # SHORT
            threshold = limit_price + max(0, req_ticks - 1) * step
            if c_high > threshold:
                return True
            if abs(c_high - limit_price) < 1e-6 and c_close >= limit_price:
                return True
            return False

    def _get_entry_fill_price(
        self,
        direction: str,
        limit_price: float,
        candle: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Определяет фактическую цену исполнения лимитной заявки с учётом гэпов и проскальзывания.
        """
        c_open = float(candle["open"])
        slippage = 0.0

        if direction == "LONG":
            # Если открытие произошло с гэпом вниз ниже цены лимитки
            if c_open < limit_price:
                slip = c_open * self.cfg.slippage_gap_pct
                fill = c_open + slip
                return fill, slip
            return limit_price, 0.0
        else:  # SHORT
            # Если открытие произошло с гэпом вверх выше цены лимитки
            if c_open > limit_price:
                slip = c_open * self.cfg.slippage_gap_pct
                fill = c_open - slip
                return fill, slip
            return limit_price, 0.0

    def _is_price_reached(
        self,
        direction: str,
        is_stop: bool,
        price: float,
        candle: Dict[str, Any]
    ) -> bool:
        """
        Проверяет, достигла ли цена уровня Stop или TP внутри диапазона свечи [Low, High].
        """
        c_low = float(candle["low"])
        c_high = float(candle["high"])

        if direction == "LONG":
            if is_stop:
                return c_low <= price
            else:
                return c_high >= price
        else:  # SHORT
            if is_stop:
                return c_high >= price
            else:
                return c_low <= price

    def _apply_entry_trade(self, pos: Position, trade: Trade, candle_idx: int):
        """Обновляет состояние открытой позиции при исполнении входа."""
        if pos.total_shares == 0:
            pos.opened_bar_idx = candle_idx
            pos.is_open = True
            pos.avg_price = trade.price
            pos.total_shares = trade.shares
        else:
            new_shares = pos.total_shares + trade.shares
            pos.avg_price = (pos.avg_price * pos.total_shares + trade.price * trade.shares) / new_shares
            pos.total_shares = new_shares

        pos.commission_paid += trade.commission
        pos.trades.append(trade)

    def _close_position_by_stop(
        self,
        plan: TradePlan,
        candle_idx: int,
        candle: Dict[str, Any],
        events: List[str]
    ):
        """Закрывает позицию по Stop-Loss с учётом проскальзывания при сбросе."""
        pos = plan.position
        if pos.total_shares <= 0:
            return

        c_time = candle.get("time")
        c_open = float(candle["open"])
        slip_pct = self.cfg.slippage_stop_loss_pct

        # Если свеча открылась гэпом за стопом — выбивает по Open со скольжением
        if plan.direction == "LONG":
            exit_price = min(plan.stop_loss_price, c_open)
            slip = exit_price * slip_pct
            final_exit = exit_price - slip
            pnl_gross = (final_exit - pos.avg_price) * pos.total_shares
        else:  # SHORT
            exit_price = max(plan.stop_loss_price, c_open)
            slip = exit_price * slip_pct
            final_exit = exit_price + slip
            pnl_gross = (pos.avg_price - final_exit) * pos.total_shares

        comm = final_exit * pos.total_shares * self.cfg.commission_exit_rate
        trade = Trade(
            trade_id=f"{plan.plan_id}_SL",
            ticker=plan.ticker,
            side=OrderSide.SELL if plan.direction == "LONG" else OrderSide.BUY,
            shares=pos.total_shares,
            price=final_exit,
            commission=comm,
            timestamp=c_time,
            reason="STOP_LOSS",
            slippage=slip
        )
        pos.commission_paid += comm
        pos.realized_pnl += pnl_gross
        pos.trades.append(trade)
        pos.total_shares = 0
        pos.is_open = False
        pos.closed_bar_idx = candle_idx

        plan.state = PlanState.CLOSED_STOP_LOSS
        plan.close_reason = "STOP_LOSS"
        events.append(
            f"[{c_time}] {plan.ticker} СТОП-ЛОСС сработал: закрыто по {final_exit:.3f} ₽ "
            f"(Убыток gross: {pnl_gross:+.2f} ₽, ком.: {comm:.2f} ₽, проскальзывание: {slip:.3f} ₽)"
        )

    def _execute_take_profit_1(
        self,
        plan: TradePlan,
        candle_idx: int,
        candle: Dict[str, Any],
        events: List[str]
    ):
        """Фиксирует частичную прибыль по Take 1 и отменяет добор Entry 2."""
        pos = plan.position
        c_time = candle.get("time")
        c_open = float(candle["open"])

        # Объем для закрытия по TP1:
        # Если позиция набрана частично (только Entry 1), закрываем 50% от фактически набранного объема,
        # чтобы вторая половина могла продолжить движение к TP2.
        # Обязательно учитываем кратность размеру лота (lot_size).
        if plan.state == PlanState.ACTIVE_E1:
            half_shares = pos.total_shares // 2
            if plan.lot_size > 1:
                lots = half_shares // plan.lot_size
                shares_to_close = lots * plan.lot_size if lots > 0 else pos.total_shares
            else:
                shares_to_close = half_shares if half_shares > 0 else pos.total_shares
        else:
            shares_to_close = min(plan.take_1_shares, pos.total_shares)

        if shares_to_close <= 0:
            shares_to_close = pos.total_shares

        slip_pct = self.cfg.slippage_market_take_pct
        if plan.direction == "LONG":
            exit_price = max(plan.take_1_price, c_open)
            slip = exit_price * slip_pct
            final_exit = exit_price - slip
            pnl_gross = (final_exit - pos.avg_price) * shares_to_close
        else:  # SHORT
            exit_price = min(plan.take_1_price, c_open)
            slip = exit_price * slip_pct
            final_exit = exit_price + slip
            pnl_gross = (pos.avg_price - final_exit) * shares_to_close

        comm = final_exit * shares_to_close * self.cfg.commission_exit_rate
        trade = Trade(
            trade_id=f"{plan.plan_id}_TP1",
            ticker=plan.ticker,
            side=OrderSide.SELL if plan.direction == "LONG" else OrderSide.BUY,
            shares=shares_to_close,
            price=final_exit,
            commission=comm,
            timestamp=c_time,
            reason="TAKE_1",
            slippage=slip
        )
        pos.commission_paid += comm
        pos.realized_pnl += pnl_gross
        pos.trades.append(trade)
        pos.total_shares -= shares_to_close

        if pos.total_shares <= 0:
            pos.is_open = False
            pos.closed_bar_idx = candle_idx
            plan.state = PlanState.CLOSED_FULL_PROFIT
            plan.close_reason = "TAKE_1_FULL"
        else:
            plan.state = PlanState.ACTIVE_PARTIAL_TP1

        events.append(
            f"[{c_time}] {plan.ticker} ТЕЙК 1 зафиксирован: {shares_to_close} шт. по {final_exit:.3f} ₽ "
            f"(Прибыль gross: {pnl_gross:+.2f} ₽, ком.: {comm:.2f} ₽). Добор Entry 2 отменён."
        )

    def _execute_take_profit_2(
        self,
        plan: TradePlan,
        candle_idx: int,
        candle: Dict[str, Any],
        events: List[str]
    ):
        """Закрывает остаток позиции по Take 2 (полный профит)."""
        pos = plan.position
        shares_to_close = pos.total_shares
        if shares_to_close <= 0:
            return

        c_time = candle.get("time")
        c_open = float(candle["open"])
        slip_pct = self.cfg.slippage_market_take_pct

        if plan.direction == "LONG":
            exit_price = max(plan.take_2_price, c_open)
            slip = exit_price * slip_pct
            final_exit = exit_price - slip
            pnl_gross = (final_exit - pos.avg_price) * shares_to_close
        else:  # SHORT
            exit_price = min(plan.take_2_price, c_open)
            slip = exit_price * slip_pct
            final_exit = exit_price + slip
            pnl_gross = (pos.avg_price - final_exit) * shares_to_close

        comm = final_exit * shares_to_close * self.cfg.commission_exit_rate
        trade = Trade(
            trade_id=f"{plan.plan_id}_TP2",
            ticker=plan.ticker,
            side=OrderSide.SELL if plan.direction == "LONG" else OrderSide.BUY,
            shares=shares_to_close,
            price=final_exit,
            commission=comm,
            timestamp=c_time,
            reason="TAKE_2",
            slippage=slip
        )
        pos.commission_paid += comm
        pos.realized_pnl += pnl_gross
        pos.trades.append(trade)
        pos.total_shares = 0
        pos.is_open = False
        pos.closed_bar_idx = candle_idx

        plan.state = PlanState.CLOSED_FULL_PROFIT
        plan.close_reason = "TAKE_2_FULL"
        events.append(
            f"[{c_time}] {plan.ticker} ТЕЙК 2 исполнен: закрыт остаток {shares_to_close} шт. по {final_exit:.3f} ₽ "
            f"(Прибыль gross: {pnl_gross:+.2f} ₽, ком.: {comm:.2f} ₽). План полностью завершён."
        )

    def close_position_at_market(
        self,
        plan: TradePlan,
        candle_idx: int,
        candle: Dict[str, Any],
        events: List[str],
        reason: str = "END_OF_BACKTEST"
    ):
        """
        Закрывает открытую позицию по рыночной цене закрытия текущей свечи (Close).
        Используется принудительно в конце бэктеста без стопового штрафного проскальзывания.
        """
        pos = plan.position
        if pos.total_shares <= 0 or not pos.is_open:
            return

        c_time = candle.get("time")
        c_close = float(candle["close"])
        final_exit = c_close

        if plan.direction == "LONG":
            pnl_gross = (final_exit - pos.avg_price) * pos.total_shares
        else:
            pnl_gross = (pos.avg_price - final_exit) * pos.total_shares

        comm = final_exit * pos.total_shares * self.cfg.commission_exit_rate
        trade = Trade(
            trade_id=f"{plan.plan_id}_{reason}",
            ticker=plan.ticker,
            side=OrderSide.SELL if plan.direction == "LONG" else OrderSide.BUY,
            shares=pos.total_shares,
            price=final_exit,
            commission=comm,
            timestamp=c_time,
            reason=reason,
            slippage=0.0
        )
        pos.commission_paid += comm
        pos.realized_pnl += pnl_gross
        pos.trades.append(trade)
        pos.total_shares = 0
        pos.is_open = False
        pos.closed_bar_idx = candle_idx

        plan.state = PlanState.CLOSED_END_OF_BACKTEST
        plan.close_reason = reason
        events.append(
            f"[{c_time}] {plan.ticker} Закрытие по рынку ({reason}) по {final_exit:.3f} ₽ "
            f"(PnL gross: {pnl_gross:+.2f} ₽, ком.: {comm:.2f} ₽)"
        )
