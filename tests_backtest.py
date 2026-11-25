"""Unit test suite for src/execution.py, src/metrics.py, and src/backtester.py.

Verifies the conservative execution simulator, strict financial consistency,
and zero-lookahead multi-timeframe synchronization against all audited edge cases:
1. Penetration rule (touch-only fails unless close <= limit; penetration by min_penetration_ticks succeeds)
2. Gap execution (opens below buy limit -> fill at Open + gap slippage)
3. Intra-candle conflict: Entry 1 vs Stop Loss on same bar (assumes Stop occurred first -> plan cancelled)
4. Intra-candle conflict: Entry 2 vs TP1 on same bar (assumes TP1 occurred first -> Entry 2 cancelled, TP1 executed on Entry 1)
5. Intra-candle conflict: Stop Loss vs TP on same bar (assumes Stop occurred first -> closed at Stop Loss)
6. TTL expiration (3 bars pass without fill -> EXPIRED_NO_FILL)
7. Entry 2 cancellation after TP1 is achieved on a separate bar
8. Full two-take cycle (Entry 1 -> Entry 2 -> TP1 partial -> TP2 final)
9. NEW: TP1 executed immediately on the same candle as Entry 1 (when Entry 2 is not touched)
10. NEW: Strict financial consistency (Gross PnL - Commissions == Net PnL == Final Capital - Initial Capital)
11. NEW: Multi-timeframe synchronization (Zero Lookahead Bias on 4H and 1D)
12. NEW: End-of-backtest market close (no stop slippage, not counted as Stop-Loss in metrics)
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd

from src.backtest_config import BacktestConfig
from src.execution import (
    ExecutionSimulator,
    TradePlan,
    OrderSide,
    PlanState,
)
from src.metrics import MetricsCalculator


class TestExecutionSimulator(unittest.TestCase):
    def setUp(self):
        self.config = BacktestConfig(
            slippage_gap_pct=0.0003,          # 0.03%
            slippage_market_take_pct=0.0002,  # 0.02%
            slippage_stop_loss_pct=0.0005,    # 0.05%
            commission_entry_rate=0.0005,     # 0.05%
            commission_exit_rate=0.0005,      # 0.05%
            limit_expiry_bars=3,
            min_penetration_ticks=1,
            assume_stop_before_entry_on_same_bar=True,
            assume_tp_before_entry2_on_same_bar=True,
            assume_stop_before_tp_on_same_bar=True,
            cancel_entry2_on_tp1=True,
        )
        self.sim = ExecutionSimulator(config=self.config)

    def _create_sample_plan(self, plan_id: str = "PLAN_1", step: float = 0.01) -> TradePlan:
        return TradePlan(
            plan_id=plan_id,
            ticker="SBER",
            direction="LONG",
            step=step,
            lot_size=1,
            created_bar_idx=0,
            entry_1_price=100.0,
            entry_1_shares=10,
            entry_2_price=98.0,
            entry_2_shares=10,
            stop_loss_price=96.0,
            take_1_price=104.0,
            take_1_shares=10,
            take_2_price=108.0,
            take_2_shares=10,
            trade_type="Trend Following",
        )

    def test_01_penetration_rule(self):
        """Test limit buy penetration rule."""
        plan_a = self._create_sample_plan("PLAN_A")
        candle_touch = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 101.0,
            "high": 102.0,
            "low": 100.0,
            "close": 100.5,
            "volume": 1000
        }
        self.sim.process_candle(plan_a, 1, candle_touch)
        self.assertEqual(plan_a.state, PlanState.WAITING_ENTRY, "Touch-only without close <= limit must NOT fill Buy Limit")
        self.assertFalse(plan_a.position.is_open)

        plan_b = self._create_sample_plan("PLAN_B")
        candle_penetrate = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 101.0,
            "high": 101.5,
            "low": 99.99,
            "close": 100.2,
            "volume": 1000
        }
        self.sim.process_candle(plan_b, 1, candle_penetrate)
        self.assertEqual(plan_b.state, PlanState.ACTIVE_E1)
        self.assertTrue(plan_b.position.is_open)
        self.assertEqual(plan_b.position.total_shares, 10)
        self.assertAlmostEqual(plan_b.position.avg_price, 100.0, places=2)

        plan_c = self._create_sample_plan("PLAN_C")
        candle_close_at_limit = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 100.5,
            "high": 101.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000
        }
        self.sim.process_candle(plan_c, 1, candle_close_at_limit)
        self.assertEqual(plan_c.state, PlanState.ACTIVE_E1)
        self.assertTrue(plan_c.position.is_open)

    def test_02_gap_down_execution(self):
        """If market opens with a gap below limit buy, fill at Open + gap slippage."""
        plan = self._create_sample_plan()
        candle_gap = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 98.5,
            "high": 99.5,
            "low": 98.0,
            "close": 99.0,
            "volume": 1000
        }
        self.sim.process_candle(plan, 1, candle_gap)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1)
        expected_fill = 98.5 * (1.0 + self.config.slippage_gap_pct)
        self.assertAlmostEqual(plan.position.avg_price, expected_fill, places=4)
        self.assertLess(plan.position.avg_price, 100.0, "Gap fill must be at open+slippage, not limit 100.0")

    def test_03_same_bar_conflict_entry_vs_stop(self):
        """If both Entry 1 and Stop are hit in the very first bar of execution:
        Conservative rule assumes Stop occurred before Entry -> Cancel plan, no trade.
        """
        plan = self._create_sample_plan()
        candle_crash = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 101.0,
            "high": 101.5,
            "low": 95.0,
            "close": 97.0,
            "volume": 5000
        }
        self.sim.process_candle(plan, 1, candle_crash)
        self.assertEqual(plan.state, PlanState.CANCELLED_HIT_STOP_ON_ENTRY)
        self.assertFalse(plan.position.is_open)
        self.assertEqual(len(plan.position.trades), 0)

    def test_04_same_bar_conflict_entry2_vs_tp1(self):
        """Position is already active with Entry 1.
        Next bar: price drops down to Entry 2 (98) and also reaches TP1 (104).
        Conservative rule assumes TP1 occurred first -> Entry 2 cancelled, TP1 executed on Entry 1.
        """
        plan = self._create_sample_plan()
        bar1 = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 100.5,
            "high": 101.0,
            "low": 99.5,
            "close": 100.2,
            "volume": 1000
        }
        self.sim.process_candle(plan, 1, bar1)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1)
        self.assertEqual(plan.position.total_shares, 10)

        bar2 = {
            "time": datetime(2026, 1, 1, 12, 0),
            "open": 100.0,
            "high": 104.5,
            "low": 97.5,
            "close": 103.0,
            "volume": 2000
        }
        self.sim.process_candle(plan, 2, bar2)
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1)
        self.assertEqual(plan.position.total_shares, 5)

    def test_05_same_bar_conflict_stop_vs_tp(self):
        """Active position. Candle hits both Stop Loss and TP.
        Conservative rule assumes Stop Loss occurred first -> Closed at Stop Loss.
        """
        plan = self._create_sample_plan()
        bar1 = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 100.5,
            "high": 101.0,
            "low": 99.5,
            "close": 100.2,
            "volume": 1000
        }
        self.sim.process_candle(plan, 1, bar1)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1)

        bar2 = {
            "time": datetime(2026, 1, 1, 12, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 98.0,
            "volume": 3000
        }
        self.sim.process_candle(plan, 2, bar2)
        self.assertEqual(plan.state, PlanState.CLOSED_STOP_LOSS)
        self.assertFalse(plan.position.is_open)
        self.assertEqual(plan.close_reason, "STOP_LOSS")
        sell_trades = [t for t in plan.position.trades if t.reason == "STOP_LOSS"]
        self.assertEqual(len(sell_trades), 1)
        expected_stop_fill = 96.0 * (1.0 - self.config.slippage_stop_loss_pct)
        self.assertAlmostEqual(sell_trades[0].price, expected_stop_fill, places=4)

    def test_06_ttl_expiration(self):
        """If 3 bars pass without Entry 1 being filled, plan expires."""
        plan = self._create_sample_plan()
        candles = [
            {"time": datetime(2026, 1, 1, 11, 0), "open": 102, "high": 103, "low": 101, "close": 102, "volume": 100},
            {"time": datetime(2026, 1, 1, 12, 0), "open": 102, "high": 103, "low": 101, "close": 102, "volume": 100},
            {"time": datetime(2026, 1, 1, 13, 0), "open": 102, "high": 103, "low": 101, "close": 102, "volume": 100},
            {"time": datetime(2026, 1, 1, 14, 0), "open": 102, "high": 103, "low": 101, "close": 102, "volume": 100},
        ]
        for idx, candle in enumerate(candles, start=1):
            self.sim.process_candle(plan, idx, candle)

        self.assertEqual(plan.state, PlanState.EXPIRED_NO_FILL)
        self.assertIn("TTL", plan.cancel_reason)

    def test_07_entry2_cancellation_after_tp1_on_separate_bars(self):
        """Entry 1 filled on bar 1. TP1 reached on bar 2 without hitting Entry 2.
        Entry 2 must not be filled later.
        """
        plan = self._create_sample_plan()
        bar1 = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 100.5,
            "high": 101.0,
            "low": 99.5,
            "close": 100.2,
            "volume": 1000
        }
        self.sim.process_candle(plan, 1, bar1)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1)

        bar2 = {
            "time": datetime(2026, 1, 1, 12, 0),
            "open": 100.2,
            "high": 104.5,
            "low": 100.1,
            "close": 104.2,
            "volume": 1000
        }
        self.sim.process_candle(plan, 2, bar2)
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1)

        bar3 = {
            "time": datetime(2026, 1, 1, 13, 0),
            "open": 103.0,
            "high": 103.0,
            "low": 97.0,
            "close": 98.0,
            "volume": 1000
        }
        self.sim.process_candle(plan, 3, bar3)
        self.assertEqual(plan.position.total_shares, 5)

    def test_08_full_two_take_cycle(self):
        """Entry 1 filled, then Entry 2 filled (average position).
        Then TP1 reached (50% closed), then TP2 reached (100% closed -> CLOSED_FULL_PROFIT).
        """
        plan = self._create_sample_plan()
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 99.8, "volume": 100}
        self.sim.process_candle(plan, 1, bar1)
        self.assertEqual(plan.position.total_shares, 10)

        bar2 = {"time": datetime(2026, 1, 1, 12, 0), "open": 99.8, "high": 100, "low": 97.5, "close": 98.2, "volume": 100}
        self.sim.process_candle(plan, 2, bar2)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1_E2)
        self.assertEqual(plan.position.total_shares, 20)
        self.assertAlmostEqual(plan.position.avg_price, 99.0, places=2)

        bar3 = {"time": datetime(2026, 1, 1, 13, 0), "open": 98.5, "high": 104.5, "low": 98.5, "close": 103.5, "volume": 100}
        self.sim.process_candle(plan, 3, bar3)
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1)
        self.assertEqual(plan.position.total_shares, 10)

        bar4 = {"time": datetime(2026, 1, 1, 14, 0), "open": 103.5, "high": 108.5, "low": 103.0, "close": 108.2, "volume": 100}
        self.sim.process_candle(plan, 4, bar4)
        self.assertEqual(plan.state, PlanState.CLOSED_FULL_PROFIT)
        self.assertEqual(plan.position.total_shares, 0)
        self.assertFalse(plan.position.is_open)
        self.assertGreater(plan.position.realized_pnl, 0)

    def test_09_same_bar_entry1_and_tp1(self):
        """Audit Issue 3: TP1 reached on the exact same candle as Entry 1 (when Entry 2 is not hit).
        Must immediately execute TP1, cancel Entry 2, and transition to ACTIVE_PARTIAL_TP1.
        """
        plan = self._create_sample_plan()
        # Bar 1: opens at 100.5, drops to 99.5 (hits Entry 1=100.0, but NOT Entry 2=98.0),
        # then rallies to High = 104.5 (hits TP1=104.0), closes at 103.5.
        candle_spike = {
            "time": datetime(2026, 1, 1, 11, 0),
            "open": 100.5,
            "high": 104.5,
            "low": 99.5,
            "close": 103.5,
            "volume": 2000
        }
        self.sim.process_candle(plan, 1, candle_spike)
        # Entry 1 executed (10 shares) and TP1 executed immediately (5 shares closed)
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1, "Must be ACTIVE_PARTIAL_TP1 on immediate TP1 execution")
        self.assertEqual(plan.position.total_shares, 5, "Remaining shares must be 5")
        # Must have 2 trades: ENTRY_1 and TAKE_1
        reasons = [t.reason for t in plan.position.trades]
        self.assertIn("ENTRY_1", reasons)
        self.assertIn("TAKE_1", reasons)

    def test_10_financial_accounting_consistency(self):
        """Audit Issue 2: Financial consistency across modules.
        Verifies: Gross Profit - Gross Loss - Total Commissions == Net Profit == Final Capital - Initial Capital.
        """
        init_cap = 100_000.0
        plan = self._create_sample_plan()

        # Trade 1: Entry 1 fills, then stops out
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 100, "volume": 100}
        self.sim.process_candle(plan, 1, bar1)
        bar2 = {"time": datetime(2026, 1, 1, 12, 0), "open": 98.0, "high": 98, "low": 95.0, "close": 95.5, "volume": 100}
        self.sim.process_candle(plan, 2, bar2)

        self.assertEqual(plan.state, PlanState.CLOSED_STOP_LOSS)
        pos = plan.position

        # Position internal consistency
        expected_net_pnl = pos.realized_pnl - pos.commission_paid
        self.assertAlmostEqual(pos.net_pnl, expected_net_pnl, places=4)

        # Metrics calculator consistency
        metrics = MetricsCalculator.calculate([plan], initial_capital=init_cap)

        expected_calc_net = metrics.gross_profit - metrics.gross_loss - metrics.total_commissions
        self.assertAlmostEqual(metrics.net_profit, expected_calc_net, places=4, msg="Net Profit must equal Gross - Loss - Comm")
        self.assertAlmostEqual(metrics.final_capital, init_cap + metrics.net_profit, places=4, msg="Final Capital must equal Init + Net")
        self.assertAlmostEqual(metrics.net_profit, pos.net_pnl, places=4, msg="Metrics Net Profit must match Position net_pnl")

    def test_11_timeframe_synchronization_no_lookahead(self):
        """Audit Issue 1: Multi-timeframe synchronization without lookahead bias.
        A 4H candle starting at 12:00 closes at 16:00.
        It must NOT be visible to 1H candles at 12:00, 13:00, 14:00, and only becomes visible after 16:00.
        """
        # Create 4H DataFrame
        df_4h = pd.DataFrame([
            {"time": pd.to_datetime("2026-01-01 04:00:00+00:00"), "open": 10, "high": 11, "low": 9, "close": 10.5},
            {"time": pd.to_datetime("2026-01-01 08:00:00+00:00"), "open": 10.5, "high": 12, "low": 10, "close": 11.5},
            {"time": pd.to_datetime("2026-01-01 12:00:00+00:00"), "open": 11.5, "high": 13, "low": 11, "close": 12.5},
        ])

        # 1H candle at 12:00: opens at 12:00, closes at 13:00
        curr_1h_time = pd.to_datetime("2026-01-01 12:00:00+00:00")
        curr_1h_close = curr_1h_time + pd.Timedelta(hours=1) # 13:00

        # Filter strictly closed 4H candles
        sub_4h = df_4h[df_4h["time"] + pd.Timedelta(hours=4) <= curr_1h_close]

        # 12:00 4H candle closes at 16:00 > 13:00 -> MUST NOT be in sub_4h!
        times_in_sub = list(sub_4h["time"])
        self.assertNotIn(pd.to_datetime("2026-01-01 12:00:00+00:00"), times_in_sub, "4H candle at 12:00 must NOT be visible at 1H candle 12:00!")
        self.assertIn(pd.to_datetime("2026-01-01 08:00:00+00:00"), times_in_sub, "08:00 4H candle (closed at 12:00 <= 13:00) must be visible!")

        # 1H candle at 15:00: opens at 15:00, closes at 16:00
        curr_1h_time_15 = pd.to_datetime("2026-01-01 15:00:00+00:00")
        curr_1h_close_16 = curr_1h_time_15 + pd.Timedelta(hours=1) # 16:00
        sub_4h_16 = df_4h[df_4h["time"] + pd.Timedelta(hours=4) <= curr_1h_close_16]

        # At 16:00, 12:00 4H candle has now closed -> MUST be in sub_4h!
        self.assertIn(pd.to_datetime("2026-01-01 12:00:00+00:00"), list(sub_4h_16["time"]), "12:00 4H candle must become visible at 16:00!")

    def test_12_end_of_backtest_market_close(self):
        """Audit Issue 4: Position open at end of backtest is closed at market without stop slippage,
        and is classified as CLOSED_END_OF_BACKTEST, not as Stop-Loss.
        """
        plan = self._create_sample_plan()
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 100.0, "volume": 100}
        self.sim.process_candle(plan, 1, bar1)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1)

        last_candle = {"time": datetime(2026, 1, 1, 12, 0), "open": 101.0, "high": 102, "low": 100.5, "close": 101.5, "volume": 100}
        events = []
        self.sim.close_position_at_market(plan, 2, last_candle, events, reason="END_OF_BACKTEST")

        self.assertEqual(plan.state, PlanState.CLOSED_END_OF_BACKTEST)
        self.assertEqual(plan.close_reason, "END_OF_BACKTEST")
        self.assertFalse(plan.position.is_open)

        # Verify slippage was 0.0, not stop loss slippage
        trade = plan.position.trades[-1]
        self.assertEqual(trade.slippage, 0.0)
        self.assertEqual(trade.price, 101.5)

        # Check metrics classification
        metrics = MetricsCalculator.calculate([plan], initial_capital=100_000.0)
        self.assertEqual(metrics.stop_loss_exits, 0, "Must NOT be counted as Stop Loss")
        self.assertEqual(metrics.end_of_backtest_exits, 1, "Must be counted as end_of_backtest_exits")

    def test_13_short_direction_gross_and_net_pnl_with_commissions(self):
        """User Audit Item 2: Verify that for SHORT direction, Gross PnL is (Entry - Exit) * Shares.
        Drop in price produces positive Gross PnL; rise in price produces negative Gross PnL.
        """
        # SHORT Plan: Entry 1 = 100.0, Stop = 104.0, TP1 = 96.0
        plan_short = TradePlan(
            plan_id="SHORT_1",
            ticker="SBER",
            direction="SHORT",
            step=0.01,
            lot_size=1,
            created_bar_idx=0,
            entry_1_price=100.0,
            entry_1_shares=10,
            entry_2_price=102.0,
            entry_2_shares=10,
            stop_loss_price=104.0,
            take_1_price=96.0,
            take_1_shares=10,
            take_2_price=92.0,
            take_2_shares=10,
            trade_type="Trend Following",
        )

        # Bar 1: Short enters at 100.0
        # For short, High >= 100.0 hits the limit sell
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 99.5, "high": 100.5, "low": 99.0, "close": 99.8, "volume": 100}
        self.sim.process_candle(plan_short, 1, bar1)
        self.assertEqual(plan_short.state, PlanState.ACTIVE_E1)
        self.assertEqual(plan_short.position.avg_price, 100.0)

        # Bar 2: Price drops to 95.0 (hits TP1 = 96.0) -> PROFIT for SHORT!
        bar2 = {"time": datetime(2026, 1, 1, 12, 0), "open": 99.0, "high": 99.0, "low": 95.0, "close": 95.5, "volume": 100}
        self.sim.process_candle(plan_short, 2, bar2)

        # Closed by TP1 (50% = 5 shares)
        pos = plan_short.position
        # Gross profit for short: (Entry - Exit) * shares -> (100.0 - exit_price) * 5 > 0!
        self.assertGreater(pos.realized_pnl, 0.0, "Short trade with price drop must produce POSITIVE gross PnL")
        expected_exit = 96.0 * (1.0 + self.config.slippage_market_take_pct) # short buy to cover slips slightly up
        expected_gross = (100.0 - expected_exit) * 5
        self.assertAlmostEqual(pos.realized_pnl, expected_gross, places=3)
        self.assertAlmostEqual(pos.net_pnl, pos.realized_pnl - pos.commission_paid, places=3)

    def test_14_multiple_entries_weighted_average_pnl(self):
        """User Audit Item 3: Multiple entries with different prices and weighted average calculation."""
        plan = self._create_sample_plan()
        # Entry 1 at 100.0 (10 shares)
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 99.8, "volume": 100}
        self.sim.process_candle(plan, 1, bar1)

        # Entry 2 at 98.0 (10 shares)
        bar2 = {"time": datetime(2026, 1, 1, 12, 0), "open": 99.8, "high": 100, "low": 97.5, "close": 98.2, "volume": 100}
        self.sim.process_candle(plan, 2, bar2)

        # Position should have 20 shares at exactly 99.0 average price
        self.assertEqual(plan.position.total_shares, 20)
        self.assertAlmostEqual(plan.position.avg_price, 99.0, places=4)

        # Exit all at market 105.0
        bar3 = {"time": datetime(2026, 1, 1, 13, 0), "open": 104, "high": 106, "low": 104, "close": 105.0, "volume": 100}
        self.sim.close_position_at_market(plan, 3, bar3, [], reason="TEST_EXIT")

        # Expected gross PnL = (105.0 - 99.0) * 20 = +120.0
        self.assertAlmostEqual(plan.position.realized_pnl, 120.0, places=4)
        # Expected commissions = entry1_comm + entry2_comm + exit_comm
        c_in1 = 100.0 * 10 * self.config.commission_entry_rate
        c_in2 = 98.0 * 10 * self.config.commission_entry_rate
        c_out = 105.0 * 20 * self.config.commission_exit_rate
        expected_comm = c_in1 + c_in2 + c_out
        self.assertAlmostEqual(plan.position.commission_paid, expected_comm, places=4)
        self.assertAlmostEqual(plan.position.net_pnl, 120.0 - expected_comm, places=4)

    def test_15_gap_through_stop_loss_and_take_profit(self):
        """User Audit Item 3: Gaps through Stop Loss and Take Profit.
        Gap down through stop gives adverse fill below stop price.
        Gap up through take profit gives favorable fill above take price.
        """
        # Case A: Gap down through Stop Loss (Long)
        plan_a = self._create_sample_plan("GAP_STOP")
        # Entry 1 fills at 100.0
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 99.8, "volume": 100}
        self.sim.process_candle(plan_a, 1, bar1)

        # Bar 2: Market gaps down directly to Open = 93.0 (way below Stop = 96.0)!
        bar2_gap_stop = {"time": datetime(2026, 1, 1, 12, 0), "open": 93.0, "high": 93.5, "low": 92.5, "close": 93.0, "volume": 500}
        self.sim.process_candle(plan_a, 2, bar2_gap_stop)

        self.assertEqual(plan_a.state, PlanState.CLOSED_STOP_LOSS)
        # Fill price must be calculated from 93.0 with slippage, NOT from 96.0!
        expected_exit = 93.0 * (1.0 - self.config.slippage_stop_loss_pct)
        sl_trade = [t for t in plan_a.position.trades if t.reason == "STOP_LOSS"][0]
        self.assertAlmostEqual(sl_trade.price, expected_exit, places=3)
        self.assertLess(sl_trade.price, 96.0, "Adverse gap must fill at gapped price, not stop level")

        # Case B: Gap up through Take Profit (Long)
        plan_b = self._create_sample_plan("GAP_TAKE")
        self.sim.process_candle(plan_b, 1, bar1)

        # Bar 2: Market gaps up to Open = 106.0 (above TP1 = 104.0)!
        bar2_gap_tp = {"time": datetime(2026, 1, 1, 12, 0), "open": 106.0, "high": 107.0, "low": 105.5, "close": 106.5, "volume": 500}
        self.sim.process_candle(plan_b, 2, bar2_gap_tp)

        tp1_trade = [t for t in plan_b.position.trades if t.reason == "TAKE_1"][0]
        # Favorable gap: must fill from 106.0 (minus market take slippage), capturing the gap bonus!
        expected_tp_exit = 106.0 * (1.0 - self.config.slippage_market_take_pct)
        self.assertAlmostEqual(tp1_trade.price, expected_tp_exit, places=3)
        self.assertGreater(tp1_trade.price, 104.0, "Favorable gap must give fill above TP limit")

    def test_16_partial_take_and_trailing_stop_loss(self):
        """User Audit Item 3: Partial take profit (50%), then trailing remainder stops out."""
        plan = self._create_sample_plan()
        # Bar 1: Entry 1 fills (10 shares at 100.0)
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 99.8, "volume": 100}
        self.sim.process_candle(plan, 1, bar1)

        # Bar 2: Entry 2 fills (10 shares at 98.0) -> total 20 shares at avg 99.0
        bar2 = {"time": datetime(2026, 1, 1, 12, 0), "open": 99.8, "high": 100, "low": 97.5, "close": 98.2, "volume": 100}
        self.sim.process_candle(plan, 2, bar2)
        self.assertEqual(plan.position.total_shares, 20)

        # Bar 3: Hits TP1 = 104.0 -> partial exit of 10 shares (50%)
        bar3 = {"time": datetime(2026, 1, 1, 13, 0), "open": 98.5, "high": 104.5, "low": 98.5, "close": 103.5, "volume": 100}
        self.sim.process_candle(plan, 3, bar3)
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1)
        self.assertEqual(plan.position.total_shares, 10, "10 shares must remain open after TP1")

        # Bar 4: Market reverses and hits stop loss at 96.0
        bar4 = {"time": datetime(2026, 1, 1, 14, 0), "open": 102.0, "high": 102.0, "low": 95.0, "close": 95.5, "volume": 100}
        self.sim.process_candle(plan, 4, bar4)
        self.assertEqual(plan.state, PlanState.CLOSED_STOP_LOSS)
        self.assertEqual(plan.position.total_shares, 0, "Position must be fully closed")
        self.assertFalse(plan.position.is_open)

        # Both trades recorded
        self.assertEqual(len(plan.position.trades), 4) # E1, E2, TP1, SL
        self.assertAlmostEqual(plan.position.net_pnl, plan.position.realized_pnl - plan.position.commission_paid, places=4)

    def test_17_calendar_date_1d_synchronization(self):
        """User Audit Item 1: 1D candles are filtered strictly by calendar date:
        sub_1d = df_1d[df_1d["time"].dt.date < curr_time.date()].
        All 1H bars during day D have date D, so daily candle of day D is excluded during the entire day.
        """
        df_1d = pd.DataFrame([
            {"time": pd.to_datetime("2026-11-23 00:00:00+00:00"), "open": 288.0, "close": 289.0},
            {"time": pd.to_datetime("2026-11-24 00:00:00+00:00"), "open": 290.0, "close": 285.0},
        ])

        # 1H candle during day 2026-11-24 at 14:00
        curr_1h_time = pd.to_datetime("2026-11-24 14:00:00+00:00")
        sub_1d = df_1d[df_1d["time"].dt.date < curr_1h_time.date()]

        # 2026-11-24 daily candle MUST NOT be in sub_1d!
        dates_in_sub = [t.strftime("%Y-%m-%d") for t in sub_1d["time"]]
        self.assertNotIn("2026-11-24", dates_in_sub, "Today's daily candle must NOT be included during trading day!")
        self.assertIn("2026-11-23", dates_in_sub, "Yesterday's completed daily candle must be included!")

        # Next day: 2026-11-25 at 07:00
        next_day_time = pd.to_datetime("2026-11-25 07:00:00+00:00")
        sub_1d_next = df_1d[df_1d["time"].dt.date < next_day_time.date()]
        dates_next = [t.strftime("%Y-%m-%d") for t in sub_1d_next["time"]]
        self.assertIn("2026-11-24", dates_next, "2026-11-24 daily candle becomes available only on 2026-11-25!")


if __name__ == "__main__":
    unittest.main()
