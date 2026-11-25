"""Unit test suite for src/execution.py.

Verifies the conservative execution simulator against all specified edge cases:
1. Penetration rule (touch-only fails unless close <= limit; penetration by min_penetration_ticks succeeds)
2. Gap execution (opens below buy limit -> fill at Open + gap slippage)
3. Intra-candle conflict: Entry 1 vs Stop Loss on the same bar (assumes Stop occurred first -> plan cancelled)
4. Intra-candle conflict: Entry 2 vs TP1 on the same bar (assumes TP1 occurred first -> Entry 2 cancelled, TP1 executed on Entry 1)
5. Intra-candle conflict: Stop Loss vs TP on the same bar (assumes Stop occurred first -> closed at Stop Loss)
6. TTL expiration (3 bars pass without fill -> EXPIRED_NO_FILL)
7. Entry 2 cancellation after TP1 is achieved on a separate bar
8. Partial take-profit execution (TP1 closes 50%, TP2 closes remaining 50%)
"""

import unittest
from datetime import datetime
from src.backtest_config import BacktestConfig
from src.execution import (
    ExecutionSimulator,
    TradePlan,
    OrderSide,
    PlanState,
)


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
        # Case A: Touch only (Low == 100.0, Close > 100.0). Step = 0.01, min_penetration = 1 tick.
        # Threshold: 100.0 - 0 = 100.0. Low < 100.0 is False. Low == 100.0 and Close (100.5) <= 100.0 is False.
        # Result: Must NOT fill.
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

        # Case B: Penetration by 1 tick (Low = 99.99 < 100.0).
        # Result: Must fill!
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

        # Case C: Exact touch with Close <= limit (Low = 100.0, Close = 100.0).
        # Result: Must fill!
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
        # Entry 1 limit is 100.0. Next candle opens at 98.5.
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
        # Expected price: Open (98.5) + slippage (0.03% = 98.5 * 0.0003 = 0.02955) -> ~98.53
        expected_fill = 98.5 * (1.0 + self.config.slippage_gap_pct)
        self.assertAlmostEqual(plan.position.avg_price, expected_fill, places=4)
        self.assertLess(plan.position.avg_price, 100.0, "Gap fill must be at open+slippage, not limit 100.0")

    def test_03_same_bar_conflict_entry_vs_stop(self):
        """If both Entry 1 and Stop are hit in the very first bar of execution:
        Conservative rule assumes Stop occurred before Entry -> Cancel plan, no trade.
        """
        plan = self._create_sample_plan()
        # Entry 1 = 100, Stop = 96.
        # Candle drops from 101 to 95 and closes at 97.
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
        self.assertEqual(len(plan.position.trades), 0, "No trades should be registered when entry is cancelled by stop ambiguity")

    def test_04_same_bar_conflict_entry2_vs_tp1(self):
        """Position is already active with Entry 1.
        Next bar: price drops down to Entry 2 (98) and also reaches TP1 (104).
        Conservative rule assumes TP1 occurred first -> Entry 2 cancelled, TP1 executed on Entry 1.
        """
        plan = self._create_sample_plan()
        # Bar 1: Fill Entry 1 only (Low = 99.5)
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

        # Bar 2: Wild candle hitting both Entry 2 (98.0) and TP1 (104.0). Low = 97.5, High = 104.5.
        bar2 = {
            "time": datetime(2026, 1, 1, 12, 0),
            "open": 100.0,
            "high": 104.5,
            "low": 97.5,
            "close": 103.0,
            "volume": 2000
        }
        self.sim.process_candle(plan, 2, bar2)
        # TP1 should be executed, state should be ACTIVE_PARTIAL_TP1
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1)
        # Position should now have 5 shares left (10 - take_1_shares // 2)
        self.assertEqual(plan.position.total_shares, 5)

    def test_05_same_bar_conflict_stop_vs_tp(self):
        """Active position. Candle hits both Stop Loss and TP.
        Conservative rule assumes Stop Loss occurred first -> Closed at Stop Loss.
        """
        plan = self._create_sample_plan()
        # Bar 1: Fill Entry 1
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

        # Bar 2: Candle spans from 95.0 (below Stop 96) to 105.0 (above TP1 104).
        bar2 = {
            "time": datetime(2026, 1, 1, 12, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 98.0,
            "volume": 3000
        }
        self.sim.process_candle(plan, 2, bar2)
        # Must close at Stop Loss, NOT at TP1
        self.assertEqual(plan.state, PlanState.CLOSED_STOP_LOSS)
        self.assertFalse(plan.position.is_open)
        self.assertEqual(plan.close_reason, "STOP_LOSS")
        # Verify sell trade was at stop price (with stop slippage)
        sell_trades = [t for t in plan.position.trades if t.reason == "STOP_LOSS"]
        self.assertEqual(len(sell_trades), 1)
        expected_stop_fill = 96.0 * (1.0 - self.config.slippage_stop_loss_pct)
        self.assertAlmostEqual(sell_trades[0].price, expected_stop_fill, places=4)

    def test_06_ttl_expiration(self):
        """If 3 bars pass without Entry 1 being filled, plan expires."""
        plan = self._create_sample_plan()
        # Created at bar_idx=0. TTL = 3 bars -> bars 1, 2, 3 are valid.
        # At bar 4 (> created_bar_idx + limit_expiry_bars), it expires.
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
        # Bar 1: Fill Entry 1
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

        # Bar 2: Moves directly up to TP1 (104.0), Low is 100.1 (nowhere near Entry 2 at 98.0)
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

        # Bar 3: Price now drops to 97.0 (where Entry 2 used to be at 98.0).
        # But since TP1 was already taken, Entry 2 must NOT be filled!
        bar3 = {
            "time": datetime(2026, 1, 1, 13, 0),
            "open": 103.0,
            "high": 103.0,
            "low": 97.0,
            "close": 98.0,
            "volume": 1000
        }
        self.sim.process_candle(plan, 3, bar3)
        # It should still be ACTIVE_PARTIAL_TP1 (or closed if stop was hit), shares should remain 5!
        self.assertEqual(plan.position.total_shares, 5)

    def test_08_full_two_take_cycle(self):
        """Entry 1 filled, then Entry 2 filled (average position).
        Then TP1 reached (50% closed), then TP2 reached (100% closed -> CLOSED_FULL_PROFIT).
        """
        plan = self._create_sample_plan()
        # Bar 1: Entry 1 fills
        bar1 = {"time": datetime(2026, 1, 1, 11, 0), "open": 100.5, "high": 101, "low": 99.5, "close": 99.8, "volume": 100}
        self.sim.process_candle(plan, 1, bar1)
        self.assertEqual(plan.position.total_shares, 10)

        # Bar 2: Entry 2 fills
        bar2 = {"time": datetime(2026, 1, 1, 12, 0), "open": 99.8, "high": 100, "low": 97.5, "close": 98.2, "volume": 100}
        self.sim.process_candle(plan, 2, bar2)
        self.assertEqual(plan.state, PlanState.ACTIVE_E1_E2)
        self.assertEqual(plan.position.total_shares, 20)
        self.assertAlmostEqual(plan.position.avg_price, 99.0, places=2)

        # Bar 3: Hits TP1 (104.0)
        bar3 = {"time": datetime(2026, 1, 1, 13, 0), "open": 98.5, "high": 104.5, "low": 98.5, "close": 103.5, "volume": 100}
        self.sim.process_candle(plan, 3, bar3)
        self.assertEqual(plan.state, PlanState.ACTIVE_PARTIAL_TP1)
        self.assertEqual(plan.position.total_shares, 10) # 50% left

        # Bar 4: Hits TP2 (108.0)
        bar4 = {"time": datetime(2026, 1, 1, 14, 0), "open": 103.5, "high": 108.5, "low": 103.0, "close": 108.2, "volume": 100}
        self.sim.process_candle(plan, 4, bar4)
        self.assertEqual(plan.state, PlanState.CLOSED_FULL_PROFIT)
        self.assertEqual(plan.position.total_shares, 0)
        self.assertFalse(plan.position.is_open)
        self.assertGreater(plan.position.realized_pnl, 0)


if __name__ == "__main__":
    unittest.main()
