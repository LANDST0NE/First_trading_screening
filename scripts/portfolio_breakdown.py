import sys
import os
sys.path.insert(0, os.path.abspath("."))

import collections
from src.client import TInvestClient
from src.config import DEFAULT_TICKERS
from src.backtest_config import BacktestConfig
from src.backtester import Backtester
from src.metrics import MetricsCalculator

client = TInvestClient()
cfg = BacktestConfig(initial_capital=100_000.0, fixed_risk_per_trade=1_000.0)
backtester = Backtester(client=client, config=cfg)

all_plans = []
ticker_results = []

for t in DEFAULT_TICKERS:
    inst = client.get_instrument_by_ticker(t)
    if not inst:
        continue
    candles = backtester.load_historical_data(ticker=t, days_1h=60, days_4h=240, days_1d=500, force_reload=False)
    m = backtester.run_ticker(ticker=t, candles_dict=candles, instrument=inst, direction_pref="auto")
    all_plans.extend(backtester.last_plans)
    ticker_results.append((t, m))

# Separate plans by trade_type
plans_by_type = collections.defaultdict(list)
for p in all_plans:
    if p.position.trades and not p.position.is_open:
        plans_by_type[p.trade_type].append(p)

print("\n" + "=" * 115)
print("ИТОГОВАЯ СВОДКА ПО ТИПАМ СЕТАПОВ В ПОРТФЕЛЬНОМ БЭКТЕСТЕ (60 ДНЕЙ, 16 ТИКЕРОВ)")
print("=" * 115)
print(f"{'Тип сетапа':<25} | {'Сделок':<6} | {'Win Rate':<8} | {'PF':<6} | {'Gross Win':<12} | {'Gross Loss':<12} | {'Комиссии':<10} | {'Чистый PnL':<12} | {'Max DD %':<8}")
print("-" * 115)

for tt, plans in sorted(plans_by_type.items(), key=lambda x: -len(x[1])):
    # Sort plans by close timestamp
    def get_time(plan):
        if plan.position.trades:
            return str(plan.position.trades[-1].timestamp)
        return str(plan.created_bar_idx)

    plans_sorted = sorted(plans, key=get_time)
    
    wins = [p for p in plans_sorted if p.position.net_pnl > 0.01]
    losses = [p for p in plans_sorted if p.position.net_pnl < -0.01]
    breakevens = [p for p in plans_sorted if abs(p.position.net_pnl) <= 0.01]
    
    total = len(plans_sorted)
    win_rate = (len(wins) / total * 100.0) if total > 0 else 0.0
    
    gross_win = sum(p.position.realized_pnl for p in plans_sorted if p.position.realized_pnl > 0.0001)
    gross_loss = sum(abs(p.position.realized_pnl) for p in plans_sorted if p.position.realized_pnl < -0.0001)
    total_comm = sum(p.position.commission_paid for p in plans_sorted)
    net_pnl = sum(p.position.net_pnl for p in plans_sorted)
    
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    
    # Calculate Max Drawdown for this setup type (assuming 100k starting equity)
    eq = 100_000.0
    peak = eq
    max_dd_pct = 0.0
    for p in plans_sorted:
        eq += p.position.net_pnl
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd
            
    print(f"{tt:<25} | {total:<6} | {win_rate:>7.1f}% | {pf:>6.2f} | {gross_win:>11.2f}₽ | {gross_loss:>11.2f}₽ | {total_comm:>9.2f}₽ | {net_pnl:>11.2f}₽ | {max_dd_pct:>7.2f}%")

print("=" * 115)
