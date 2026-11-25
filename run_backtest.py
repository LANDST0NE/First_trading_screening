"""
Точка входа для запуска исторического тестирования торговой стратегии (Backtester CLI).

Использование:
  python run_backtest.py SBER                   # Тест Сбера в режиме auto
  python run_backtest.py SBER long              # Только лонг
  python run_backtest.py MAGN --days 180        # За последние 180 дней
  python run_backtest.py all                    # Портфельный тест по всем ликвидным бумагам
  python run_backtest.py SBER --reload          # Принудительно перезагрузить историю из API
"""

import sys
import argparse
from typing import List

from rich.console import Console
from rich.table import Table

from src.client import TInvestClient
from src.config import DEFAULT_TICKERS
from src.backtest_config import BacktestConfig
from src.backtester import Backtester, print_backtest_report
from src.metrics import BacktestMetrics

console = Console()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Историческое тестирование торговой системы MOEX (TQBR)")
    parser.add_argument("ticker", type=str, help="Тикер акции (например, SBER, MAGN) или 'all' для портфельного теста")
    parser.add_argument("direction", nargs="?", default="auto", choices=["auto", "long", "short"], help="Направление торговли (auto, long, short)")
    parser.add_argument("--days", type=int, default=180, help="Глубина исторического тестирования в днях (по умолчанию: 180)")
    parser.add_argument("--deposit", type=float, default=100_000.0, help="Начальный капитал в рублях (по умолчанию: 100 000)")
    parser.add_argument("--risk", type=float, default=1_000.0, help="Фиксированный риск на сделку в рублях (по умолчанию: 1 000)")
    parser.add_argument("--reload", action="store_true", help="Принудительно обновить кэш исторических данных через API")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ticker_arg = args.ticker.upper().strip()
    direction = args.direction.lower()
    days = args.days

    try:
        client = TInvestClient()
    except Exception as e:
        console.print(f"[bold red][ОШИБКА ИНИЦИАЛИЗАЦИИ КЛИЕНТА][/bold red] {e}")
        sys.exit(1)

    cfg = BacktestConfig(
        initial_capital=args.deposit,
        fixed_risk_per_trade=args.risk
    )
    backtester = Backtester(client=client, config=cfg)

    if ticker_arg == "ALL":
        console.print(f"\n[bold cyan]Запуск портфельного бэктеста по корзине акций ({len(DEFAULT_TICKERS)} тикеров)...[/bold cyan]")
        console.print(f"[dim]Горизонт: {days} дней | Депозит: {args.deposit:,.0f} ₽ | Риск: {args.risk:,.0f} ₽ | Направление: {direction}[/dim]\n")

        all_plans = []
        ticker_metrics = []

        for t in DEFAULT_TICKERS:
            try:
                console.print(f"[cyan]Тестирование {t}...[/cyan]", end="\r")
                inst = client.get_instrument_by_ticker(t)
                if not inst:
                    continue

                candles = backtester.load_historical_data(
                    ticker=t,
                    days_1h=days,
                    days_4h=max(240, days * 2),
                    days_1d=max(500, days * 3),
                    force_reload=args.reload
                )
                m = backtester.run_ticker(
                    ticker=t,
                    candles_dict=candles,
                    instrument=inst,
                    direction_pref=direction
                )
                ticker_metrics.append((t, m))
            except Exception as e:
                console.print(f"[dim red]Ошибка {t}: {e}[/dim red]")

        # Сводная таблица по всем тикерам
        console.print("\n")
        t_summary = Table(title=f"Сводка портфельного бэктеста ({days} дней)", header_style="bold magenta")
        t_summary.add_column("Тикер", style="bold")
        t_summary.add_column("Сигналы", justify="right")
        t_summary.add_column("Сделки", justify="right")
        t_summary.add_column("Win Rate", justify="right")
        t_summary.add_column("Profit Factor", justify="right")
        t_summary.add_column("Чистый PnL", justify="right")
        t_summary.add_column("Max DD %", justify="right")

        total_net = 0.0
        total_trades = 0
        total_wins = 0

        for t, m in ticker_metrics:
            total_net += m.net_profit
            total_trades += m.executed_plans
            total_wins += m.winning_plans
            pnl_c = "green" if m.net_profit >= 0 else "red"
            t_summary.add_row(
                t,
                str(m.total_plans),
                str(m.executed_plans),
                f"{m.win_rate_pct:.1f}%",
                f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "N/A",
                f"[{pnl_c}]{m.net_profit:+,.2f} ₽[/{pnl_c}]",
                f"{m.max_drawdown_pct:.1f}%"
            )

        console.print(t_summary)
        port_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
        p_color = "green" if total_net >= 0 else "red"
        console.print(f"\n[bold]ИТОГО по портфелю: Сделок = {total_trades}, Win Rate = {port_wr:.1f}%, Чистый PnL = [{p_color}]{total_net:+,.2f} ₽[/{p_color}][/bold]\n")

    else:
        # Тестирование одиночного тикера
        console.print(f"\n[bold cyan]Запуск исторического тестирования {ticker_arg}...[/bold cyan]")
        console.print(f"[dim]Горизонт: {days} дней | Депозит: {args.deposit:,.0f} ₽ | Риск: {args.risk:,.0f} ₽ | Режим: {direction}[/dim]\n")

        inst = client.get_instrument_by_ticker(ticker_arg)
        if not inst:
            console.print(f"[bold red]Тикер {ticker_arg} не найден в TQBR.[/bold red]")
            sys.exit(1)

        candles = backtester.load_historical_data(
            ticker=ticker_arg,
            days_1h=days,
            days_4h=max(240, days * 2),
            days_1d=max(500, days * 3),
            force_reload=args.reload
        )

        metrics = backtester.run_ticker(
            ticker=ticker_arg,
            candles_dict=candles,
            instrument=inst,
            direction_pref=direction
        )
        print_backtest_report(metrics, ticker=ticker_arg)


if __name__ == "__main__":
    main()
