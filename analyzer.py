import sys
import os
from typing import Optional, List, Dict, Any

from rich.panel import Panel
from src.client import TInvestClient
from src.indicators import calculate_all_indicators
from src.strategy import calculate_setup, get_senior_trend, SetupResult
from src.reporting import (
    print_single_report,
    export_single_markdown,
    print_screener_summary,
    export_screener_markdown,
    console
)
from src.config import DEFAULT_TICKERS


def analyze_single_ticker(
    client: TInvestClient,
    ticker: str,
    direction_pref: str = "auto",
    imoex_info: Optional[Dict[str, Any]] = None,
    save_report: bool = True
) -> Optional[SetupResult]:
    """
    Анализирует один тикер.
    direction_pref: 'auto', 'long', 'short'.
    """
    ticker = ticker.upper().strip()
    try:
        instrument = client.get_instrument_by_ticker(ticker)
        if not instrument:
            console.print(f"[bold red][ОШИБКА][/bold red] Акция с тикером [yellow]{ticker}[/yellow] не найдена в режиме торгов TQBR на Мосбирже.")
            return None

        candles = client.get_multi_tf_candles(instrument.figi)
        indicators = calculate_all_indicators(candles)

        if len(indicators) < 3:
            console.print(f"[bold red][ОШИБКА][/bold red] Не удалось загрузить достаточное число свечей по всем таймфреймам для {ticker}.")
            return None

        senior_trend = get_senior_trend(indicators)

        chosen_setup: SetupResult
        if direction_pref == "long":
            chosen_setup = calculate_setup(instrument, indicators, "LONG")
        elif direction_pref == "short":
            chosen_setup = calculate_setup(instrument, indicators, "SHORT")
        else:
            # Режим auto: фильтр господствующего тренда
            if senior_trend == "BEARISH":
                # Оцениваем ТОЛЬКО SHORT
                chosen_setup = calculate_setup(instrument, indicators, "SHORT")
            elif senior_trend == "BULLISH":
                # Оцениваем ТОЛЬКО LONG
                chosen_setup = calculate_setup(instrument, indicators, "LONG")
            else:
                # NEUTRAL: оцениваем оба и выбираем лучший по R/R
                setup_long = calculate_setup(instrument, indicators, "LONG")
                setup_short = calculate_setup(instrument, indicators, "SHORT")

                is_long_active = setup_long.status.startswith("ACTIVE")
                is_short_active = setup_short.status.startswith("ACTIVE")

                if is_long_active and is_short_active:
                    chosen_setup = setup_long if setup_long.rr >= setup_short.rr else setup_short
                elif is_long_active:
                    chosen_setup = setup_long
                elif is_short_active:
                    chosen_setup = setup_short
                else:
                    chosen_setup = setup_long if setup_long.rr >= setup_short.rr else setup_short

        if save_report:
            print_single_report(instrument, indicators, chosen_setup, imoex_info)
            md_path = f"report_{ticker}.md"
            export_single_markdown(instrument, indicators, chosen_setup, imoex_info, md_path)
            console.print(f"[dim]Отчёт сохранён в файл: {md_path}[/dim]")

        return chosen_setup

    except Exception as e:
        console.print(f"[bold red][ИСКЛЮЧЕНИЕ][/bold red] Ошибка при анализе {ticker}: {e}")
        return None


def run_screener_all(client: TInvestClient, direction_pref: str = "auto"):
    """
    Пакетный скрининг по всем ликвидным акциям из DEFAULT_TICKERS.
    direction_pref: 'auto', 'long', 'short'.
    """
    mode_text = "ВСЕ (AUTO)" if direction_pref == "auto" else (
        "ТОЛЬКО ЛОНГ (LONG)" if direction_pref == "long" else "ТОЛЬКО ШОРТ (SHORT)"
    )
    console.print(f"[bold cyan]Запуск скрининга ликвидных акций Мосбиржи [{mode_text}]...[/bold cyan]")
    imoex_info = client.get_imoex_context()

    results: List[SetupResult] = []
    total = len(DEFAULT_TICKERS)

    for i, ticker in enumerate(DEFAULT_TICKERS, start=1):
        console.print(f"[dim][{i}/{total}][/dim] Анализ [bold yellow]{ticker}[/bold yellow]...", end="\r")
        setup = analyze_single_ticker(
            client=client,
            ticker=ticker,
            direction_pref=direction_pref,
            imoex_info=imoex_info,
            save_report=False
        )
        if setup:
            results.append(setup)

    # Сортировка: сначала ACTIVE по убыванию RR, затем остальные по убыванию RR
    def sort_key(s: SetupResult):
        is_active = 1 if s.status.startswith("ACTIVE") else 0
        return (is_active, s.rr)

    results.sort(key=sort_key, reverse=True)

    # Отбор лучших бумаг (TOP-3)
    active_candidates = [s for s in results if s.status.startswith("ACTIVE")]
    if len(active_candidates) >= 3:
        top_setups = active_candidates[:3]
    elif active_candidates:
        # Если активных меньше 3, добавляем лучшие по R/R из оставшихся
        remaining = [s for s in results if s not in active_candidates]
        top_setups = active_candidates + remaining[:3 - len(active_candidates)]
    else:
        top_setups = results[:3] if results else []

    print_screener_summary(results, imoex_info, top_setups, direction_pref=direction_pref)

    md_path = "screener_report.md"
    export_screener_markdown(results, imoex_info, top_setups, md_path, direction_pref=direction_pref)
    console.print(f"\n[bold green]Сводный отчёт скрининга сохранён в файл: {md_path}[/bold green]")


def interactive_mode(client: TInvestClient):
    """Интерактивный режим работы в терминале."""
    console.print(Panel(
        "[bold cyan]Интерактивный режим CLI-анализатора MOEX[/bold cyan]\n"
        "Поддерживаемые команды:\n"
        "  • [yellow]TICKER[/yellow] (например, [yellow]NVTK[/yellow], [yellow]SBER[/yellow]) — авто-анализ тикера\n"
        "  • [yellow]TICKER long[/yellow] / [yellow]TICKER short[/yellow] — принудительный лонг/шорт\n"
        "  • [yellow]all[/yellow] — пакетный скрининг по тренду\n"
        "  • [yellow]all long[/yellow] — целенаправленный отбор лонг-идей по всем акциям\n"
        "  • [yellow]all short[/yellow] — целенаправленный отбор шорт-идей по всем акциям\n"
        "Для выхода введите [bold red]q[/bold red] или [bold red]exit[/bold red].",
        border_style="cyan"
    ))

    imoex_info = client.get_imoex_context()

    while True:
        try:
            line = input("\n[scan] Введите команду: ").strip()
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                console.print("[dim]Завершение работы.[/dim]")
                break

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "all":
                direction = parts[1].lower() if len(parts) > 1 else "auto"
                if direction not in ("auto", "long", "short"):
                    console.print(f"[red]Неизвестное направление для all: {direction}. Допустимы: auto, long, short.[/red]")
                    continue
                run_screener_all(client, direction_pref=direction)
            else:
                ticker = cmd.upper()
                direction = parts[1].lower() if len(parts) > 1 else "auto"
                if direction not in ("auto", "long", "short"):
                    console.print(f"[red]Неизвестное направление: {direction}. Допустимы: auto, long, short.[/red]")
                    continue
                analyze_single_ticker(client, ticker, direction_pref=direction, imoex_info=imoex_info)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Завершение работы.[/dim]")
            break


def main():
    try:
        client = TInvestClient()
    except Exception as e:
        console.print(f"[bold red][ОШИБКА ИНИЦИАЛИЗАЦИИ][/bold red] {e}")
        sys.exit(1)

    args = sys.argv[1:]

    if not args:
        # Режим без аргументов -> интерактивный режим
        interactive_mode(client)
        return

    first_arg = args[0].lower()

    if first_arg == "all":
        # Пакетный скрининг: .\scan all [long|short|auto]
        direction = args[1].lower() if len(args) > 1 else "auto"
        if direction not in ("auto", "long", "short"):
            console.print(f"[red]Неизвестное направление для all: {direction}. Допустимы: auto, long, short.[/red]")
            sys.exit(1)
        run_screener_all(client, direction_pref=direction)
    else:
        # Анализ одного тикера: .\scan TICKER [long|short|auto]
        ticker = args[0].upper()
        direction = args[1].lower() if len(args) > 1 else "auto"
        imoex_info = client.get_imoex_context()
        analyze_single_ticker(client, ticker, direction_pref=direction, imoex_info=imoex_info)


if __name__ == "__main__":
    main()
