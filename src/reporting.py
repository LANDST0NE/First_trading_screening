import os
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.indicators import TFIndicators
from src.strategy import SetupResult
from src.client import InstrumentInfo
from src.utils import format_price, format_rub

console = Console()


def _get_trade_type_style(trade_type: str) -> str:
    if "Following" in trade_type:
        return "bold green"
    elif "Retest Confirmed" in trade_type:
        return "bold green"
    elif "Breakout Confirmed" in trade_type or "Confirmed Reversal" in trade_type:
        return "bold magenta"
    elif "Candidate" in trade_type:
        return "bold cyan"
    elif "MIXED" in trade_type:
        return "bold red"
    return "yellow"


def print_single_report(
    instrument: InstrumentInfo,
    indicators: Dict[str, TFIndicators],
    setup: SetupResult,
    imoex_info: Optional[Dict[str, Any]] = None
):
    """Выводит подробный торговый отчёт по инструменту в консоль."""
    step = instrument.min_price_increment

    # 1. Заголовок
    imoex_str = "N/A"
    if imoex_info and imoex_info.get("price", 0) > 0:
        chg = imoex_info["change_pct"]
        sign = "+" if chg > 0 else ""
        imoex_str = f"{imoex_info['price']:.2f} ({sign}{chg:.2f}%) [{imoex_info['trend']}]"

    header_text = Text()
    header_text.append(f"{instrument.name} ({instrument.ticker})", style="bold cyan")
    header_text.append(f"  |  Цена: {format_price(setup.current_price, step)} ₽")
    header_text.append(f"  |  Лот: {instrument.lot} шт.")
    short_style = "green" if instrument.short_enabled_flag else "bold red"
    short_text = "Доступен" if instrument.short_enabled_flag else "Запрещён"
    header_text.append(f"  |  Шорт у брокера: {short_text}", style=short_style)
    trend_style = "bold green" if setup.senior_trend == "BULLISH" else ("bold red" if setup.senior_trend == "BEARISH" else "yellow")
    header_text.append(f"  |  Тренд 1D/4H: {setup.senior_trend}", style=trend_style)
    header_text.append(f"  |  Фон IMOEX: {imoex_str}", style="magenta")

    console.print()
    console.print(Panel(header_text, title="АНАЛИЗ АКЦИИ", border_style="cyan"))

    # 2. Таблица индикаторов с eVPVR
    table_ind = Table(
        title="Технические индикаторы и eVPVR (Estimated Volume Profile)",
        caption="* eVPVR — приближённый профиль объёма, рассчитанный распределением OHLCV-объёма по корзинам.",
        caption_style="dim italic",
        header_style="bold yellow"
    )
    table_ind.add_column("ТФ", justify="center")
    table_ind.add_column("EMA 20", justify="right")
    table_ind.add_column("EMA 50", justify="right")
    table_ind.add_column("ATR(14)", justify="right")
    table_ind.add_column("RSI(14)", justify="right")
    table_ind.add_column("ePOC(80)", justify="right")
    table_ind.add_column("eVAL 70%", justify="right")
    table_ind.add_column("eVAH 70%", justify="right")
    table_ind.add_column("Swing Low", justify="right")
    table_ind.add_column("Swing High", justify="right")

    for tf in ["1H", "4H", "1D"]:
        if tf in indicators:
            ind = indicators[tf]
            rsi_style = "bold red" if ind.rsi14 >= 70 else ("bold green" if ind.rsi14 <= 30 else "white")
            table_ind.add_row(
                tf,
                format_price(ind.ema20, step),
                format_price(ind.ema50, step),
                format_price(ind.atr14, step),
                Text(f"{ind.rsi14:.1f}", style=rsi_style),
                format_price(ind.poc80, step),
                format_price(ind.val80, step),
                format_price(ind.vah80, step),
                format_price(ind.swing_low_20, step),
                format_price(ind.swing_high_20, step),
            )

    console.print(table_ind)

    # 3. Блок: Качество сетапа и триггеры подтверждения
    ph_str = f"{format_price(setup.last_pivot_high, step)} ₽" if setup.last_pivot_high else "Не сформирован"
    pl_str = f"{format_price(setup.last_pivot_low, step)} ₽" if setup.last_pivot_low else "Не сформирован"

    quality_table = Table(show_header=False, box=None, padding=(0, 2))
    quality_table.add_column("Параметр", style="bold")
    quality_table.add_column("Значение")

    quality_table.add_row("Старший тренд (1D/4H):", Text(setup.senior_trend, style=trend_style))
    quality_table.add_row("Часовая структура (1H):", Text(setup.market_structure_1h, style="cyan"))
    quality_table.add_row("Последний Pivot High (L=3, R=2):", ph_str)
    quality_table.add_row("Последний Pivot Low (L=3, R=2):", pl_str)

    if setup.breakout_threshold:
        dir_sign = "+" if setup.direction == "LONG" else "-"
        quality_table.add_row(
            f"Порог пробоя (с буфером 0.1 ATR):",
            f"{format_price(setup.breakout_threshold, step)} ₽  (Pivot {dir_sign} 0.10 ATR)"
        )

    if setup.breakout_status and setup.breakout_status != "None":
        bo_style = "bold green" if "Retest" in setup.breakout_status else "bold magenta"
        quality_table.add_row("Статус пробоя / ретеста:", Text(setup.breakout_status, style=bo_style))

    quality_table.add_row(
        "Классификация сделки:",
        Text(setup.trade_type, style=_get_trade_type_style(setup.trade_type))
    )

    if setup.reversal_diagnostics:
        quality_table.add_row("Диагностика кандидата:", Text(setup.reversal_diagnostics, style="cyan"))

    quality_table.add_row("Условие подтверждения:", Text(setup.confirmation_trigger, style="bold green"))
    quality_table.add_row("Условие отмены сценария:", Text(setup.invalidation_trigger, style="bold red"))

    console.print(Panel(quality_table, title="СТРУКТУРНЫЙ АНАЛИЗ И ВАЛИДАЦИЯ СЕТАПА", border_style="blue"))

    # 4. Торговый билет / Сетап
    status_style = "bold green" if "ACTIVE (Strong)" in setup.status else (
        "bold yellow" if "ACTIVE (Standard)" in setup.status else "bold red"
    )

    ticket_table = Table(show_header=False, box=None, padding=(0, 2))
    ticket_table.add_column("Параметр", style="bold")
    ticket_table.add_column("Значение")

    ticket_table.add_row("Направление сделки:", Text(setup.direction, style="bold cyan"))
    ticket_table.add_row("Статус сетапа:", Text(setup.status, style=status_style))

    if setup.warning:
        ticket_table.add_row("Предупреждение:", Text(setup.warning, style="bold red on yellow"))

    if setup.status.startswith("SKIP"):
        ticket_table.add_row("Причина пропуска:", Text(setup.skip_reason or "R/R ниже порога", style="red"))
        if setup.ideal_entry and setup.ideal_entry > 0:
            ticket_table.add_row(
                "Идеальный вход (R/R = 1.8):",
                Text(f"{format_price(setup.ideal_entry, step)} ₽ (отложенный лимит)", style="bold yellow")
            )
        console.print(Panel(ticket_table, title="РЕЗУЛЬТАТ АНАЛИЗА", border_style="red"))
        return

    # Для ACTIVE выводим подробные уровни и объёмы
    ticket_table.add_row("---", "---")
    ticket_table.add_row("Вход 1 (50% позиции):", f"{format_price(setup.entry_1, step)} ₽  [{setup.lots_entry_1} лот.]")
    ticket_table.add_row(
        "Вход 2 (50% добор):",
        f"{format_price(setup.entry_2, step)} ₽  [{setup.lots_entry_2} лот.]  (Опора: [cyan]{setup.entry_2_source}[/cyan])"
    )
    ticket_table.add_row("Средняя цена входа:", f"{format_price(setup.entry_avg, step)} ₽")
    ticket_table.add_row("Якорь стопа (Anchor):", f"{format_price(setup.anchor_stop, step)} ₽")
    ticket_table.add_row("Стоп-лосс (Stop):", Text(f"{format_price(setup.stop, step)} ₽", style="bold red"))
    ticket_table.add_row("Тейк-профит 1 (Target 1):", f"{format_price(setup.take_1, step)} ₽  [{setup.lots_take_1} лот.]")
    ticket_table.add_row("Тейк-профит 2 (Target 2):", f"{format_price(setup.take_2, step)} ₽  [{setup.lots_take_2} лот.]")

    ticket_table.add_row("---", "---")
    ticket_table.add_row("Риск на акцию:", f"{format_price(setup.risk_per_share, step)} ₽")
    ticket_table.add_row("Прибыль на акцию (Тейк 1):", f"{format_price(setup.reward_t1_per_share, step)} ₽")
    ticket_table.add_row("Прибыль на акцию (Тейк 2):", f"{format_price(setup.reward_t2_per_share, step)} ₽")
    ticket_table.add_row("Средневзвеш. R/R:", Text(f"{setup.rr:.2f}", style="bold green"))

    ticket_table.add_row("---", "---")
    ticket_table.add_row("Всего лотов / акций:", f"{setup.total_lots} лот. ({setup.total_lots * setup.lot_size} шт.)")
    ticket_table.add_row("Макс. убыток на стопе:", Text(f"-{format_rub(setup.actual_loss_rub)}", style="bold red"))
    ticket_table.add_row("Прибыль на Тейк 1:", Text(f"+{format_rub(setup.profit_take_1_rub)}", style="green"))
    ticket_table.add_row("Прибыль на Тейк 2:", Text(f"+{format_rub(setup.profit_take_2_rub)}", style="green"))
    ticket_table.add_row("Итого чистая прибыль:", Text(f"+{format_rub(setup.total_profit_rub)}", style="bold green"))

    if setup.total_lots == 1:
        ticket_table.add_row(
            "Особенность позиции (1 лот):",
            "100% позиции закрывается на Тейк 2; при достижении Тейк 1 стоп переносится в безубыток."
        )

    console.print(Panel(ticket_table, title="ТОРГОВЫЙ БИЛЕТ", border_style="green"))


def export_single_markdown(
    instrument: InstrumentInfo,
    indicators: Dict[str, TFIndicators],
    setup: SetupResult,
    imoex_info: Optional[Dict[str, Any]],
    filepath: str
):
    """Сохраняет отчёт по тикеру в файл report_<TICKER>.md в Markdown."""
    step = instrument.min_price_increment
    imoex_str = "N/A"
    if imoex_info and imoex_info.get("price", 0) > 0:
        chg = imoex_info["change_pct"]
        sign = "+" if chg > 0 else ""
        imoex_str = f"{imoex_info['price']:.2f} ({sign}{chg:.2f}%) [{imoex_info['trend']}]"

    ph_str = f"{format_price(setup.last_pivot_high, step)} ₽" if setup.last_pivot_high else "Не сформирован"
    pl_str = f"{format_price(setup.last_pivot_low, step)} ₽" if setup.last_pivot_low else "Не сформирован"

    md = []
    md.append(f"# Торговый отчёт по акции {instrument.ticker} ({instrument.name})")
    md.append(f"- **Текущая цена:** {format_price(setup.current_price, step)} ₽")
    md.append(f"- **Размер лота:** {instrument.lot} шт.")
    md.append(f"- **Шорт у брокера:** {'Доступен' if instrument.short_enabled_flag else 'Запрещён'}")
    md.append(f"- **Господствующий тренд (1D/4H):** `{setup.senior_trend}`")
    md.append(f"- **Фон индекса IMOEX:** {imoex_str}")
    md.append("")

    # Структурный анализ рынка и валидация
    md.append("## Структурный анализ рынка (1H Structure & Pivots)")
    md.append(f"- **Классификация сделки:** `{setup.trade_type}`")
    md.append(f"- **Часовая структура:** `{setup.market_structure_1h}`")
    md.append(f"- **Последний подтверждённый Pivot High (L=3, R=2):** `{ph_str}`")
    md.append(f"- **Последний подтверждённый Pivot Low (L=3, R=2):** `{pl_str}`")
    if setup.breakout_threshold:
        dir_sign = "+" if setup.direction == "LONG" else "-"
        md.append(f"- **Порог пробоя (с буфером 0.1 ATR):** `{format_price(setup.breakout_threshold, step)} ₽` (Pivot {dir_sign} 0.10 ATR)")
    if setup.breakout_status and setup.breakout_status != "None":
        md.append(f"- **Статус пробоя / ретеста:** `{setup.breakout_status}`")
    if setup.reversal_diagnostics:
        md.append(f"- **Диагностика кандидата в разворот:** `{setup.reversal_diagnostics}`")
    md.append(f"- **Условие подтверждения сценария:** {setup.confirmation_trigger}")
    md.append(f"- **Условие отмены сценария (Invalidation):** `{setup.invalidation_trigger}`")
    md.append("")

    # Таблица индикаторов
    md.append("## Технические индикаторы и eVPVR")
    md.append("> *Примечание:* eVPVR (Estimated Volume Profile) — оценочное приближение профиля объёма, полученное распределением объёма OHLCV-свечей по ценовым корзинам.")
    md.append("")
    md.append("| ТФ | EMA 20 | EMA 50 | ATR(14) | RSI(14) | ePOC(80) | eVAL 70% | eVAH 70% | Swing Low | Swing High |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for tf in ["1H", "4H", "1D"]:
        if tf in indicators:
            ind = indicators[tf]
            md.append(
                f"| {tf} | {format_price(ind.ema20, step)} | {format_price(ind.ema50, step)} | "
                f"{format_price(ind.atr14, step)} | {ind.rsi14:.1f} | {format_price(ind.poc80, step)} | "
                f"{format_price(ind.val80, step)} | {format_price(ind.vah80, step)} | "
                f"{format_price(ind.swing_low_20, step)} | {format_price(ind.swing_high_20, step)} |"
            )
    md.append("")

    # Сетап
    md.append(f"## Торговый план ({setup.direction})")
    md.append(f"- **Статус сетапа:** `{setup.status}`")
    if setup.warning:
        md.append(f"- **Предупреждение:** `{setup.warning}`")

    if setup.status.startswith("SKIP"):
        md.append(f"- **Причина пропуска:** {setup.skip_reason or 'R/R ниже 1.80'}")
        if setup.ideal_entry and setup.ideal_entry > 0:
            md.append(f"- **Идеальная цена лимитного входа (R/R = 1.8):** `{format_price(setup.ideal_entry, step)} ₽`")
    else:
        md.append("")
        md.append("### Параметры сделки и уровни")
        md.append(f"- **Вход 1 (50%):** `{format_price(setup.entry_1, step)} ₽` ({setup.lots_entry_1} лот.)")
        md.append(f"- **Вход 2 (50% добор):** `{format_price(setup.entry_2, step)} ₽` ({setup.lots_entry_2} лот.) [Опора: {setup.entry_2_source}]")
        md.append(f"- **Средняя цена входа:** `{format_price(setup.entry_avg, step)} ₽`")
        md.append(f"- **Якорь стопа:** `{format_price(setup.anchor_stop, step)} ₽`")
        md.append(f"- **Стоп-лосс:** `{format_price(setup.stop, step)} ₽`")
        md.append(f"- **Тейк-профит 1:** `{format_price(setup.take_1, step)} ₽` ({setup.lots_take_1} лот.)")
        md.append(f"- **Тейк-профит 2:** `{format_price(setup.take_2, step)} ₽` ({setup.lots_take_2} лот.)")
        md.append(f"- **Риск на акцию:** `{format_price(setup.risk_per_share, step)} ₽`")
        md.append(f"- **Прибыль на акцию (Тейк 1):** `{format_price(setup.reward_t1_per_share, step)} ₽`")
        md.append(f"- **Прибыль на акцию (Тейк 2):** `{format_price(setup.reward_t2_per_share, step)} ₽`")
        md.append(f"- **Средневзвешенный R/R (Reward / Risk):** `{setup.rr:.2f}`")
        md.append("")
        md.append("### Расчёт объёма и финансовый результат (комиссия 0.05% учтена)")
        md.append(f"- **Всего лотов:** `{setup.total_lots}` ({setup.total_lots * setup.lot_size} акций)")
        md.append(f"- **Риск при срабатывании стопа:** `-{format_rub(setup.actual_loss_rub)}`")
        md.append(f"- **Прибыль на Тейк 1:** `+{format_rub(setup.profit_take_1_rub)}`")
        md.append(f"- **Прибыль на Тейк 2:** `+{format_rub(setup.profit_take_2_rub)}`")
        md.append(f"- **Итого чистая прибыль:** `+{format_rub(setup.total_profit_rub)}`")
        if setup.total_lots == 1:
            md.append("- *Примечание:* Для 1 лота 100% позиции удерживается до Тейк 2, на Тейк 1 стоп передвигается в безубыток.")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def print_screener_summary(
    results: List[SetupResult],
    imoex_info: Optional[Dict[str, Any]],
    top_setups: List[SetupResult],
    direction_pref: str = "auto"
):
    """Выводит сводную таблицу скринера и TOP-3 бумаги дня с полными билетами."""
    console.print()
    imoex_str = "N/A"
    if imoex_info and imoex_info.get("price", 0) > 0:
        chg = imoex_info["change_pct"]
        sign = "+" if chg > 0 else ""
        imoex_str = f"{imoex_info['price']:.2f} ({sign}{chg:.2f}%) [{imoex_info['trend']}]"

    mode_title = "ВСЕ НАПРАВЛЕНИЯ (AUTO)" if direction_pref == "auto" else (
        "ТОЛЬКО ЛОНГ (LONG)" if direction_pref == "long" else "ТОЛЬКО ШОРТ (SHORT)"
    )
    console.print(Panel(f"РЫНОЧНЫЙ ФОН ИНДЕКСА IMOEX: {imoex_str}  |  РЕЖИМ: {mode_title}", style="bold magenta"))

    table = Table(title="Результаты скрининга акций Мосбиржи", header_style="bold cyan")
    table.add_column("Тикер", justify="center")
    table.add_column("Тип сделки", justify="center")
    table.add_column("Тренд", justify="center")
    table.add_column("Напр.", justify="center")
    table.add_column("Текущая цена", justify="right")
    table.add_column("Вход (avg)", justify="right")
    table.add_column("Стоп", justify="right")
    table.add_column("Тейк 1", justify="right")
    table.add_column("Тейк 2", justify="right")
    table.add_column("R/R", justify="right")
    table.add_column("Статус", justify="center")
    table.add_column("Прибыль / Примечание", justify="left")

    for s in results:
        step = s.step
        r_style = "bold green" if "ACTIVE (Strong)" in s.status else (
            "bold yellow" if "ACTIVE (Standard)" in s.status else "red"
        )
        t_style = "green" if s.senior_trend == "BULLISH" else ("red" if s.senior_trend == "BEARISH" else "yellow")
        note = f"+{format_rub(s.total_profit_rub)}" if s.total_lots > 0 else (s.skip_reason or "-")
        table.add_row(
            s.ticker,
            Text(s.trade_type, style=_get_trade_type_style(s.trade_type)),
            Text(s.senior_trend, style=t_style),
            s.direction,
            format_price(s.current_price, step),
            format_price(s.entry_avg, step) if s.entry_avg > 0 else "-",
            format_price(s.stop, step) if s.stop > 0 else "-",
            format_price(s.take_1, step) if s.take_1 > 0 else "-",
            format_price(s.take_2, step) if s.take_2 > 0 else "-",
            f"{s.rr:.2f}" if s.rr > 0 else "-",
            Text(s.status, style=r_style),
            note
        )

    console.print(table)

    # Вывод карточек TOP-3
    if top_setups:
        medals = ["🥇 TOP-1", "🥈 TOP-2", "🥉 TOP-3"]
        console.print(f"\n[bold green]═══ ЛУЧШИЕ БУМАГИ ДНЯ (TOP-{len(top_setups)}) ═══[/bold green]")
        for i, s in enumerate(top_setups):
            medal = medals[i] if i < len(medals) else f"TOP-{i+1}"
            border_col = "bold green" if i == 0 else ("cyan" if i == 1 else "yellow")

            warn_line = f"\n[bold red on yellow]Предупреждение: {s.warning}[/bold red on yellow]" if s.warning else ""
            triggers_line = (
                f"\n[bold blue]Часовая структура:[/] {s.market_structure_1h}\n"
                f"[bold green]Подтверждение:[/] {s.confirmation_trigger}\n"
                f"[bold red]Отмена сценария:[/] {s.invalidation_trigger}"
            )

            if s.total_lots > 0:
                card_text = (
                    f"[bold]{medal}: {s.ticker} ({s.direction})[/bold]  |  Тип: [{_get_trade_type_style(s.trade_type)}]{s.trade_type}[/]  |  Тренд 1D/4H: {s.senior_trend}\n"
                    f"R/R (средневзвеш.): [bold green]{s.rr:.2f}[/bold green]  |  Статус: {s.status}{warn_line}{triggers_line}\n"
                    f"Вход 1 (50%): {format_price(s.entry_1, s.step)} ₽ [{s.lots_entry_1} лот.]  |  "
                    f"Вход 2 (50% добор): {format_price(s.entry_2, s.step)} ₽ [{s.lots_entry_2} лот.]  (Опора: [cyan]{s.entry_2_source}[/cyan])\n"
                    f"Средний вход: {format_price(s.entry_avg, s.step)} ₽  |  "
                    f"Стоп-лосс: [bold red]{format_price(s.stop, s.step)} ₽[/bold red] (риск: -{format_rub(s.actual_loss_rub)})\n"
                    f"Тейк 1: [green]{format_price(s.take_1, s.step)} ₽[/green] [{s.lots_take_1} лот.]  |  "
                    f"Тейк 2: [bold green]{format_price(s.take_2, s.step)} ₽[/bold green] [{s.lots_take_2} лот.]\n"
                    f"Всего лотов: {s.total_lots} ({s.total_lots * s.lot_size} шт.)  |  "
                    f"Ожидаемая чистая прибыль: [bold green]+{format_rub(s.total_profit_rub)}[/bold green]"
                )
            else:
                ideal_str = f"  |  Идеальный вход: {format_price(s.ideal_entry, s.step)} ₽" if s.ideal_entry else ""
                card_text = (
                    f"[bold]{medal}: {s.ticker} ({s.direction})[/bold]  |  Тип: [{_get_trade_type_style(s.trade_type)}]{s.trade_type}[/]  |  Тренд 1D/4H: {s.senior_trend}\n"
                    f"R/R: {s.rr:.2f}  |  Статус: [red]{s.status}[/red]{warn_line}{triggers_line}\n"
                    f"Причина пропуска: {s.skip_reason or 'R/R ниже порога'}{ideal_str}"
                )
            console.print(Panel(card_text, border_style=border_col))
    else:
        console.print(Panel("[yellow]На текущий момент нет активных сетапов с R/R >= 1.80.[/yellow]", border_style="yellow"))


def export_screener_markdown(
    results: List[SetupResult],
    imoex_info: Optional[Dict[str, Any]],
    top_setups: List[SetupResult],
    filepath: str,
    direction_pref: str = "auto"
):
    """Сохраняет результат скрининга в screener_report.md в Markdown."""
    imoex_str = "N/A"
    if imoex_info and imoex_info.get("price", 0) > 0:
        chg = imoex_info["change_pct"]
        sign = "+" if chg > 0 else ""
        imoex_str = f"{imoex_info['price']:.2f} ({sign}{chg:.2f}%) [{imoex_info['trend']}]"

    mode_title = "Все направления (Auto)" if direction_pref == "auto" else (
        "Только лонг (Long)" if direction_pref == "long" else "Только шорт (Short)"
    )

    md = []
    md.append("# Сводный отчёт скринера сделок (MOEX TQBR)")
    md.append(f"- **Режим отбора:** `{mode_title}`")
    md.append(f"- **Рыночный фон (Индекс IMOEX):** `{imoex_str}`")
    md.append("")

    # Статистика по типам сделок
    tf_count = sum(1 for s in results if s.trade_type == "Trend Following" and s.total_lots > 0)
    cr_count = sum(1 for s in results if s.trade_type == "Confirmed Reversal" and s.total_lots > 0)
    rc_count = sum(1 for s in results if s.trade_type == "Reversal Candidate" and s.total_lots > 0)
    ct_count = sum(1 for s in results if s.trade_type == "Counter Trend" and s.total_lots > 0)
    mx_count = sum(1 for s in results if "MIXED" in s.trade_type)
    md.append("## Статистика активных сделок по режимам структуры")
    md.append(f"- 📈 **Trend Following (по старшему тренду):** {tf_count}")
    md.append(f"- 🚀 **Confirmed Reversal (подтверждённый слом структуры):** {cr_count}")
    md.append(f"- 🔄 **Reversal Candidate (кандидат в разворот 1H):** {rc_count}")
    md.append(f"- ⚡ **Counter Trend (контртренд):** {ct_count}")
    if mx_count > 0:
        md.append(f"- ⚠️ **MIXED / NO TRADE (рынок в распиле):** {mx_count}")
    md.append("")

    if top_setups:
        medals = ["🥇 TOP-1", "🥈 TOP-2", "🥉 TOP-3"]
        md.append(f"## 🏆 ЛУЧШИЕ БУМАГИ ДНЯ (TOP-{len(top_setups)})")
        for i, s in enumerate(top_setups):
            medal = medals[i] if i < len(medals) else f"TOP-{i+1}"
            step = s.step
            ph_str = f"{format_price(s.last_pivot_high, step)} ₽" if s.last_pivot_high else "—"
            pl_str = f"{format_price(s.last_pivot_low, step)} ₽" if s.last_pivot_low else "—"

            md.append(f"### {medal}: {s.ticker} ({s.direction})")
            md.append(f"- **Классификация сделки:** `{s.trade_type}`")
            md.append(f"- **Господствующий тренд (1D/4H):** `{s.senior_trend}`")
            md.append(f"- **Часовая структура:** `{s.market_structure_1h}`")
            md.append(f"- **Подтверждённый Pivot High:** `{ph_str}` | **Pivot Low:** `{pl_str}`")
            md.append(f"- **Условие подтверждения:** {s.confirmation_trigger}")
            md.append(f"- **Условие отмены сценария:** `{s.invalidation_trigger}`")
            md.append(f"- **Статус:** `{s.status}`")
            if s.warning:
                md.append(f"- **Предупреждение:** `{s.warning}`")
            md.append(f"- **Средневзвешенный R/R:** `{s.rr:.2f}`")
            md.append(f"- **Текущая цена:** `{format_price(s.current_price, step)} ₽`")
            if s.total_lots > 0:
                md.append(f"- **Вход 1 (50%):** `{format_price(s.entry_1, step)} ₽` ({s.lots_entry_1} лот.)")
                md.append(f"- **Вход 2 (50% добор):** `{format_price(s.entry_2, step)} ₽` ({s.lots_entry_2} лот.) [Опора: {s.entry_2_source}]")
                md.append(f"- **Средний вход:** `{format_price(s.entry_avg, step)} ₽`")
                md.append(f"- **Стоп-лосс:** `{format_price(s.stop, step)} ₽`")
                md.append(f"- **Тейк 1:** `{format_price(s.take_1, step)} ₽` ({s.lots_take_1} лот.)")
                md.append(f"- **Тейк 2:** `{format_price(s.take_2, step)} ₽` ({s.lots_take_2} лот.)")
                md.append(f"- **Всего лотов:** `{s.total_lots}` ({s.total_lots * s.lot_size} шт.)")
                md.append(f"- **Риск (стоп):** `-{format_rub(s.actual_loss_rub)}`")
                md.append(f"- **Ожидаемая чистая прибыль:** `+{format_rub(s.total_profit_rub)}`")
            else:
                md.append(f"- **Причина пропуска:** {s.skip_reason or 'R/R ниже порога'}")
                if s.ideal_entry:
                    md.append(f"- **Идеальный вход:** `{format_price(s.ideal_entry, step)} ₽`")
            md.append("")
    else:
        md.append("## 🏆 Лучшие бумаги дня: Нет сетапов с R/R >= 1.80 на текущий момент\n")

    md.append("## Таблица всех проверенных бумаг")
    md.append("| Тикер | Тип сделки | Тренд 1D/4H | Напр. | Текущая цена | Вход (avg) | Стоп | Тейк 1 | Тейк 2 | R/R | Статус | Прибыль / Причина |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for s in results:
        step = s.step
        p_curr = format_price(s.current_price, step)
        p_avg = format_price(s.entry_avg, step) if s.entry_avg > 0 else "-"
        p_stop = format_price(s.stop, step) if s.stop > 0 else "-"
        p_t1 = format_price(s.take_1, step) if s.take_1 > 0 else "-"
        p_t2 = format_price(s.take_2, step) if s.take_2 > 0 else "-"
        rr_str = f"{s.rr:.2f}" if s.rr > 0 else "-"
        note = f"+{format_rub(s.total_profit_rub)}" if s.total_lots > 0 else (s.skip_reason or "-")
        md.append(f"| {s.ticker} | {s.trade_type} | {s.senior_trend} | {s.direction} | {p_curr} | {p_avg} | {p_stop} | {p_t1} | {p_t2} | {rr_str} | {s.status} | {note} |")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
