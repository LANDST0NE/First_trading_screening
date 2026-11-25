import math
from decimal import Decimal, ROUND_HALF_UP


def get_step_decimals(step: float) -> int:
    """Определяет количество знаков после запятой у шага цены."""
    if step <= 0:
        return 2
    d = Decimal(str(step)).normalize()
    exponent = d.as_tuple().exponent
    return max(0, -exponent)


def round_to_step(price: float, step: float) -> float:
    """Округляет цену до ближайшего шага цены инструмента (min_price_increment)."""
    if step <= 0:
        return round(price, 4)
    decimals = get_step_decimals(step)
    # Используем Decimal для предотвращения проблем с точностью float
    d_price = Decimal(str(price))
    d_step = Decimal(str(step))
    steps = (d_price / d_step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    rounded = float(steps * d_step)
    return round(rounded, decimals)


def format_price(price: float, step: float) -> str:
    """Форматирует цену в строку в соответствии с шагом инструмента."""
    decimals = get_step_decimals(step)
    rounded = round_to_step(price, step)
    return f"{rounded:.{decimals}f}"


def format_rub(amount: float) -> str:
    """Форматирует сумму в рублях с пробелами между разрядами."""
    return f"{amount:,.2f} ₽".replace(",", " ")
