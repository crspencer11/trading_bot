from __future__ import annotations


def fixed_fractional_position_size(
    capital: float, risk_fraction: float, entry_price: float, stop_price: float
) -> float:
    """Return position quantity based on fractional capital at risk."""
    if risk_fraction <= 0 or risk_fraction > 1:
        raise ValueError("risk_fraction must be in (0, 1]")
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        raise ValueError("entry_price and stop_price cannot be equal")
    risk_budget = capital * risk_fraction
    return risk_budget / risk_per_unit
