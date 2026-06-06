"""Provider factory."""
from __future__ import annotations

from margin_cost.providers.base import DataProvider


def get_provider(source: str = "finmind", **kwargs) -> DataProvider:
    """Return the appropriate DataProvider instance.

    Parameters
    ----------
    source : "finmind" | "goodinfo"
    kwargs : passed to the provider constructor
             e.g. token="..." for FinMind, delay_range=(3,6) for GoodInfo
    """
    if source == "goodinfo":
        from margin_cost.providers.goodinfo import GoodInfoProvider
        return GoodInfoProvider(**kwargs)

    from margin_cost.providers.finmind import FinMindProvider
    return FinMindProvider(**kwargs)
