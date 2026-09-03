"""Framework adapters. Adapters translate; they never contain policy."""

from .wrap import bind, current, gw_wrap

__all__ = ["bind", "current", "gw_wrap"]
