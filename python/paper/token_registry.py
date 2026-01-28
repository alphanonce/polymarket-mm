"""
Token ID Registry

Maps Polymarket token IDs to market slugs and outcomes.
Token IDs are 66-character hex strings (0x...) that identify specific tokens.
Each market has two tokens: one for UP outcome, one for DOWN outcome.
"""

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class TokenMapping:
    """Mapping from token ID to market metadata."""

    token_id: str
    slug: str
    outcome: str  # "up" or "down"
    asset: str  # e.g., "btc", "eth"
    timeframe: str  # e.g., "15m", "1h"


class TokenIdRegistry:
    """
    Registry for mapping token IDs to market slugs and outcomes.

    Token IDs from SHM are 66-char hex strings that don't contain
    human-readable information. This registry maintains the mapping
    to proper market slugs for logging, storage, and orderbook lookup.
    """

    def __init__(self) -> None:
        self._mappings: dict[str, TokenMapping] = {}
        self._logger = logger.bind(component="token_registry")

    def register(
        self,
        token_id: str,
        slug: str,
        outcome: str,
        asset: str = "",
        timeframe: str = "",
    ) -> None:
        """
        Register a token ID mapping.

        Args:
            token_id: The 66-char token ID (e.g., 0x123...)
            slug: Market slug (e.g., btc-updown-15m-1768665600)
            outcome: "up" or "down"
            asset: Asset symbol (e.g., "btc")
            timeframe: Market timeframe (e.g., "15m")
        """
        mapping = TokenMapping(
            token_id=token_id,
            slug=slug,
            outcome=outcome,
            asset=asset,
            timeframe=timeframe,
        )
        self._mappings[token_id] = mapping
        self._logger.debug(
            "Registered token mapping",
            token_id=token_id[:16] + "...",
            slug=slug,
            outcome=outcome,
        )

    def register_from_market_info(self, market_info: dict[str, Any]) -> None:
        """
        Register token mappings from a market info dict.

        Expected market_info structure:
        {
            "slug": "btc-updown-15m-1768665600",
            "asset": "btc",
            "timeframe": "15m",
            "up_token_id": "0x...",
            "down_token_id": "0x...",
        }
        """
        slug = market_info.get("slug", "")
        asset = market_info.get("asset", "")
        timeframe = market_info.get("timeframe", "")

        up_token = market_info.get("up_token_id")
        down_token = market_info.get("down_token_id")

        if up_token:
            self.register(up_token, slug, "up", asset, timeframe)

        if down_token:
            self.register(down_token, slug, "down", asset, timeframe)

    def get_mapping(self, token_id: str) -> TokenMapping | None:
        """Get the full mapping for a token ID."""
        return self._mappings.get(token_id)

    def get_slug(self, token_id: str) -> str | None:
        """Get the market slug for a token ID, or None if not registered."""
        mapping = self._mappings.get(token_id)
        return mapping.slug if mapping else None

    def get_slug_or_fallback(self, token_id: str, prefix: str = "unknown") -> str:
        """
        Get the market slug for a token ID, or a fallback string.

        Args:
            token_id: The token ID to look up
            prefix: Prefix for the fallback string

        Returns:
            The slug if registered, otherwise "{prefix}-{token_id[:8]}"
        """
        mapping = self._mappings.get(token_id)
        if mapping:
            return mapping.slug
        return f"{prefix}-{token_id[:8]}"

    def get_outcome(self, token_id: str) -> str | None:
        """Get the outcome ("up" or "down") for a token ID, or None if not registered."""
        mapping = self._mappings.get(token_id)
        return mapping.outcome if mapping else None

    def get_asset(self, token_id: str) -> str | None:
        """Get the asset for a token ID, or None if not registered."""
        mapping = self._mappings.get(token_id)
        return mapping.asset if mapping else None

    def get_timeframe(self, token_id: str) -> str | None:
        """Get the timeframe for a token ID, or None if not registered."""
        mapping = self._mappings.get(token_id)
        return mapping.timeframe if mapping else None

    def is_registered(self, token_id: str) -> bool:
        """Check if a token ID is registered."""
        return token_id in self._mappings

    def clear(self) -> None:
        """Clear all registrations."""
        self._mappings.clear()
        self._logger.debug("Registry cleared")

    def __len__(self) -> int:
        return len(self._mappings)

    def __contains__(self, token_id: str) -> bool:
        return token_id in self._mappings


# Global singleton instance
_global_registry: TokenIdRegistry | None = None


def get_global_registry() -> TokenIdRegistry:
    """Get the global token ID registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = TokenIdRegistry()
    return _global_registry
