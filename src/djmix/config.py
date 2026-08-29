"""Configuration and entitlements.

Both the occasion profiles and the tier limits are loaded from YAML rather than
written into code, so that adding an occasion or changing a plan's cap is a
config edit. The entitlement check exists now, with a single free tier, so that
Phase 3 billing swaps where a user's tier comes from and nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_DIR = Path(os.environ.get("DJMIX_CONFIG_DIR", REPO_ROOT / "config"))


class ConfigError(RuntimeError):
    pass


class EntitlementError(RuntimeError):
    """Raised when a request exceeds the caller's tier limits."""


@lru_cache(maxsize=8)
def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    return yaml.safe_load(path.read_text()) or {}


@dataclass(frozen=True)
class OccasionProfile:
    name: str
    arc: list[float]
    bpm_band: tuple[float, float]
    moods: dict[str, float]

    def target_energy(self, slot: int, total: int) -> float:
        """Sample the arc at slot `slot` of `total`, linearly interpolated."""
        if total <= 1:
            return float(self.arc[len(self.arc) // 2])
        position = slot / (total - 1) * (len(self.arc) - 1)
        low = int(position)
        high = min(low + 1, len(self.arc) - 1)
        frac = position - low
        return float(self.arc[low] * (1 - frac) + self.arc[high] * frac)


def occasions_config() -> dict[str, Any]:
    return _load_yaml("occasions.yaml")


def available_occasions() -> list[str]:
    return sorted(occasions_config().get("occasions", {}))


def planner_weights() -> dict[str, float]:
    return dict(occasions_config().get("weights", {}))


def get_occasion(name: str | None) -> OccasionProfile:
    config = occasions_config()
    occasions = config.get("occasions", {})
    default = config.get("default_occasion", "road trip")
    key = (name or default).strip().lower()
    if key not in occasions:
        key = default
    block = occasions[key]
    return OccasionProfile(
        name=key,
        arc=[float(v) for v in block["arc"]],
        bpm_band=(float(block["bpm_band"][0]), float(block["bpm_band"][1])),
        moods={k: float(v) for k, v in block.get("moods", {}).items()},
    )


def occasion_from_prompt(prompt: str) -> str:
    """Map free text to an occasion by keyword.

    Deliberately crude: this is the offline fallback. Real natural-language
    interpretation is the LLM planner's job, and the difference between the two
    is the clearest demonstration of what the LLM layer adds.
    """
    config = occasions_config()
    text = prompt.lower()
    best, best_hits = None, 0
    for occasion, keywords in config.get("prompt_keywords", {}).items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best_hits:
            best, best_hits = occasion, hits
    return best or config.get("default_occasion", "road trip")


@dataclass(frozen=True)
class Tier:
    name: str
    max_mix_minutes: float
    max_mixes_per_month: int | None
    max_tracks_per_mix: int
    max_library_tracks: int
    priority_processing: bool


def get_tier(name: str | None = None) -> Tier:
    config = _load_yaml("entitlements.yaml")
    tiers = config.get("tiers", {})
    key = (name or os.environ.get("DJMIX_TIER") or config.get("default_tier", "free")).lower()
    if key not in tiers:
        raise EntitlementError(f"unknown tier {key!r}; known tiers: {sorted(tiers)}")
    block = tiers[key]
    return Tier(
        name=key,
        max_mix_minutes=float(block["max_mix_minutes"]),
        max_mixes_per_month=block.get("max_mixes_per_month"),
        max_tracks_per_mix=int(block["max_tracks_per_mix"]),
        max_library_tracks=int(block["max_library_tracks"]),
        priority_processing=bool(block.get("priority_processing", False)),
    )


def check_mix_request(tier: Tier, n_tracks: int, target_minutes: float | None) -> None:
    if n_tracks > tier.max_tracks_per_mix:
        raise EntitlementError(
            f"{n_tracks} tracks exceeds the {tier.name} tier limit of "
            f"{tier.max_tracks_per_mix} per mix"
        )
    if target_minutes is not None and target_minutes > tier.max_mix_minutes:
        raise EntitlementError(
            f"a {target_minutes:.0f}-minute mix exceeds the {tier.name} tier limit of "
            f"{tier.max_mix_minutes:.0f} minutes"
        )
