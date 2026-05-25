from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse


KNOWN_PLATFORMS = {
    "bsky.app": "bluesky",
    "facebook.com": "facebook",
    "instagram.com": "instagram",
    "linkedin.com": "linkedin",
    "mastodon.social": "mastodon",
    "medium.com": "medium",
    "reddit.com": "reddit",
    "substack.com": "substack",
    "t.me": "telegram",
    "telegram.org": "telegram",
    "threads.net": "threads",
    "x.com": "x",
    "twitter.com": "x",
    "tiktok.com": "tiktok",
    "truthsocial.com": "truthsocial",
    "vk.com": "vk",
    "youtube.com": "youtube",
}

SUSPICIOUS_HANDLE_HINTS = {
    "alert",
    "backup",
    "bez_cenzury",
    "breaking",
    "daily",
    "exposed",
    "leaks",
    "mirror",
    "news24",
    "official2",
    "prawda",
    "prawdziwe",
    "real",
    "rumor",
    "secret",
    "sensacja",
    "sekret",
    "stopcenzurze",
    "truth",
    "uncensored",
    "ukryte",
    "viral",
    "wolnosc",
}


@dataclass(frozen=True)
class ProfileInput:
    profile_name: str | None = None
    profile_url: str | None = None
    platform: str | None = None
    is_verified: bool | None = None
    follower_count: int | None = None
    account_age_days: int | None = None


@dataclass(frozen=True)
class ProfileFeatures:
    profile_name: str | None
    profile_url: str | None
    platform: str | None
    has_profile_name: bool
    has_profile_url: bool
    known_platform: bool
    is_verified: bool | None
    follower_count: int | None
    account_age_days: int | None
    suspicious_handle_hint: bool
    handle_from_text: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_profile_features(profile: ProfileInput | None, text: str = "") -> ProfileFeatures:
    profile = profile or ProfileInput()
    handle_from_text = _extract_handle(text)
    profile_name = _clean(profile.profile_name) or handle_from_text
    profile_url = _clean(profile.profile_url)
    platform = _clean(profile.platform) or _platform_from_url(profile_url)

    return ProfileFeatures(
        profile_name=profile_name,
        profile_url=profile_url,
        platform=platform,
        has_profile_name=bool(profile_name),
        has_profile_url=bool(profile_url),
        known_platform=platform in set(KNOWN_PLATFORMS.values()) if platform else False,
        is_verified=profile.is_verified,
        follower_count=profile.follower_count,
        account_age_days=profile.account_age_days,
        suspicious_handle_hint=_has_suspicious_handle_hint(profile_name),
        handle_from_text=handle_from_text,
    )


def _platform_from_url(url: str | None) -> str | None:
    if not url:
        return None
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    for known_domain, platform in KNOWN_PLATFORMS.items():
        if domain == known_domain or domain.endswith(f".{known_domain}"):
            return platform
    return None


def _extract_handle(text: str) -> str | None:
    match = re.search(r"(?<!\w)@[\w.\-]{3,30}", text)
    return match.group(0) if match else None


def _has_suspicious_handle_hint(profile_name: str | None) -> bool:
    if not profile_name:
        return False
    normalized = profile_name.lower()
    return any(hint in normalized for hint in SUSPICIOUS_HANDLE_HINTS)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
