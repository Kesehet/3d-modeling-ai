"""License-aware discovery of reusable third-party meshes.

A catalog hit is metadata, not a 3D export or an accepted model. The official
download operation requires the user's Sketchfab OAuth authorization; never
scrape source files or reuse signed download URLs.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import httpx

CATALOG_URL = "https://api.sketchfab.com/v3/models"
LICENSES = {"cc0", "by", "by-sa"}
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
WORD = re.compile(r"[a-z0-9]+")
PROMPT_PREFIXES = (
    r"^(?:please\s+)?(?:make|create|generate|build|model|render|give me|show me)\s+",
    r"^(?:a|an|the)\s+",
    r"^3d\s+(?:model|asset|object)\s+(?:of\s+)?",
    r"^(?:of|for)\s+",
)


def search_subject(prompt: str) -> str:
    """Extract the identity, retaining make/model/year without templates."""
    value = " ".join(prompt.strip().split())
    for _ in range(4):
        original = value
        for pattern in PROMPT_PREFIXES:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        if original == value:
            break
    value = re.sub(r"(?:\s+(?:as a|in)\s+(?:3d\s+)?model)\s*$", "", value, flags=re.I)
    return value.strip(" .,:;!?-")[:140]


def _words(value: str) -> list[str]:
    return WORD.findall(value.lower())


def _noai(record: dict[str, Any]) -> bool:
    if record.get("isNoAi") is True or record.get("isNoAI") is True:
        return True
    for tag in record.get("tags") or []:
        word = (tag.get("slug") or tag.get("name")) if isinstance(tag, dict) else tag
        if isinstance(word, str) and word.lower().replace("-", "") == "noai":
            return True
    return False


def rank_catalog_result(subject: str, record: dict[str, Any]) -> dict | None:
    """Conservatively reject unrelated, non-downloadable or restricted files."""
    name, uid = record.get("name"), record.get("uid")
    license_info = record.get("license")
    license_id = (
        (license_info.get("uid") or license_info.get("slug"))
        if isinstance(license_info, dict) else None
    )
    if (not isinstance(name, str) or not isinstance(uid, str)
            or not re.fullmatch(r"[0-9a-fA-F]{32}", uid)
            or not isinstance(license_id, str)
            or license_id.lower() not in LICENSES or _noai(record)
            or record.get("isDownloadable") is not True):
        return None
    target, title = _words(subject), _words(name)
    if not target or not set(target).issubset(set(title)):
        return None
    requested_years, title_years = set(YEAR.findall(subject)), set(YEAR.findall(name))
    if requested_years and title_years != requested_years:
        return None
    faces = record.get("faceCount")
    if not isinstance(faces, int) or not 100 <= faces <= 1_500_000:
        return None  # Exclude point clouds and excessive models.
    user = record.get("user")
    author = (user.get("username") or user.get("displayName")) if isinstance(user, dict) else None
    publish = record.get("publishedAt")
    recency = 0.0
    if isinstance(publish, str):
        try:
            recency = max(0.0, datetime.fromisoformat(
                publish.replace("Z", "+00:00")
            ).year - 2020) * .15
        except ValueError:
            pass
    phrase = " ".join(target) in " ".join(title)
    exact = target == title
    detail = 2.0 if 4_000 <= faces <= 250_000 else 1.0 if faces <= 600_000 else 0.0
    return {
        "provider": "sketchfab",
        "uid": uid.lower(),
        "title": name,
        "source_url": f"https://sketchfab.com/models/{uid.lower()}",
        "license": license_id.lower(),
        "creator": (author or "unknown creator")[:160],
        "face_count": faces,
        "score": round((30 if exact else 20 if phrase else 13) + detail + recency, 2),
        "attribution_required": license_id.lower() != "cc0",
        "requires_user_oauth_to_download": True,
        "quality_status": "unverified_candidate",
    }


async def search_licensed_models(
    prompt: str, *, client: httpx.AsyncClient | None = None, limit: int = 8
) -> dict:
    """Use public metadata API only. Never claim a model was imported."""
    subject = search_subject(prompt)
    if len(subject) < 2:
        return {"status": "insufficient_subject", "subject": subject, "candidates": []}
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0))
    try:
        response = await client.get(
            CATALOG_URL,
            params={"q": subject, "downloadable": "true", "count": "30"},
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            raise ValueError("Unexpected model catalog response")
        candidates: dict[str, dict] = {}
        for entry in raw:
            if isinstance(entry, dict):
                selected = rank_catalog_result(subject, entry)
                if selected:
                    candidates[selected["uid"]] = selected
        ordered = sorted(
            candidates.values(), key=lambda item: (-item["score"], item["face_count"])
        )
        return {
            "status": "candidates_need_visual_review" if ordered else "no_licensed_match",
            "subject": subject,
            "candidates": ordered[:max(1, min(limit, 12))],
            "source": "sketchfab_public_metadata",
            "download_ready": False,
            "note": "Search metadata is not an imported or visually accepted 3D model.",
        }
    finally:
        if owns_client:
            await client.aclose()
