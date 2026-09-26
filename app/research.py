from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "3d-modeling-ai/0.3 (reference research; contact via repository)"
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def _clean_metadata(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return None
    text = " ".join(value.replace("<br>", " ").split())
    return text[:1000] or None


async def research_web_references(
    query: str,
    target_dir: Path,
    *,
    max_images: int = 6,
) -> dict[str, Any]:
    """Build a small traceable reference pack from Wikimedia projects.

    This intentionally starts with public MediaWiki APIs rather than scraping arbitrary
    search-result pages. A broader provider can be plugged in later without changing
    the job/reference manifest format.
    """
    query = query.strip()
    if not query:
        raise ValueError("Research query is empty.")

    target_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        wiki_params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": "5",
            "prop": "extracts|pageimages|info",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "piprop": "thumbnail|original|name",
            "pithumbsize": "1200",
            "pilicense": "free",
        }
        wiki = (await client.get(WIKIPEDIA_API, params=wiki_params)).json()
        for page in (wiki.get("query") or {}).get("pages") or []:
            pages.append(
                {
                    "provider": "wikipedia",
                    "title": page.get("title"),
                    "url": page.get("fullurl"),
                    "extract": (page.get("extract") or "")[:4000],
                }
            )
            image = page.get("thumbnail") or page.get("original") or {}
            if image.get("source"):
                candidates.append(
                    {
                        "provider": "wikipedia",
                        "title": page.get("title"),
                        "image_url": image["source"],
                        "source_url": page.get("fullurl"),
                        "license": "free-license-filtered",
                    }
                )

        commons_params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(max(8, max_images * 2)),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": "1400",
        }
        commons = (await client.get(COMMONS_API, params=commons_params)).json()
        for page in (commons.get("query") or {}).get("pages") or []:
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            meta = info.get("extmetadata") or {}
            candidates.append(
                {
                    "provider": "wikimedia_commons",
                    "title": page.get("title"),
                    "image_url": info.get("thumburl") or info.get("url"),
                    "source_url": info.get("descriptionurl"),
                    "license": _clean_metadata(meta.get("LicenseShortName")) or "unknown",
                    "artist": _clean_metadata(meta.get("Artist")),
                    "credit": _clean_metadata(meta.get("Credit")),
                    "mime": info.get("mime"),
                }
            )

        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        references: list[dict[str, Any]] = []
        for candidate in candidates:
            image_url = candidate.get("image_url")
            if not image_url or image_url in seen_urls or len(references) >= max_images:
                continue
            seen_urls.add(image_url)
            try:
                response = await client.get(image_url)
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            data = response.content
            if not data or len(data) > 12 * 1024 * 1024:
                continue

            try:
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                with Image.open(BytesIO(data)) as image:
                    fmt = (image.format or "").upper()
                    width, height = image.size
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                continue
            extension = FORMAT_EXTENSIONS.get(fmt)
            if not extension:
                continue

            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            stored_name = f"web-{len(references) + 1:02d}-{digest[:12]}{extension}"
            (target_dir / stored_name).write_bytes(data)
            references.append(
                {
                    **candidate,
                    "stored_name": stored_name,
                    "sha256": digest,
                    "bytes": len(data),
                    "width": width,
                    "height": height,
                    "mime": response.headers.get("content-type", candidate.get("mime") or ""),
                    "researched_at": datetime.now(UTC).isoformat(),
                }
            )

    return {
        "query": query,
        "provider": "wikimedia",
        "created_at": datetime.now(UTC).isoformat(),
        "pages": pages,
        "references": references,
    }


def write_research_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
