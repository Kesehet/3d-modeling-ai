from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "3d-modeling-ai/0.4 (reference research; contact via repository)"
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "build",
    "create",
    "for",
    "from",
    "generate",
    "in",
    "make",
    "model",
    "of",
    "on",
    "please",
    "render",
    "the",
    "to",
    "with",
}


def _clean_metadata(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return None
    text = " ".join(value.replace("<br>", " ").split())
    return text[:1500] or None


def _canonical_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _query_terms(query: str) -> list[str]:
    terms = [
        token
        for token in _canonical_text(query).split()
        if len(token) > 1 and token not in SEARCH_STOPWORDS
    ]
    # Preserve order while deduplicating.
    return list(dict.fromkeys(terms))


def _candidate_relevance_score(query: str, candidate: dict[str, Any]) -> float:
    """Cheap lexical pre-ranking before the expensive multimodal verifier.

    This is intentionally permissive: the vision model is the authority. The goal here is
    only to avoid wasting its image budget on obviously unrelated Wikimedia results.
    """
    query_text = _canonical_text(query)
    terms = _query_terms(query)
    title = _canonical_text(candidate.get("title") or "")
    description = _canonical_text(candidate.get("description") or "")
    source_title = _canonical_text(candidate.get("source_title") or "")
    haystack = " ".join(item for item in (title, description, source_title) if item)

    score = 0.0
    if query_text and query_text in haystack:
        score += 12.0
    if terms:
        matched = sum(term in haystack for term in terms)
        score += matched * 2.5
        if matched == len(terms):
            score += 8.0
        elif matched >= max(1, len(terms) - 1):
            score += 3.0

    provider = candidate.get("provider")
    if provider == "wikipedia":
        # A lead image from a closely matching article is often an excellent identity anchor.
        score += 2.0

    try:
        rank = int(candidate.get("search_rank") or 99)
    except (TypeError, ValueError):
        rank = 99
    score += max(0.0, 4.0 - (rank * 0.25))

    # Common Wikimedia assets that are usually bad 3D reconstruction references.
    bad_markers = (
        "logo",
        "icon",
        "map",
        "diagram",
        "chart",
        "coat of arms",
        "flag",
        "signature",
        "wordmark",
        "poster",
        "screenshot",
    )
    if any(marker in title for marker in bad_markers):
        score -= 8.0

    return score


async def research_web_references(
    query: str,
    target_dir: Path,
    *,
    max_images: int = 6,
) -> dict[str, Any]:
    """Build a traceable candidate pack from Wikimedia projects.

    The function deliberately returns *more* candidate images than the final requested
    reference count. A multimodal relevance gate in the API then inspects the pixels and
    accepts only images that actually depict the requested subject.
    """
    query = query.strip()
    if not query:
        raise ValueError("Research query is empty.")

    target_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    candidate_limit = min(24, max(12, max_images * 3))
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        wiki_params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": "8",
            "prop": "extracts|pageimages|info",
            "exintro": "1",
            "explaintext": "1",
            "inprop": "url",
            "piprop": "thumbnail|original|name",
            "pithumbsize": "1400",
            "pilicense": "free",
        }
        wiki = (await client.get(WIKIPEDIA_API, params=wiki_params)).json()
        for rank, page in enumerate((wiki.get("query") or {}).get("pages") or [], start=1):
            page_record = {
                "provider": "wikipedia",
                "title": page.get("title"),
                "url": page.get("fullurl"),
                "extract": (page.get("extract") or "")[:4000],
                "search_rank": rank,
            }
            pages.append(page_record)
            image = page.get("thumbnail") or page.get("original") or {}
            if image.get("source"):
                candidates.append(
                    {
                        "provider": "wikipedia",
                        "title": page.get("title"),
                        "source_title": page.get("title"),
                        "description": (page.get("extract") or "")[:1500],
                        "image_url": image["source"],
                        "source_url": page.get("fullurl"),
                        "license": "free-license-filtered",
                        "search_rank": rank,
                    }
                )

        # Search both an exact phrase and the ordinary full-text query. Exact phrase tends
        # to keep specific products/vehicles/characters together; the fallback preserves
        # recall for generic subjects and Commons naming quirks.
        commons_queries = [query]
        if len(_query_terms(query)) >= 2:
            commons_queries.insert(0, f'"{query}"')

        seen_commons_titles: set[str] = set()
        for search_round, commons_query in enumerate(commons_queries):
            commons_params = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "generator": "search",
                "gsrsearch": commons_query,
                "gsrnamespace": "6",
                "gsrlimit": str(min(40, max(18, max_images * 5))),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": "1600",
            }
            commons = (await client.get(COMMONS_API, params=commons_params)).json()
            for rank, page in enumerate((commons.get("query") or {}).get("pages") or [], start=1):
                title = str(page.get("title") or "")
                if not title or title in seen_commons_titles:
                    continue
                seen_commons_titles.add(title)
                info_list = page.get("imageinfo") or []
                if not info_list:
                    continue
                info = info_list[0]
                meta = info.get("extmetadata") or {}
                description = (
                    _clean_metadata(meta.get("ImageDescription"))
                    or _clean_metadata(meta.get("ObjectName"))
                    or _clean_metadata(meta.get("Categories"))
                )
                candidates.append(
                    {
                        "provider": "wikimedia_commons",
                        "title": title,
                        "source_title": _clean_metadata(meta.get("ObjectName")) or title,
                        "description": description,
                        "image_url": info.get("thumburl") or info.get("url"),
                        "source_url": info.get("descriptionurl"),
                        "license": _clean_metadata(meta.get("LicenseShortName")) or "unknown",
                        "artist": _clean_metadata(meta.get("Artist")),
                        "credit": _clean_metadata(meta.get("Credit")),
                        "mime": info.get("mime"),
                        # Exact-phrase results get a small rank advantage.
                        "search_rank": rank + (search_round * 5),
                    }
                )

        for candidate in candidates:
            candidate["lexical_score"] = round(_candidate_relevance_score(query, candidate), 3)
        candidates.sort(
            key=lambda candidate: (
                float(candidate.get("lexical_score") or 0.0),
                1 if candidate.get("provider") == "wikipedia" else 0,
            ),
            reverse=True,
        )

        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        references: list[dict[str, Any]] = []
        for candidate in candidates:
            image_url = candidate.get("image_url")
            if not image_url or image_url in seen_urls or len(references) >= candidate_limit:
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

            # Avoid tiny icons/thumbnails even if text relevance is high.
            if width < 320 or height < 240:
                continue

            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            stored_name = f"candidate-{len(references) + 1:02d}-{digest[:12]}{extension}"
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
        "candidate_count": len(references),
        "references": references,
    }


def write_research_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
