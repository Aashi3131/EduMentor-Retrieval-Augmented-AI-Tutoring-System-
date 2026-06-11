"""Fetch short web evidence for verification (no API keys). DuckDuckGo HTML + Wikipedia fallback."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

MAX_WEB_EVIDENCE_CHARS = 12_000
USER_AGENT = "Mozilla/5.0 (compatible; EduMentor/1.0; +https://github.com/EduMentor-local)"
REQUEST_TIMEOUT_SEC = 18


def _http_get(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _http_post_form(url: str, fields: dict[str, str]) -> str | None:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _snippets_from_ddg_html(html: str) -> list[str]:
    if not html:
        return []
    out: list[str] = []
    # DuckDuckGo lite / html results commonly use result__snippet
    for pat in (
        r'class="result__snippet"[^>]*>([^<]+)',
        r'class="result-snippet"[^>]*>([^<]+)',
    ):
        for m in re.finditer(pat, html, flags=re.I):
            t = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(t) > 20:
                out.append(t)
    return out


def _wikipedia_evidence(query: str, max_chars: int) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    base = "https://en.wikipedia.org/w/api.php"
    search_url = (
        f"{base}?action=query&format=json&list=search&srsearch="
        f"{urllib.parse.quote(q)}&srlimit=5"
    )
    raw = _http_get(search_url)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    hits = (data.get("query") or {}).get("search") or []
    pageids = [str(h.get("pageid")) for h in hits if h.get("pageid")]
    if not pageids:
        return ""
    ids_param = "|".join(pageids[:5])
    extract_url = (
        f"{base}?action=query&format=json&prop=extracts&exintro=1&explaintext=1"
        f"&pageids={ids_param}"
    )
    raw2 = _http_get(extract_url)
    if not raw2:
        return ""
    try:
        data2 = json.loads(raw2)
    except json.JSONDecodeError:
        return ""
    pages = (data2.get("query") or {}).get("pages") or {}
    parts: list[str] = []
    for _pid, page in pages.items():
        title = page.get("title")
        ex = (page.get("extract") or "").strip()
        if title and ex:
            parts.append(f"{title}: {ex}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated...]"
    return text


def fetch_web_evidence(query: str, max_total_chars: int = MAX_WEB_EVIDENCE_CHARS) -> str:
    """
    Return plain text snippets suitable for a verifier prompt.
    Wikipedia extracts are listed first (usually cleaner); HTML search snippets follow.
    """
    q = (query or "").strip()
    if not q:
        return ""

    budget = max(2000, max_total_chars)
    sections: list[str] = []

    wiki = _wikipedia_evidence(q, max_chars=min(budget, 8000))
    if wiki.strip():
        sections.append("--- Wikipedia extracts ---\n\n" + wiki.strip())

    html = _http_post_form("https://html.duckduckgo.com/html/", {"q": q})
    snippets = _snippets_from_ddg_html(html or "")
    ddg = "\n\n".join(snippets[:6]) if snippets else ""
    if ddg.strip():
        sections.append("--- Search snippets ---\n\n" + ddg.strip())

    blob = "\n\n".join(sections).strip()
    if not blob:
        return ""

    if len(blob) > max_total_chars:
        blob = blob[:max_total_chars] + "\n\n[...truncated...]"
    return blob
