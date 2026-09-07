import asyncio
from typing import Iterable

import aiohttp

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
DEFAULT_HEADERS = {"User-Agent": "NEL-MDPO/0.1 (research code)"}


class WikidataClient:
    def __init__(
        self,
        languages: Iterable[str],
        limit: int = 5,
        max_retries: int = 5,
        backoff: float = 1.0,
        max_concurrent: int = 8,
    ) -> None:
        self.languages = list(dict.fromkeys(languages))
        self.limit = limit
        self.max_retries = max_retries
        self.backoff = backoff
        self._cache: dict[tuple[str, str, int], list[str]] = {}
        self._info_cache: dict[tuple[str, str], list[str]] = {}
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_concurrent)

    async def search(self, session: aiohttp.ClientSession, query: str, language: str) -> list[str]:
        key = (query.lower().strip(), language, self.limit)
        async with self._lock:
            if key in self._cache:
                return self._cache[key]

        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": language,
            "format": "json",
            "type": "item",
            "limit": self.limit,
        }
        async with self._sem:
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with session.get(WIKIDATA_API, params=params, headers=DEFAULT_HEADERS) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(self.backoff * (2 ** (attempt - 1)))
                            continue
                        if resp.status != 200:
                            return []
                        data = await resp.json()
                        results = [item["id"] for item in data.get("search", []) if item.get("id")]
                        async with self._lock:
                            self._cache[key] = results
                        return results
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(self.backoff * attempt)
        return []

    async def search_multilang(self, session: aiohttp.ClientSession, query: str) -> list[str]:
        results = await asyncio.gather(*(self.search(session, query, lang) for lang in self.languages))
        seen = set()
        out = []
        for entries in results:
            for qid in entries:
                if qid not in seen:
                    seen.add(qid)
                    out.append(qid)
        return out

    async def get_entity_info(self, session: aiohttp.ClientSession, qid: str, language: str) -> list[str]:
        key = (qid.strip(), language)
        async with self._lock:
            if key in self._info_cache:
                return self._info_cache[key]

        params = {
            "action": "wbgetentities",
            "ids": qid,
            "languages": language,
            "format": "json",
            "props": "labels|descriptions",
        }
        async with self._sem:
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with session.get(WIKIDATA_API, params=params, headers=DEFAULT_HEADERS) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(self.backoff * (2 ** (attempt - 1)))
                            continue
                        if resp.status != 200:
                            return []
                        data = await resp.json()
                        out = []
                        for item_id, item in data.get("entities", {}).items():
                            if "missing" in item:
                                continue
                            label = item.get("labels", {}).get(language, {}).get("value", "")
                            desc = item.get("descriptions", {}).get(language, {}).get("value", "")
                            out.append(f"{label} - {item_id} - {desc}")
                        async with self._lock:
                            self._info_cache[key] = out
                        return out
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(self.backoff * attempt)
        return []
