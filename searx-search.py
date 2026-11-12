import asyncio
import logging
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Tools:
    """
    Search a self-hosted SearXNG instance and return a limited set of results.
    """

    class Valves(BaseModel):
        searxng_url: str = Field(
            default="http://searxng:8080",
            description="Base URL of the SearXNG instance.",
        )
        num_results: int = Field(
            default=10,
            ge=1,
            le=30,
            description="Maximum number of results to return.",
        )
        time_range: str = Field(
            default="",
            description="Optional freshness filter: day, week, month, or year.",
        )
        categories: str = Field(
            default="",
            description="Optional comma-separated SearXNG categories.",
        )
        timeout: int = Field(
            default=15,
            ge=3,
            le=60,
            description="Request timeout in seconds.",
        )
        max_text_chars: int = Field(
            default=1200,
            ge=200,
            le=5000,
            description="Maximum characters returned for each result's title and content.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def searxng_search(self, query: str) -> Dict[str, Any]:
        """
        Search the public web via a self-hosted SearXNG instance.

        Use this for current events, external references, news, or any topic
        not covered by the user's internal documents (use rag_search for
        those). Results are from the open web, not from uploaded knowledge.

        Freshness and category filters are controlled by the time_range and
        categories valves in the tool settings (not by this function's args).

        Args:
            query: The search terms to look up on the web.

        Returns:
            {"status": "ok", "count": N,
             "results": [{"title": ..., "url": ..., "content": ...}, ...]}
            On error: {"status": "error", "detail": "..."}
        """
        query = (query or "").strip()
        if not query:
            return {"status": "error", "detail": "Search query cannot be empty."}

        # Read the current valve value each call, rather than caching the URL
        # during initialization. This ensures valve changes take effect.
        base_url = (self.valves.searxng_url or "").strip().rstrip("/")
        parsed_url = urlparse(base_url)

        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            return {
                "status": "error",
                "detail": "SearXNG URL must be a valid http:// or https:// URL.",
            }

        time_range = (self.valves.time_range or "").strip().lower()
        allowed_time_ranges = {"", "day", "week", "month", "year"}
        if time_range not in allowed_time_ranges:
            return {
                "status": "error",
                "detail": "Invalid time range. Use day, week, month, year, or leave it empty.",
            }

        params: Dict[str, str] = {
            "q": query[:1000],
            "format": "json",
        }

        if time_range:
            params["time_range"] = time_range

        categories = (self.valves.categories or "").strip()
        if categories:
            params["categories"] = categories[:200]

        endpoint = f"{base_url}/search"

        try:
            response = await asyncio.to_thread(
                requests.get,
                endpoint,
                params=params,
                timeout=self.valves.timeout,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.warning("SearXNG request timed out.")
            return {"status": "error", "detail": "SearXNG request timed out."}
        except requests.exceptions.RequestException as exc:
            logger.warning("SearXNG request failed: %s", exc)
            return {
                "status": "error",
                "detail": "Could not reach SearXNG or the request failed.",
            }
        except ValueError:
            logger.warning("SearXNG returned invalid JSON.")
            return {"status": "error", "detail": "SearXNG returned invalid JSON."}

        if not isinstance(data, dict):
            return {
                "status": "error",
                "detail": "SearXNG returned an unexpected response.",
            }

        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            return {
                "status": "error",
                "detail": "SearXNG response has an invalid results field.",
            }

        limit = self.valves.num_results
        text_limit = self.valves.max_text_chars
        results: List[Dict[str, str]] = []

        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue

            title = item.get("title")
            url = item.get("url")
            content = item.get("content")

            results.append(
                {
                    "title": str(title or "")[:text_limit],
                    "url": str(url or "")[:2000],
                    "content": str(content or "")[:text_limit],
                }
            )

        return {
            "status": "ok",
            "count": len(results),
            "results": results,
        }
