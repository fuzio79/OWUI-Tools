"""
title: Portainer Read-Only
description: Read-only container and stack inspection through the Portainer API.
requirements: requests
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field


class Tools:
    """
    Portainer read-only inspection tools.
    Talks to the Portainer HTTP API (X-API-Key auth) to list/inspect
    containers, read logs, and view stacks. All operations are read-only.
    """

    # ── VALVES ──────────────────────────────────────────────────────
    class Valves(BaseModel):
        PORTAINER_URL: str = Field(
            default="http://portainer:9000",
            description="Portainer base URL, without a trailing slash.",
        )
        API_TOKEN: str = Field(
            default="",
            description="Portainer API access token.",
        )
        ENDPOINT_ID: int = Field(
            default=1,
            description="Portainer environment/endpoint ID containing the Docker host.",
        )
        VERIFY_TLS: bool = Field(
            default=True,
            description="Verify Portainer's TLS certificate. Keep enabled when using HTTPS.",
        )
        TIMEOUT_SECONDS: int = Field(
            default=20,
            description="HTTP request timeout.",
        )
        MAX_LOG_LINES: int = Field(
            default=300,
            description="Maximum number of log lines returned.",
        )

    # ── INIT ────────────────────────────────────────────────────────
    def __init__(self):
        self.valves = self.Valves()

    # ── HELPERS ──────────────────────────────────────────────────────
    def _request(self, method: str, path: str, params: Optional[dict] = None) -> Any:
        """Make an authenticated request to Portainer and return parsed JSON or text."""
        base = self.valves.PORTAINER_URL.rstrip("/")
        if not self.valves.API_TOKEN:
            return {"error": "Set the API_TOKEN valve in the tool settings."}

        url = base + path
        headers = {"X-API-Key": self.valves.API_TOKEN}

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=self.valves.TIMEOUT_SECONDS,
                verify=self.valves.VERIFY_TLS,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return response.text

        except requests.RequestException as exc:
            # Avoid returning request headers, which contain the API token.
            return {"error": f"Portainer request failed: {exc}"}

    def _request_bytes(self, method: str, path: str, params: Optional[dict] = None) -> bytes:
        """Make an authenticated request and return the raw response body."""
        base = self.valves.PORTAINER_URL.rstrip("/")
        if not self.valves.API_TOKEN:
            return b""

        url = base + path
        response = requests.request(
            method=method,
            url=url,
            headers={"X-API-Key": self.valves.API_TOKEN},
            params=params,
            timeout=self.valves.TIMEOUT_SECONDS,
            verify=self.valves.VERIFY_TLS,
        )
        response.raise_for_status()
        return response.content

    @staticmethod
    def _parse_logs(raw: bytes) -> List[str]:
        """Parse the Docker logs multiplexed stream into plain lines.

        Each record: 1 byte stream (0=stdout, 1=stderr), 4 bytes timestamp
        (big-endian), 3 bytes message length (big-endian), then `length`
        bytes of message.
        """
        lines: List[str] = []
        i, n = 0, len(raw)
        while i + 8 <= n:
            stream = raw[i]
            length = int.from_bytes(raw[i + 5 : i + 8], "big")
            i += 8
            if i + length > n:
                break
            msg = raw[i : i + length]
            i += length
            for part in msg.decode("utf-8", errors="replace").split("\n"):
                if part:
                    lines.append(f"[{'stdout' if stream == 0 else 'stderr'}] {part}")
        return lines

    # ── READ-ONLY ───────────────────────────────────────────────────

    async def list_endpoints(self) -> Dict[str, Any]:
        """
        List Portainer endpoints (environments).

        Useful for discovering the correct ENDPOINT_ID to configure in the
        tool valves, without checking the Portainer UI.

        Returns:
            {"count": N, "endpoints": [{"id", "name", "type", "status", "url"}, ...]}
            or {"error": ...} on failure.
        """
        data = await asyncio.to_thread(self._request, "GET", "/api/endpoints")
        if isinstance(data, dict) and "error" in data:
            return data
        if not isinstance(data, list):
            return {"error": "Unexpected response from Portainer."}

        summary: List[Dict[str, Any]] = []
        for e in data:
            summary.append(
                {
                    "id": e.get("Id"),
                    "name": e.get("Name", ""),
                    "type": e.get("Type"),
                    "status": e.get("Status", ""),
                    "url": e.get("URL", ""),
                }
            )
        return {"count": len(summary), "endpoints": summary}

    async def list_containers(self, all_containers: bool = True) -> Dict[str, Any]:
        """
        List containers in the configured Portainer environment.

        Returns a compact summary (name, state, status, image) per container
        instead of the full Docker JSON, so the result stays small enough to
        fit in the chat model's context.

        Args:
            all_containers: If True, include stopped containers (default).

        Returns:
            {"count": N, "containers": [{"name", "state", "status", "image"}, ...]}
            or {"error": ...} on failure.
        """
        data = await asyncio.to_thread(
            self._request,
            "GET",
            f"/api/endpoints/{self.valves.ENDPOINT_ID}/docker/containers/json",
            params={"all": str(all_containers).lower()},
        )
        if isinstance(data, dict) and "error" in data:
            return data
        if not isinstance(data, list):
            return {"error": "Unexpected response from Portainer."}

        summary: List[Dict[str, Any]] = []
        for c in data:
            names = c.get("Names") or []
            name = names[0].lstrip("/") if names else (c.get("Id", "")[:12])
            summary.append(
                {
                    "name": name,
                    "state": c.get("State", ""),
                    "status": c.get("Status", ""),
                    "image": c.get("Image", ""),
                }
            )
        return {"count": len(summary), "containers": summary}

    async def inspect_container(self, container_id_or_name: str) -> Dict[str, Any]:
        """
        Inspect one container by ID or name.

        Args:
            container_id_or_name: Container ID or name.

        Returns:
            The full container inspect JSON, or {"error": ...} on failure.
        """
        container = quote(container_id_or_name.strip().lstrip("/"), safe="")
        data = await asyncio.to_thread(
            self._request,
            "GET",
            f"/api/endpoints/{self.valves.ENDPOINT_ID}/docker/containers/{container}/json",
        )
        if isinstance(data, dict) and "error" in data:
            return data
        return {"container": data}

    async def container_logs(
        self,
        container_id_or_name: str,
        tail: int = 100,
        timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Get recent stdout/stderr logs for a container.

        Args:
            container_id_or_name: Container ID or name.
            tail: Number of lines to tail. Clamped to MAX_LOG_LINES.
            timestamps: If True, keep Docker's own timestamps in the stream.

        Returns:
            {"container": ..., "lines": [...]} or {"error": ...} on failure.
        """
        container = quote(container_id_or_name.strip().lstrip("/"), safe="")
        tail = max(1, min(int(tail), self.valves.MAX_LOG_LINES))

        if not self.valves.API_TOKEN:
            return {"error": "Set the API_TOKEN valve in the tool settings."}

        path = (
            f"/api/endpoints/{self.valves.ENDPOINT_ID}"
            f"/docker/containers/{container}/logs"
        )
        params = {
            "stdout": "true",
            "stderr": "true",
            "tail": str(tail),
            "timestamps": str(timestamps).lower(),
        }
        try:
            raw = await asyncio.to_thread(
                self._request_bytes, "GET", path, params=params
            )
        except requests.RequestException as exc:
            return {"error": f"Portainer request failed: {exc}"}

        return {"container": container_id_or_name, "lines": self._parse_logs(raw)}

    async def list_stacks(self) -> Dict[str, Any]:
        """
        List Portainer stacks.

        Returns:
            The list of stacks, or {"error": ...} on failure.
        """
        data = await asyncio.to_thread(self._request, "GET", "/api/stacks")
        if isinstance(data, dict) and "error" in data:
            return data
        return {"stacks": data}

    async def get_stack_compose(self, stack_id: int) -> Dict[str, Any]:
        """
        Retrieve the compose file for a stack by its Portainer stack ID.

        Args:
            stack_id: Portainer stack ID.

        Returns:
            {"stack_id": ..., "compose": ...} or {"error": ...} on failure.
        """
        data = await asyncio.to_thread(
            self._request, "GET", f"/api/stacks/{int(stack_id)}/file"
        )
        if isinstance(data, dict) and "error" in data:
            return data

        # Portainer commonly returns the compose content inside StackFileContent.
        if isinstance(data, dict) and "StackFileContent" in data:
            return {"stack_id": int(stack_id), "compose": str(data["StackFileContent"])}
        return {"stack_id": int(stack_id), "data": data}
