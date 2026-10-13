"""Thin HTTP client for the backend API."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

# Free hosting puts the backend to sleep when idle; the first request can take
# around a minute while it wakes up, so keep trying for a while.
WAKE_UP_SECONDS = 90
RETRY_EVERY_SECONDS = 3
WAKING_STATUS_CODES = {502, 503, 504}


class ApiError(Exception):
    """A response the user should see: a message plus optional specifics."""

    def __init__(self, message: str, errors: list[str] | None = None, status: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors or []
        self.status = status


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"Could not reach the backend at {self.base_url}.") from exc
        if response.status_code >= 400:
            try:
                detail = response.json()["detail"]
                raise ApiError(detail["message"], detail.get("errors"), response.status_code)
            except (ValueError, KeyError, TypeError):
                raise ApiError(
                    f"The backend returned HTTP {response.status_code}.",
                    status=response.status_code,
                ) from None
        return response

    def wait_until_ready(self) -> None:
        """Block until /health answers, or raise ApiError after WAKE_UP_SECONDS."""
        deadline = time.monotonic() + WAKE_UP_SECONDS
        while True:
            try:
                self._request("GET", "/health")
                return
            except ApiError as exc:
                asleep = exc.status == 0 or exc.status in WAKING_STATUS_CODES
                if not asleep or time.monotonic() >= deadline:
                    raise
                time.sleep(RETRY_EVERY_SECONDS)

    def get(self, path: str) -> Any:
        return self._request("GET", f"/api/v1{path}").json()

    def get_many(self, paths: dict[str, str]) -> dict[str, Any]:
        """Fetch several API paths at once; returns results under the same keys.

        The requests are independent, so running them side by side makes a page
        load take as long as the slowest call instead of the sum of all of them.
        """
        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            futures = {name: pool.submit(self.get, path) for name, path in paths.items()}
            return {name: future.result() for name, future in futures.items()}

    def create_dataset(self, name: str) -> str:
        slug: str = self._request("POST", "/api/v1/datasets", json={"name": name}).json()["slug"]
        return slug

    def add_records(self, slug: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Add days, or merge values into days that already exist."""
        result: dict[str, Any] = self._request(
            "POST", f"/api/v1/datasets/{slug}/records", json={"records": records}
        ).json()
        return result

    def delete_dataset(self, slug: str) -> None:
        self._request("DELETE", f"/api/v1/datasets/{slug}")

    def import_file(self, slug: str, kind: str, filename: str, content: bytes) -> dict[str, Any]:
        """`kind` is "csv" or "apple-health"."""
        result: dict[str, Any] = self._request(
            "POST",
            f"/api/v1/datasets/{slug}/import/{kind}",
            files={"file": (filename, content)},
            timeout=120,
        ).json()
        return result
