"""Compare the public Cloudflare dashboard payload with the local dashboard artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard-url",
        required=True,
        help="Full public URL for dashboard.json on Cloudflare Pages.",
    )
    parser.add_argument(
        "--local-dashboard",
        default="cloudflare-app/data/dashboard.json",
        help="Path to the local generated dashboard artifact.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="HTTP timeout when fetching the public dashboard payload.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object.")
    return payload


def fetch_json(url: str, timeout_seconds: int) -> tuple[dict[str, Any], dict[str, str]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "User-Agent": "HomeRunAlgoCloudflareFreshnessCheck/1.0",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError(f"{url} must return a top-level JSON object.")
        headers = {key.lower(): value for key, value in response.headers.items()}
        return payload, headers


def summarize_payload(payload: dict[str, Any]) -> str:
    return (
        f"generated_at={payload.get('generated_at')} "
        f"latest_available_date={payload.get('latest_available_date')} "
        f"latest_slate_size={payload.get('overview', {}).get('latest_slate_size')}"
    )


def main() -> int:
    args = parse_args()
    local_payload = load_json(Path(args.local_dashboard))

    try:
        remote_payload, headers = fetch_json(args.dashboard_url, args.timeout_seconds)
    except HTTPError as exc:
        print(f"Cloudflare dashboard request failed with HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 2
    except URLError as exc:
        print(f"Cloudflare dashboard request failed: {exc.reason}", file=sys.stderr)
        return 2

    print("Local dashboard :", summarize_payload(local_payload))
    print("Remote dashboard:", summarize_payload(remote_payload))
    if headers:
        cache_status = headers.get("cf-cache-status") or headers.get("x-cache") or "n/a"
        age = headers.get("age") or "n/a"
        print(f"Remote cache     : cf-cache-status={cache_status} age={age}")

    local_generated_at = str(local_payload.get("generated_at") or "")
    remote_generated_at = str(remote_payload.get("generated_at") or "")
    if local_generated_at == remote_generated_at:
        print("Cloudflare dashboard is fresh and matches the local artifact.")
        return 0

    local_latest_date = str(local_payload.get("latest_available_date") or "")
    remote_latest_date = str(remote_payload.get("latest_available_date") or "")
    print("Cloudflare dashboard is stale or serving a different artifact.", file=sys.stderr)
    print(
        "Check Cloudflare Pages production deployment, branch/output settings, and browser/CDN cache. "
        f"Local generated_at={local_generated_at}, remote generated_at={remote_generated_at}, "
        f"local latest_available_date={local_latest_date}, remote latest_available_date={remote_latest_date}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
