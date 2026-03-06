from __future__ import annotations

import json
import urllib.error
import urllib.request


def ollama_is_running(base_url: str, timeout_seconds: float = 2.0) -> bool:
    url = "{base}/api/version".format(base=base_url.rstrip("/"))
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("version"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False
