from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Iterable, List, Optional
from urllib.parse import urlparse, urlunparse


def groq_api_key_is_configured(api_key: str) -> bool:
    return bool(api_key.strip())


def pinecone_api_key_is_configured(api_key: str) -> bool:
    return bool(api_key.strip())


def ollama_is_running(base_url: str, timeout_seconds: float = 2.0) -> bool:
    for candidate_base_url in _candidate_base_urls(base_url):
        url = "{base}/api/version".format(base=candidate_base_url.rstrip("/"))
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 200:
                    continue
                payload = json.loads(response.read().decode("utf-8"))
                return bool(payload.get("version"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue
    return _list_ollama_models_from_cli(timeout_seconds=timeout_seconds) is not None


def list_ollama_models(base_url: str, timeout_seconds: float = 2.0) -> List[str]:
    for candidate_base_url in _candidate_base_urls(base_url):
        url = "{base}/api/tags".format(base=candidate_base_url.rstrip("/"))
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 200:
                    continue
                payload = json.loads(response.read().decode("utf-8"))
                models = payload.get("models", [])
                return [str(model.get("name")) for model in models if model.get("name")]
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue
    cli_models = _list_ollama_models_from_cli(timeout_seconds=timeout_seconds)
    return cli_models or []


def missing_ollama_models(
    base_url: str,
    required_models: Iterable[str],
    timeout_seconds: float = 2.0,
) -> List[str]:
    installed = set(list_ollama_models(base_url, timeout_seconds=timeout_seconds))
    missing = []
    for model_name in required_models:
        if not _ollama_model_is_available(installed, model_name):
            missing.append(model_name)
    return missing


def _ollama_model_is_available(installed_models: set[str], required_model: str) -> bool:
    if required_model in installed_models:
        return True
    if ":" not in required_model:
        if "{model}:latest".format(model=required_model) in installed_models:
            return True
        return any(model.startswith("{prefix}:".format(prefix=required_model)) for model in installed_models)
    return False


def _candidate_base_urls(base_url: str) -> List[str]:
    candidates = [base_url]
    parsed = urlparse(base_url)
    if parsed.hostname != "localhost":
        return candidates

    netloc = "127.0.0.1"
    if parsed.port is not None:
        netloc = "{host}:{port}".format(host=netloc, port=parsed.port)

    fallback_url = urlunparse(parsed._replace(netloc=netloc))
    if fallback_url not in candidates:
        candidates.append(fallback_url)
    return candidates


def _list_ollama_models_from_cli(timeout_seconds: float) -> Optional[List[str]]:
    ollama_path = shutil.which("ollama") or "/usr/local/bin/ollama"
    try:
        completed = subprocess.run(
            [ollama_path, "list"],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if completed.returncode != 0:
        return None

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return []

    model_names = []
    for line in lines[1:]:
        model_name = line.split(None, 1)[0]
        if model_name:
            model_names.append(model_name)
    return model_names
