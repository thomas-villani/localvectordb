"""
localvectordb_server/utils/checkdeps.py
Utility functions for checking system dependencies.
"""

import logging
import os
import subprocess  # nosec B404
import time
from typing import Optional

import httpx

from localvectordb.exceptions import OllamaNotFoundError

logger = logging.getLogger(__name__)


def check_ollama_installation() -> Optional[str]:
    """
    Check if Ollama is installed and available in the system path.

    Returns
    -------
    Optional[str]
        Version string if Ollama is installed, None otherwise

    Raises
    ------
    OllamaNotFoundError
        If Ollama is not installed or not accessible
    """
    try:
        result = subprocess.run(
            ["ollama", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )  # nosec B603 B607
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        # ollama binary not on PATH; reported via the raise below.
        pass

    raise OllamaNotFoundError(
        "Ollama is not installed or not found in system PATH. "
        "Please install Ollama first: https://ollama.ai/download"
    )


def check_ollama_service(base_url: Optional[str] = None, timeout: float = 5.0, retries: int = 3) -> bool:
    """
    Check if Ollama service is running and responding.

    Parameters
    ----------
    base_url : Optional[str]
        Ollama server URL. Defaults to the ``OLLAMA_URL`` environment variable
        or ``http://127.0.0.1:11434``. (127.0.0.1 rather than localhost: on
        Windows, "localhost" resolves to ::1 first and stalls ~2.5s per request
        because Ollama only listens on IPv4.)
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Number of attempts before giving up.

    Returns
    -------
    bool
        True if service is running, False otherwise
    """
    if base_url is None:
        base_url = os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434"
    url = base_url.rstrip("/")
    for attempt in range(max(1, retries)):
        try:
            with httpx.Client() as client:
                response = client.get(f"{url}/api/version", timeout=timeout)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        if attempt < retries - 1:
            time.sleep(2)
    return False
