#  Copyright (c) 2023-2025 Tom Villani, Ph.D. All rights reserved.

"""
localvectordb_server/utils/checkdeps.py
Utility functions for checking system dependencies.
"""

import logging
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
        pass

    raise OllamaNotFoundError(
        "Ollama is not installed or not found in system PATH. "
        "Please install Ollama first: https://ollama.ai/download"
    )


def check_ollama_service() -> bool:
    """
    Check if Ollama service is running and responding.

    Returns
    -------
    bool
        True if service is running, False otherwise

    Raises
    ------
    OllamaNotFoundError
        If Ollama service is not running or not accessible
    """
    try:
        retries = 0
        while retries < 3:
            with httpx.Client() as client:
                response = client.get("http://localhost:11434/api/version", timeout=60.0)
            if response.status_code == 200:
                break
            time.sleep(2)
            retries += 1
        return bool(response.status_code == 200)
    except httpx.RequestError:
        return False
