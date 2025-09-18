# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/utils.py
import importlib.metadata
import os
import re


def get_system_version() -> str:
    try:
        system_version = importlib.metadata.version("localvectordb")
    except importlib.metadata.PackageNotFoundError:
        system_version = "dev"
    return system_version


def make_filename_safe(name: str, max_length: int = 255) -> str:
    # Define invalid characters based on the operating system
    if os.name == 'nt':  # Windows
        invalid_chars = r'[<>:"/\\|?*]'
        reserved_names = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
        }
    else:  # POSIX-compliant systems (Linux, macOS)
        invalid_chars = r'[/:]'
        reserved_names = set()  # No reserved names typically on POSIX

    # Replace invalid characters with an underscore
    safe_name = re.sub(invalid_chars, "_", name)

    # Strip leading/trailing spaces and periods (applies to Windows)
    if os.name == 'nt':
        safe_name = safe_name.strip(" .")
    else:
        safe_name = safe_name.strip()

    # Ensure the name is not a reserved name (Windows only)
    if os.name == 'nt' and safe_name.upper() in reserved_names:
        safe_name += "_safe"

    # Truncate to maximum length (ensure allowance for file extensions)
    safe_name = safe_name[:max_length]

    # Return a fallback name if the result is empty
    return safe_name
