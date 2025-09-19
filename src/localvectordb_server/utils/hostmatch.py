# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb_server/utils/hostmatch.py
"""
Host Header Validation Utilities

Provides robust host header validation with support for:
- Automatic port stripping (host:port -> host)
- Wildcard subdomain patterns (*.example.com)
- Exact hostname matching
- Case-insensitive comparison
- IPv4 and IPv6 address handling

This module addresses security vulnerabilities in naive string comparison
by implementing proper hostname parsing and pattern matching.

Examples
--------
Basic usage::

    from localvectordb_server.utils.hostmatch import validate_host_against_patterns

    trusted_patterns = ["localhost", "*.example.com", "api.mysite.org"]

    # These will match
    validate_host_against_patterns("localhost:5000", trusted_patterns)  # True
    validate_host_against_patterns("api.example.com", trusted_patterns)  # True
    validate_host_against_patterns("www.example.com", trusted_patterns)  # True
    validate_host_against_patterns("api.mysite.org", trusted_patterns)   # True

    # These will not match
    validate_host_against_patterns("evil.com", trusted_patterns)         # False
    validate_host_against_patterns("example.com.evil.com", trusted_patterns)  # False

Pattern Types
-------------
- **Exact**: `example.com` matches only `example.com`
- **Wildcard subdomain**: `*.example.com` matches `api.example.com`, `www.example.com`, etc.
- **Wildcard any**: `*` matches any hostname (use with caution)
- **Port handling**: `localhost:5000` is treated as `localhost`
- **Case insensitive**: `EXAMPLE.COM` matches `example.com`

Security Notes
--------------
- Wildcard patterns only match subdomains, not arbitrary domains
- IPv6 addresses in brackets are properly handled
- Port numbers are automatically stripped
- Empty or malformed hosts are rejected
"""
import fnmatch
import ipaddress
import logging
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Regex for validating hostname format (basic check)
HOSTNAME_PATTERN = re.compile(
    r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$'
)

# IPv6 address in brackets pattern
IPV6_BRACKET_PATTERN = re.compile(r'^\[([^\]]+)\](?::(\d+))?$')


def parse_host(host_string: str) -> Tuple[str, Optional[int]]:
    """
    Extract hostname and port from host:port format.

    Handles various formats including IPv6 addresses in brackets.

    Parameters
    ----------
    host_string : str
        Host string potentially including port (e.g., 'localhost:5000', '[::1]:8080')

    Returns
    -------
    Tuple[str, Optional[int]]
        Tuple of (hostname, port) where port is None if not specified

    Examples
    --------
    >>> parse_host("localhost:5000")
    ('localhost', 5000)
    >>> parse_host("example.com")
    ('example.com', None)
    >>> parse_host("[::1]:8080")
    ('::1', 8080)
    >>> parse_host("[2001:db8::1]")
    ('2001:db8::1', None)
    """
    if not host_string:
        return "", None

    host_string = host_string.strip()

    # Handle IPv6 addresses in brackets
    ipv6_match = IPV6_BRACKET_PATTERN.match(host_string)
    if ipv6_match:
        hostname = ipv6_match.group(1)
        port_str = ipv6_match.group(2)
        port = int(port_str) if port_str else None
        return hostname, port

    # Handle regular hostname:port format
    if ':' in host_string:
        # Split on last colon and check if the last part looks like a port
        parts = host_string.rsplit(':', 1)
        if len(parts) == 2:
            potential_hostname, potential_port = parts

            # Try to parse as port number
            try:
                port = int(potential_port)
                # Check if the remaining part could be a valid hostname or IPv6
                # If it's a valid IPv6 address, treat as hostname:port
                # If not, check if the whole string is a valid IPv6 address
                try:
                    ipaddress.IPv6Address(potential_hostname)
                    # Valid IPv6 address with port
                    return potential_hostname, port
                except ipaddress.AddressValueError:
                    # Not a valid IPv6, check if whole string is IPv6
                    try:
                        ipaddress.IPv6Address(host_string)
                        # Whole string is valid IPv6, no port
                        return host_string, None
                    except ipaddress.AddressValueError:
                        # Neither part is valid IPv6, treat as hostname:port
                        return potential_hostname, port
            except ValueError:
                # Port part is not a number, treat as part of hostname
                return host_string, None

    return host_string, None


def normalize_hostname(hostname: str) -> str:
    """
    Normalize hostname for comparison.

    Parameters
    ----------
    hostname : str
        Hostname to normalize

    Returns
    -------
    str
        Normalized hostname (lowercase, stripped)
    """
    if not hostname:
        return ""

    return hostname.lower().strip()


def is_valid_hostname(hostname: str) -> bool:
    """
    Check if hostname is valid format.

    Parameters
    ----------
    hostname : str
        Hostname to validate

    Returns
    -------
    bool
        True if hostname is valid format
    """
    if not hostname:
        return False

    # Check if it looks like an IP address first
    # If it contains only digits and dots, validate as IPv4
    if all(c.isdigit() or c == '.' for c in hostname):
        try:
            ipaddress.IPv4Address(hostname)
            return True
        except ipaddress.AddressValueError:
            return False

    # Check for IPv6 address (contains colons)
    if ':' in hostname:
        try:
            ipaddress.IPv6Address(hostname)
            return True
        except ipaddress.AddressValueError:
            return False

    # Check hostname pattern for regular hostnames
    return bool(HOSTNAME_PATTERN.match(hostname))


def match_host_pattern(hostname: str, pattern: str) -> bool:
    """
    Check if hostname matches pattern with wildcard support.

    Supports:
    - Exact matches: 'example.com' matches only 'example.com'
    - Wildcard subdomains: '*.example.com' matches 'api.example.com', 'www.example.com'
    - Universal wildcard: '*' matches any hostname

    Parameters
    ----------
    hostname : str
        Hostname to check (should be normalized)
    pattern : str
        Pattern to match against (supports wildcards)

    Returns
    -------
    bool
        True if hostname matches pattern

    Examples
    --------
    >>> match_host_pattern("api.example.com", "*.example.com")
    True
    >>> match_host_pattern("example.com", "*.example.com")
    False
    >>> match_host_pattern("example.com", "example.com")
    True
    >>> match_host_pattern("api.example.com", "*")
    True
    """
    if not hostname or not pattern:
        return False

    hostname = normalize_hostname(hostname)
    pattern = normalize_hostname(pattern)

    # Universal wildcard
    if pattern == "*":
        return True

    # Exact match
    if pattern == hostname:
        return True

    # Wildcard subdomain pattern
    if pattern.startswith("*."):
        domain_part = pattern[2:]  # Remove "*."
        if not domain_part:
            return False

        # Check if hostname ends with the domain part
        if hostname.endswith(f".{domain_part}"):
            # Ensure it's a subdomain, not just a suffix match
            # hostname = "api.example.com", domain_part = "example.com"
            # hostname should be "api.example.com" and not "notexample.com"
            prefix = hostname[:-len(f".{domain_part}")]
            # Prefix should be a valid subdomain (no dots for single-level wildcard)
            if prefix and '.' not in prefix:
                return True

        return False

    # Use fnmatch for other wildcard patterns (if any)
    return fnmatch.fnmatch(hostname, pattern)


def validate_host_against_patterns(host: str, trusted_patterns: List[str]) -> bool:
    """
    Validate host against list of trusted patterns.

    This is the main function to use for host validation. It handles
    port stripping, normalization, and pattern matching.

    Parameters
    ----------
    host : str
        Host string from request (may include port)
    trusted_patterns : List[str]
        List of trusted host patterns

    Returns
    -------
    bool
        True if host matches any trusted pattern

    Examples
    --------
    >>> patterns = ["localhost", "*.example.com", "api.mysite.org"]
    >>> validate_host_against_patterns("localhost:5000", patterns)
    True
    >>> validate_host_against_patterns("api.example.com", patterns)
    True
    >>> validate_host_against_patterns("evil.com", patterns)
    False
    """
    if not host or not trusted_patterns:
        logger.debug(f"Host validation failed: empty host={repr(host)} or patterns={repr(trusted_patterns)}")
        return False

    # Parse hostname from host:port format
    hostname, port = parse_host(host)

    if not hostname:
        logger.debug(f"Host validation failed: could not extract hostname from {repr(host)}")
        return False

    # Validate hostname format
    if not is_valid_hostname(hostname):
        logger.debug(f"Host validation failed: invalid hostname format {repr(hostname)}")
        return False

    # Check against each trusted pattern
    for pattern in trusted_patterns:
        if match_host_pattern(hostname, pattern):
            logger.debug(f"Host {repr(hostname)} matched pattern {repr(pattern)}")
            return True

    logger.debug(f"Host {repr(hostname)} did not match any trusted patterns: {trusted_patterns}")
    return False


def validate_trusted_host_patterns(patterns: List[str]) -> List[str]:
    """
    Validate and normalize a list of trusted host patterns.

    Parameters
    ----------
    patterns : List[str]
        List of host patterns to validate

    Returns
    -------
    List[str]
        List of validation error messages (empty if all valid)

    Examples
    --------
    >>> validate_trusted_host_patterns(["localhost", "*.example.com"])
    []
    >>> validate_trusted_host_patterns(["", "invalid..domain"])
    ['Empty pattern at index 0', 'Invalid pattern format: "invalid..domain"']
    """
    errors = []

    if not patterns:
        return ["No trusted host patterns provided"]

    for i, pattern in enumerate(patterns):
        if not pattern or not pattern.strip():
            errors.append(f"Empty pattern at index {i}")
            continue

        pattern = pattern.strip().lower()

        # Check for universal wildcard (allowed but should warn)
        if pattern == "*":
            continue

        # Check wildcard subdomain pattern
        if pattern.startswith("*."):
            domain_part = pattern[2:]
            if not domain_part:
                errors.append(f"Invalid wildcard pattern: '{pattern}' (missing domain)")
                continue
            if not is_valid_hostname(domain_part):
                errors.append(f"Invalid domain in wildcard pattern: '{pattern}'")
                continue
        else:
            # Check exact hostname pattern
            if not is_valid_hostname(pattern):
                errors.append(f"Invalid pattern format: '{pattern}'")

    return errors