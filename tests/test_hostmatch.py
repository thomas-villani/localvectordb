# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# tests/test_hostmatch.py
"""
Tests for host header validation utilities

Tests the robust host matching implementation that addresses security
vulnerabilities in naive string comparison by providing proper hostname
parsing and pattern matching with wildcard support.
"""

from localvectordb_server.utils.hostmatch import (
    is_valid_hostname,
    match_host_pattern,
    normalize_hostname,
    parse_host,
    validate_host_against_patterns,
    validate_trusted_host_patterns,
)


class TestParseHost:
    """Test host:port parsing functionality."""

    def test_parse_hostname_only(self):
        """Test parsing hostname without port."""
        hostname, port = parse_host("example.com")
        assert hostname == "example.com"
        assert port is None

    def test_parse_hostname_with_port(self):
        """Test parsing hostname with port."""
        hostname, port = parse_host("localhost:5000")
        assert hostname == "localhost"
        assert port == 5000

    def test_parse_ipv4_with_port(self):
        """Test parsing IPv4 address with port."""
        hostname, port = parse_host("192.168.1.1:8080")
        assert hostname == "192.168.1.1"
        assert port == 8080

    def test_parse_ipv6_with_brackets_and_port(self):
        """Test parsing IPv6 address in brackets with port."""
        hostname, port = parse_host("[::1]:8080")
        assert hostname == "::1"
        assert port == 8080

    def test_parse_ipv6_with_brackets_no_port(self):
        """Test parsing IPv6 address in brackets without port."""
        hostname, port = parse_host("[2001:db8::1]")
        assert hostname == "2001:db8::1"
        assert port is None

    def test_parse_empty_host(self):
        """Test parsing empty or None host."""
        hostname, port = parse_host("")
        assert hostname == ""
        assert port is None

        hostname, port = parse_host(None)
        assert hostname == ""
        assert port is None

    def test_parse_invalid_port(self):
        """Test parsing with invalid port number."""
        hostname, port = parse_host("example.com:abc")
        assert hostname == "example.com:abc"
        assert port is None

    def test_parse_multiple_colons_ipv6_no_brackets(self):
        """Test parsing IPv6 without brackets (should use last colon for port)."""
        hostname, port = parse_host("2001:db8::1:8080")
        assert hostname == "2001:db8::1"
        assert port == 8080


class TestNormalizeHostname:
    """Test hostname normalization."""

    def test_normalize_uppercase(self):
        """Test normalizing uppercase hostname."""
        assert normalize_hostname("EXAMPLE.COM") == "example.com"

    def test_normalize_mixed_case(self):
        """Test normalizing mixed case hostname."""
        assert normalize_hostname("Example.COM") == "example.com"

    def test_normalize_with_whitespace(self):
        """Test normalizing hostname with whitespace."""
        assert normalize_hostname("  example.com  ") == "example.com"

    def test_normalize_empty(self):
        """Test normalizing empty hostname."""
        assert normalize_hostname("") == ""
        assert normalize_hostname(None) == ""


class TestIsValidHostname:
    """Test hostname format validation."""

    def test_valid_hostname(self):
        """Test valid hostname formats."""
        assert is_valid_hostname("example.com") is True
        assert is_valid_hostname("www.example.com") is True
        assert is_valid_hostname("api-v2.example.com") is True
        assert is_valid_hostname("localhost") is True

    def test_valid_ipv4(self):
        """Test valid IPv4 addresses."""
        assert is_valid_hostname("192.168.1.1") is True
        assert is_valid_hostname("127.0.0.1") is True
        assert is_valid_hostname("10.0.0.1") is True

    def test_valid_ipv6(self):
        """Test valid IPv6 addresses."""
        assert is_valid_hostname("::1") is True
        assert is_valid_hostname("2001:db8::1") is True
        assert is_valid_hostname("fe80::1") is True

    def test_invalid_hostname(self):
        """Test invalid hostname formats."""
        assert is_valid_hostname("") is False
        assert is_valid_hostname("..example.com") is False
        assert is_valid_hostname("example..com") is False
        assert is_valid_hostname(".example.com") is False
        assert is_valid_hostname("example.com.") is False

    def test_invalid_ipv4(self):
        """Test invalid IPv4 addresses."""
        assert is_valid_hostname("256.256.256.256") is False
        assert is_valid_hostname("192.168.1") is False

    def test_invalid_ipv6(self):
        """Test invalid IPv6 addresses."""
        assert is_valid_hostname("gggg::1") is False
        assert is_valid_hostname("::") is True  # Valid IPv6


class TestMatchHostPattern:
    """Test pattern matching functionality."""

    def test_exact_match(self):
        """Test exact hostname matching."""
        assert match_host_pattern("example.com", "example.com") is True
        assert match_host_pattern("example.com", "different.com") is False

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        assert match_host_pattern("EXAMPLE.COM", "example.com") is True
        assert match_host_pattern("example.com", "EXAMPLE.COM") is True

    def test_wildcard_subdomain_match(self):
        """Test wildcard subdomain patterns."""
        assert match_host_pattern("api.example.com", "*.example.com") is True
        assert match_host_pattern("www.example.com", "*.example.com") is True
        assert match_host_pattern("admin.example.com", "*.example.com") is True

    def test_wildcard_subdomain_no_match(self):
        """Test wildcard subdomain patterns that should not match."""
        # Exact domain should not match wildcard subdomain
        assert match_host_pattern("example.com", "*.example.com") is False
        # Multi-level subdomains should not match single-level wildcard
        assert match_host_pattern("api.v1.example.com", "*.example.com") is False
        # Different domain should not match
        assert match_host_pattern("api.different.com", "*.example.com") is False

    def test_universal_wildcard(self):
        """Test universal wildcard pattern."""
        assert match_host_pattern("anything.com", "*") is True
        assert match_host_pattern("localhost", "*") is True
        assert match_host_pattern("192.168.1.1", "*") is True

    def test_empty_patterns(self):
        """Test empty hostname or pattern."""
        assert match_host_pattern("", "example.com") is False
        assert match_host_pattern("example.com", "") is False
        assert match_host_pattern("", "") is False

    def test_malformed_wildcard_pattern(self):
        """Test malformed wildcard patterns."""
        assert match_host_pattern("example.com", "*.") is False
        assert match_host_pattern("example.com", "*..com") is False


class TestValidateHostAgainstPatterns:
    """Test main host validation function."""

    def test_single_exact_pattern(self):
        """Test validation against single exact pattern."""
        patterns = ["localhost"]
        assert validate_host_against_patterns("localhost", patterns) is True
        assert validate_host_against_patterns("localhost:5000", patterns) is True
        assert validate_host_against_patterns("different.com", patterns) is False

    def test_multiple_patterns(self):
        """Test validation against multiple patterns."""
        patterns = ["localhost", "*.example.com", "api.mysite.org"]

        # Should match
        assert validate_host_against_patterns("localhost", patterns) is True
        assert validate_host_against_patterns("localhost:8080", patterns) is True
        assert validate_host_against_patterns("api.example.com", patterns) is True
        assert validate_host_against_patterns("www.example.com", patterns) is True
        assert validate_host_against_patterns("api.mysite.org", patterns) is True
        assert validate_host_against_patterns("api.mysite.org:443", patterns) is True

        # Should not match
        assert validate_host_against_patterns("evil.com", patterns) is False
        assert validate_host_against_patterns("example.com", patterns) is False  # No wildcard match
        assert validate_host_against_patterns("different.mysite.org", patterns) is False

    def test_ipv4_addresses(self):
        """Test validation with IPv4 addresses."""
        patterns = ["127.0.0.1", "192.168.*"]
        assert validate_host_against_patterns("127.0.0.1", patterns) is True
        assert validate_host_against_patterns("127.0.0.1:3000", patterns) is True

    def test_ipv6_addresses(self):
        """Test validation with IPv6 addresses."""
        patterns = ["::1", "fe80::*"]
        assert validate_host_against_patterns("::1", patterns) is True
        assert validate_host_against_patterns("[::1]:8080", patterns) is True

    def test_case_insensitive_validation(self):
        """Test case-insensitive validation."""
        patterns = ["localhost", "*.EXAMPLE.com"]
        assert validate_host_against_patterns("LOCALHOST", patterns) is True
        assert validate_host_against_patterns("api.example.COM", patterns) is True

    def test_empty_or_invalid_inputs(self):
        """Test validation with empty or invalid inputs."""
        patterns = ["localhost"]
        assert validate_host_against_patterns("", patterns) is False
        assert validate_host_against_patterns("localhost", []) is False
        assert validate_host_against_patterns("localhost", None) is False

    def test_malformed_hosts(self):
        """Test validation with malformed host strings."""
        patterns = ["localhost"]
        assert validate_host_against_patterns("..invalid..", patterns) is False
        assert validate_host_against_patterns("invalid..domain", patterns) is False


class TestValidateTrustedHostPatterns:
    """Test trusted host pattern validation."""

    def test_valid_patterns(self):
        """Test validation of valid patterns."""
        patterns = ["localhost", "*.example.com", "api.mysite.org"]
        errors = validate_trusted_host_patterns(patterns)
        assert errors == []

    def test_empty_patterns_list(self):
        """Test validation of empty patterns list."""
        errors = validate_trusted_host_patterns([])
        assert len(errors) == 1
        assert "No trusted host patterns provided" in errors[0]

    def test_empty_pattern_strings(self):
        """Test validation with empty pattern strings."""
        patterns = ["localhost", "", "  ", "api.example.com"]
        errors = validate_trusted_host_patterns(patterns)
        assert len(errors) == 2
        assert "Empty pattern at index 1" in errors
        assert "Empty pattern at index 2" in errors

    def test_invalid_patterns(self):
        """Test validation of invalid patterns."""
        patterns = ["localhost", "invalid..domain", "*."]
        errors = validate_trusted_host_patterns(patterns)
        assert len(errors) == 2
        assert any("invalid..domain" in error for error in errors)
        assert any("*." in error for error in errors)

    def test_universal_wildcard_allowed(self):
        """Test that universal wildcard is allowed."""
        patterns = ["*"]
        errors = validate_trusted_host_patterns(patterns)
        assert errors == []

    def test_valid_wildcard_patterns(self):
        """Test validation of valid wildcard patterns."""
        patterns = ["*.example.com", "*.api.mysite.org"]
        errors = validate_trusted_host_patterns(patterns)
        assert errors == []

    def test_invalid_wildcard_patterns(self):
        """Test validation of invalid wildcard patterns."""
        patterns = ["*.", "*.invalid..domain"]
        errors = validate_trusted_host_patterns(patterns)
        assert len(errors) == 2


class TestIntegrationSecurityScenarios:
    """Integration tests for security scenarios."""

    def test_subdomain_bypass_prevention(self):
        """Test that subdomain bypass attacks are prevented."""
        patterns = ["*.example.com"]

        # These should NOT match (bypass attempts)
        assert validate_host_against_patterns("example.com.evil.com", patterns) is False
        assert validate_host_against_patterns("notexample.com", patterns) is False
        assert validate_host_against_patterns("api.notexample.com", patterns) is False

    def test_port_stripping_security(self):
        """Test that port numbers are properly stripped for security."""
        patterns = ["localhost"]

        # All these should match (ports stripped)
        assert validate_host_against_patterns("localhost:5000", patterns) is True
        assert validate_host_against_patterns("localhost:8080", patterns) is True
        assert validate_host_against_patterns("localhost:443", patterns) is True

    def test_case_normalization_security(self):
        """Test that case normalization prevents bypasses."""
        patterns = ["api.example.com"]

        # All these should match (case normalized)
        assert validate_host_against_patterns("API.EXAMPLE.COM", patterns) is True
        assert validate_host_against_patterns("Api.Example.Com", patterns) is True
        assert validate_host_against_patterns("api.EXAMPLE.com", patterns) is True

    def test_ipv6_handling_security(self):
        """Test secure handling of IPv6 addresses."""
        patterns = ["::1"]

        # Should match
        assert validate_host_against_patterns("[::1]:8080", patterns) is True
        assert validate_host_against_patterns("::1", patterns) is True

        # Should not match
        assert validate_host_against_patterns("[::2]:8080", patterns) is False

    def test_comprehensive_attack_prevention(self):
        """Test prevention of various attack vectors."""
        patterns = ["localhost", "*.example.com"]

        # Valid requests should pass
        assert validate_host_against_patterns("localhost:5000", patterns) is True
        assert validate_host_against_patterns("api.example.com", patterns) is True

        # Attack attempts should fail
        attack_hosts = [
            "evil.com",
            "localhost.evil.com",
            "example.com.evil.com",
            "api.example.com.evil.com",
            "notlocalhost",
            "not-example.com",
            "..localhost",
            "localhost..",
            "",
            "malformed..host",
        ]

        for attack_host in attack_hosts:
            assert validate_host_against_patterns(attack_host,
                                                  patterns) is False, f"Attack host '{attack_host}' should not match"
