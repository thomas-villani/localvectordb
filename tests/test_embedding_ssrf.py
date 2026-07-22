"""B5: SSRF guard for request-supplied embedding provider URLs.

POST /databases forwards an embedding ``base_url`` into server-side HTTP calls
(validate_model hits ``{base_url}/api/tags``). On an auth-off-by-default server
an unauthenticated caller could otherwise probe internal services or cloud
metadata. The gate reads its policy from the *trusted server config* so a caller
cannot flip ``allow_custom_provider_url`` on itself via the merged request body.
"""

import pytest

from localvectordb_server._error_handlers import APIError, ValidationError
from localvectordb_server.config import EmbeddingSettings
from localvectordb_server.routers.databases import _enforce_embedding_url_policy


def _server(**kwargs) -> EmbeddingSettings:
    s = EmbeddingSettings()
    s.base_url = kwargs.get("base_url")
    s.allow_custom_provider_url = kwargs.get("allow_custom_provider_url", False)
    s.allowed_provider_hosts = kwargs.get("allowed_provider_hosts")
    if "config" in kwargs:
        s.config = kwargs["config"]
    return s


class TestEmbeddingUrlSSRFPolicy:
    def test_no_request_url_is_allowed(self):
        # A request that doesn't override the provider URL is always fine.
        _enforce_embedding_url_policy({"provider": "ollama"}, _server())

    def test_custom_url_denied_by_default(self):
        with pytest.raises(APIError) as exc:
            _enforce_embedding_url_policy(
                {"base_url": "http://169.254.169.254"}, _server(allow_custom_provider_url=False)
            )
        assert exc.value.status_code == 403

    def test_nested_config_url_is_also_gated(self):
        # base_url spread from embedding.config is a sink too (see _dbmanager).
        with pytest.raises(APIError) as exc:
            _enforce_embedding_url_policy(
                {"config": {"base_url": "http://internal.svc:8080"}}, _server(allow_custom_provider_url=False)
            )
        assert exc.value.status_code == 403

    def test_request_matching_server_url_is_allowed(self):
        # Same value as the operator's configured URL is a no-op, not an override.
        _enforce_embedding_url_policy(
            {"base_url": "http://localhost:11434"},
            _server(base_url="http://localhost:11434", allow_custom_provider_url=False),
        )

    def test_server_config_dict_url_is_trusted(self):
        _enforce_embedding_url_policy(
            {"base_url": "http://ollama:11434"},
            _server(config={"base_url": "http://ollama:11434"}, allow_custom_provider_url=False),
        )

    def test_enabled_allows_custom_url(self):
        _enforce_embedding_url_policy(
            {"base_url": "http://ollama.internal:11434"},
            _server(allow_custom_provider_url=True),
        )

    def test_enabled_but_non_http_scheme_rejected(self):
        with pytest.raises(ValidationError):
            _enforce_embedding_url_policy({"base_url": "file:///etc/passwd"}, _server(allow_custom_provider_url=True))

    def test_enabled_but_host_not_in_allowlist_rejected(self):
        with pytest.raises(APIError) as exc:
            _enforce_embedding_url_policy(
                {"base_url": "http://evil.example.com"},
                _server(allow_custom_provider_url=True, allowed_provider_hosts=["ollama.internal"]),
            )
        assert exc.value.status_code == 403

    def test_enabled_and_host_in_allowlist_allowed(self):
        _enforce_embedding_url_policy(
            {"base_url": "http://ollama.internal:11434"},
            _server(allow_custom_provider_url=True, allowed_provider_hosts=["ollama.internal"]),
        )

    def test_wildcard_allowlist_pattern(self):
        _enforce_embedding_url_policy(
            {"base_url": "http://api.ollama.example.com"},
            _server(allow_custom_provider_url=True, allowed_provider_hosts=["*.ollama.example.com"]),
        )
