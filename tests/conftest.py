"""Test-wide defaults.

`.env` configures the machine, not the test suite. Without pinning the provider here, whether
a test hits a network, a local Ollama server, or neither depends on what happens to be in
`.env` — which makes a green suite a statement about this laptop rather than about the code.
"""
import pytest

from agentic_macro import config


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch):
    """No test may reach a model unless it opts in by patching the seam itself."""
    monkeypatch.setattr(config, "MEMORY_ENABLED", False)

    def refuse(*a, **k):
        raise AssertionError(
            "a test tried to reach a real model — patch providers.complete_json or "
            "providers.embed instead")

    from agentic_macro import providers
    monkeypatch.setattr(providers, "_ollama", refuse)
    monkeypatch.setattr(providers, "_embed_gemini", refuse)
    monkeypatch.setattr(providers, "_complete_gemini", refuse)
