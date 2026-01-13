import os
import sys
import pytest

# Skip this module - websearch_env uses deprecated openenv_core imports
pytest.skip(
    "websearch_env uses deprecated openenv_core imports (optional plugin environment)",
    allow_module_level=True,
)


@pytest.mark.skipif(
    not os.environ.get("SERPER_API_KEY"), reason="SERPER_API_KEY not set"
)
def test_websearch_environment():
    # Create the environment
    env = WebSearchEnvironment()

    # Reset the environment
    obs: WebSearchObservation = env.reset()
    assert obs.web_contents == []
    assert obs.content == ""

    # Step the environment
    obs: WebSearchObservation = env.step(
        WebSearchAction(query="What is the capital of France?")
    )
    if not obs.metadata.get("error"):
        assert obs.web_contents != []
        assert len(obs.web_contents) == 5
        assert obs.metadata == {"query": "What is the capital of France?"}
    else:
        assert obs.web_contents == []
        assert "[ERROR]" in obs.content
