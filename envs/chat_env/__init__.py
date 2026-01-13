"""Chat Environment - A chat-based environment for LLMs with tokenization support."""

try:
    from .models import ChatAction, ChatObservation, ChatState  # noqa: F401
    from .client import ChatEnv  # noqa: F401
except ModuleNotFoundError as e:
    # Allow importing envs.chat_env without torch installed.
    if e.name == "torch":
        ChatAction = None  # type: ignore
        ChatObservation = None  # type: ignore
        ChatState = None  # type: ignore
        ChatEnv = None  # type: ignore
    else:
        raise

__all__ = ["ChatAction", "ChatObservation", "ChatState", "ChatEnv"]

