"""Server components for ChatEnv.

This module is optional and requires torch. We guard imports so the OpenEnv
package can be imported and tested without torch installed.
"""

try:
    from .chat_environment import ChatEnvironment  # noqa: F401
except ModuleNotFoundError as e:
    if e.name == "torch":
        ChatEnvironment = None  # type: ignore
    else:
        raise

__all__ = ["ChatEnvironment"]

