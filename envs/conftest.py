# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest configuration for envs directory.

Skips test collection for optional environment plugins that require
heavy dependencies (torch, nltk, numpy, etc.) that aren't installed.
"""

import os


def pytest_ignore_collect(collection_path, config):
    """Ignore environment directories if their dependencies aren't installed."""
    path_str = str(collection_path)
    
    # Skip chat_env if torch is not installed
    if "chat_env" in path_str:
        try:
            import torch  # noqa: F401
            return False
        except ImportError:
            return True
    
    # Skip textarena_env if nltk is not installed  
    if "textarena_env" in path_str:
        try:
            import nltk  # noqa: F401
            return False
        except ImportError:
            return True
    
    # Skip connect4_env if numpy is not installed
    if "connect4_env" in path_str:
        try:
            import numpy  # noqa: F401
            return False
        except ImportError:
            return True
    
    # Skip websearch_env (uses deprecated openenv_core imports)
    if "websearch_env" in path_str:
        return True
    
    return False
