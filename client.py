"""
Log Anomaly Investigation Environment - Client.

This module provides the client for connecting to the Log Anomaly
Investigation Environment server using the OpenEnv spec.

Example:
    >>> from log_anomaly_env import LogAnomalyEnv, LogAction
    >>>
    >>> with LogAnomalyEnv(base_url="http://localhost:8000").sync() as env:
    ...     env.reset(difficulty="easy")
    ...     result = env.step(LogAction(action_type="bash", command="grep ERROR log.txt | head -5"))
    ...     print(result.command_output)
"""

import os
import sys
from typing import Any, Dict, Optional

from openenv.core.env_client import EnvClient

# Add models to path
_parent_dir = os.path.dirname(os.path.abspath(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models import LogAction, LogObservation, LogState


class LogAnomalyEnv(EnvClient):
    """
    Client for the Log Anomaly Investigation Environment.

    This client extends OpenEnv's EnvClient to provide WebSocket-based
    interactions with the Log Anomaly Investigation environment.

    Example using async:
        >>> async with LogAnomalyEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(difficulty="easy")
        ...     result = await env.step(LogAction(action_type="bash", command="grep ERROR log.txt"))
        ...     print(result.observation.command_output)

    Example using sync:
        >>> with LogAnomalyEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(difficulty="easy")
        ...     result = env.step(LogAction(action_type="bash", command="grep ERROR log.txt"))
        ...     print(result.observation.command_output)

    Example with HuggingFace Space:
        >>> env = LogAnomalyEnv.from_env("openenv/log-anomaly-env")
        >>> env.reset(difficulty="medium")
    """

    pass  # EnvClient provides all needed functionality


# Export main classes
__all__ = ["LogAnomalyEnv", "LogAction", "LogObservation", "LogState"]
