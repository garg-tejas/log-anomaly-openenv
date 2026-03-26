# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Log Anomaly Investigation Environment.

A real-world OpenEnv environment for training AI agents to investigate
log anomalies using bash command exploration.

Example:
    >>> from log_anomaly_env import LogAnomalyEnv, InvestigationAction, BashCommand
    >>>
    >>> with LogAnomalyEnv(base_url="http://localhost:8000") as env:
    ...     result = env.reset(difficulty="easy")
    ...     print(f"Steps remaining: {result['observation']['steps_remaining']}")
    ...
    ...     # Execute a bash command
    ...     action = InvestigationAction(
    ...         action_type="bash",
    ...         bash_command=BashCommand(command="grep ERROR log.txt | head -20")
    ...     )
    ...     result = env.step(action)
    ...     print(result['observation']['command_output'][:200])
"""
from .models import (
    InvestigationAction,
    InvestigationObservation,
    InvestigationState,
    EpisodeResult,
    DifficultyLevel,
    AnomalyType,
    SubmitAnswer,
    BashCommand,
    LogLine,
)
from .client import LogAnomalyEnv

__all__ = [
    "LogAnomalyEnv",
    "InvestigationAction",
    "InvestigationObservation",
    "InvestigationState",
    "EpisodeResult",
    "DifficultyLevel",
    "AnomalyType",
    "SubmitAnswer",
    "BashCommand",
    "LogLine",
]
