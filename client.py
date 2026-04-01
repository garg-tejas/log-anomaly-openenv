"""
OpenEnv-Compliant WebSocket Client for Log Anomaly Investigation Environment.

This module provides a proper OpenEnv client that connects via WebSocket
to the Log Anomaly environment, either on HuggingFace Spaces or locally.

Usage:
    # Async (recommended)
    async with LogAnomalyEnvClient() as env:
        result = await env.reset(difficulty="easy")
        result = await env.step(LogAction(action_type="bash", command="head log.txt"))

    # Sync wrapper
    with LogAnomalyEnvClient().sync() as env:
        result = env.reset(difficulty="easy")
        result = env.step(LogAction(action_type="bash", command="head log.txt"))
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

# Add models to path
_parent_dir = os.path.dirname(os.path.abspath(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models import LogAction, LogObservation, LogState

logger = logging.getLogger(__name__)

# Default HuggingFace Space URL
DEFAULT_ENV_URL = "https://ggtejas-log-anomaly-env.hf.space"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 10.0


class LogAnomalyEnvClient(EnvClient[LogAction, LogObservation, LogState]):
    """
    WebSocket client for the Log Anomaly Investigation environment.

    Inherits from OpenEnv's EnvClient for full compliance with the
    OpenEnv ecosystem. Connects via WebSocket for stateful sessions.

    Args:
        base_url: Environment server URL (default: HuggingFace Space)
        connect_timeout_s: Timeout for WebSocket connection (default: 30s)
        message_timeout_s: Timeout for message responses (default: 120s)
        max_retries: Maximum retry attempts on connection failure (default: 3)

    Example:
        async with LogAnomalyEnvClient() as env:
            result = await env.reset(difficulty="easy")
            while not result.done:
                action = LogAction(action_type="bash", command="grep ERROR log.txt")
                result = await env.step(action)
            print(f"Episode reward: {result.reward}")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_ENV_URL,
        connect_timeout_s: float = 30.0,
        message_timeout_s: float = 120.0,
        max_retries: int = MAX_RETRIES,
        **kwargs: Any,
    ):
        """Initialize the Log Anomaly environment client."""
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            **kwargs,
        )
        self._max_retries = max_retries
        self._current_state: Optional[LogState] = None

    def _step_payload(self, action: LogAction) -> Dict[str, Any]:
        """
        Convert LogAction to JSON payload for the server.

        Args:
            action: The LogAction to convert

        Returns:
            Dictionary ready to be sent as JSON
        """
        # Use model_dump to serialize, excluding None values for cleaner payload
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LogObservation]:
        """
        Parse server response into StepResult[LogObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult containing observation, reward, and done status
        """
        # Server returns: {"observation": {...}, "reward": float, "done": bool}
        obs_data = payload.get("observation", payload)

        # Build LogObservation from response data
        observation = LogObservation(
            command_output=obs_data.get("command_output", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            steps_remaining=obs_data.get("steps_remaining", 20),
            total_steps=obs_data.get("total_steps", 20),
            answer_submitted=obs_data.get("answer_submitted", False),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> LogState:
        """
        Parse state response from server.

        Args:
            payload: JSON response from state endpoint

        Returns:
            LogState object
        """
        return LogState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            log_file_path=payload.get("log_file_path"),
            task_id=payload.get("task_id"),
            mode=payload.get("mode", "eval"),
            data_source=payload.get("data_source", "synthetic"),
            log_source=payload.get("log_source"),
            last_exit_code=payload.get("last_exit_code", 0),
        )

    async def _send_with_retry(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send message with exponential backoff retry on failure.

        Args:
            message: Message to send

        Returns:
            Server response

        Raises:
            ConnectionError: If all retries fail
        """
        last_error: Optional[Exception] = None
        backoff = INITIAL_BACKOFF_S

        for attempt in range(self._max_retries):
            try:
                return await self._send_and_receive(message)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Connection failed (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {backoff:.1f}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_S)

                    # Try to reconnect
                    try:
                        await self.disconnect()
                        await self.connect()
                    except Exception as reconnect_error:
                        logger.warning(f"Reconnection failed: {reconnect_error}")

        raise ConnectionError(f"Failed after {self._max_retries} attempts: {last_error}")

    async def reset(self, **kwargs: Any) -> StepResult[LogObservation]:
        """
        Reset the environment with optional parameters.

        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            seed: Random seed for reproducibility
            mode: Environment mode ("training" or "eval")
            **kwargs: Additional parameters

        Returns:
            StepResult containing initial observation
        """
        message = {
            "type": "reset",
            "data": kwargs,
        }
        response = await self._send_with_retry(message)
        result = self._parse_result(response.get("data", {}))

        # Store state for get_result() compatibility
        self._current_state = LogState(
            episode_id=kwargs.get("episode_id"),
            step_count=0,
        )

        return result

    async def step(self, action: LogAction, **kwargs: Any) -> StepResult[LogObservation]:
        """
        Execute an action in the environment.

        Args:
            action: LogAction to execute (bash command or submit answer)
            **kwargs: Additional parameters (currently ignored)

        Returns:
            StepResult containing observation, reward, and done status
        """
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = await self._send_with_retry(message)
        return self._parse_result(response.get("data", {}))

    async def state(self) -> LogState:
        """
        Get the current environment state from the server.

        Returns:
            LogState with current environment state
        """
        message = {"type": "state"}
        response = await self._send_with_retry(message)
        state = self._parse_state(response.get("data", {}))
        self._current_state = state
        return state


# Alias for backward compatibility
LogAnomalyEnv = LogAnomalyEnvClient


@dataclass
class LocalEnvWrapper:
    """
    Async wrapper for local LogAnomalyEnvironment.

    Provides async interface for the sync local environment,
    allowing unified code path for both local and remote usage.

    Usage:
        from server.log_anomaly_environment import LogAnomalyEnvironment
        env = LocalEnvWrapper(LogAnomalyEnvironment())
        async with env:
            result = await env.reset(difficulty="easy")
    """

    _env: Any  # LogAnomalyEnvironment

    async def reset(self, **kwargs: Any) -> StepResult[LogObservation]:
        """Reset the local environment."""
        result = self._env.reset(**kwargs)
        return self._convert_result(result)

    async def step(self, action: LogAction, **kwargs: Any) -> StepResult[LogObservation]:
        """Execute action in local environment."""
        # Convert LogAction to InvestigationAction for local env
        from models import InvestigationAction, BashCommand, SubmitAnswer, AnomalyType

        if action.action_type == "bash":
            inv_action = InvestigationAction(
                action_type="bash",
                bash_command=BashCommand(command=action.command or ""),
            )
        elif action.action_type == "submit":
            inv_action = InvestigationAction(
                action_type="submit",
                answer=SubmitAnswer(
                    anomaly_type=AnomalyType(action.anomaly_type or "error_spike"),
                    component=action.component or "unknown",
                    start_time=action.start_time or "",
                    end_time=action.end_time or "",
                    confidence=action.confidence,
                ),
            )
        else:
            inv_action = InvestigationAction(action_type=action.action_type)

        result = self._env.step(inv_action)
        return self._convert_result(result)

    async def state(self) -> LogState:
        """Get current state from local environment."""
        state = self._env.state
        return LogState(
            episode_id=getattr(state, "episode_id", None),
            step_count=getattr(state, "step_count", 0),
            log_file_path=getattr(state, "log_file_path", None),
            task_id=getattr(state, "task_id", None),
            mode=getattr(state, "mode", "eval"),
            data_source=getattr(state, "data_source", "synthetic"),
            log_source=getattr(state, "log_source", None),
            last_exit_code=getattr(state, "last_exit_code", 0),
        )

    async def close(self) -> None:
        """Close the local environment."""
        pass  # Local env doesn't need explicit close

    async def __aenter__(self) -> "LocalEnvWrapper":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()

    def _convert_result(self, result: Any) -> StepResult[LogObservation]:
        """Convert local env result to StepResult."""
        # Handle different result types
        if isinstance(result, dict):
            obs_data = result.get("observation", result)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
        elif hasattr(result, "model_dump"):
            data = result.model_dump()
            obs_data = data
            reward = data.get("reward", 0.0)
            done = data.get("done", data.get("answer_submitted", False))
        else:
            obs_data = result
            reward = getattr(result, "reward", 0.0) or 0.0
            done = getattr(result, "done", False) or getattr(result, "answer_submitted", False)

        # Build observation
        if isinstance(obs_data, dict):
            observation = LogObservation(
                command_output=obs_data.get("command_output", ""),
                stderr=obs_data.get("stderr", ""),
                exit_code=obs_data.get("exit_code", 0),
                steps_remaining=obs_data.get("steps_remaining", 20),
                total_steps=obs_data.get("total_steps", 20),
                answer_submitted=obs_data.get("answer_submitted", False),
                task_difficulty=obs_data.get("task_difficulty", "easy"),
                done=done,
                reward=reward,
                metadata=obs_data.get("metadata", {}),
            )
        else:
            observation = LogObservation(
                command_output=getattr(obs_data, "command_output", ""),
                stderr=getattr(obs_data, "stderr", ""),
                exit_code=getattr(obs_data, "exit_code", 0),
                steps_remaining=getattr(obs_data, "steps_remaining", 20),
                total_steps=getattr(obs_data, "total_steps", 20),
                answer_submitted=getattr(obs_data, "answer_submitted", False),
                task_difficulty=str(getattr(obs_data, "task_difficulty", "easy")),
                done=done,
                reward=reward,
                metadata=getattr(obs_data, "metadata", {}),
            )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def get_result(self):
        """Get episode result from local environment."""
        return self._env.get_result()

    @property
    def state_sync(self):
        """Get state synchronously (for compatibility)."""
        return self._env.state


# Export main classes
__all__ = [
    "LogAnomalyEnvClient",
    "LogAnomalyEnv",
    "LocalEnvWrapper",
    "LogAction",
    "LogObservation",
    "LogState",
    "DEFAULT_ENV_URL",
]
