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

# Try to use OpenEnv's EnvClient
try:
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

except ImportError:
    # Fallback: Use HTTP-based client for standalone usage
    import requests

    _parent_dir = os.path.dirname(os.path.abspath(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)

    from models import (
        LogAction,
        LogObservation,
        AnomalyType,
        BashCommand,
        DifficultyLevel,
        EpisodeResult,
        InvestigationAction,
        InvestigationObservation,
        InvestigationState,
        SubmitAnswer,
    )

    class LogAnomalyEnv:
        """
        HTTP-based client for the Log Anomaly Investigation Environment.

        This is a fallback client that uses HTTP REST endpoints when
        openenv-core is not installed.

        Example:
            >>> client = LogAnomalyEnv(base_url="http://localhost:8000")
            >>> client.connect()
            >>> result = client.reset(difficulty="easy")
            >>> print(result["observation"]["steps_remaining"])
            15
        """

        def __init__(self, base_url: str = "http://localhost:8000"):
            """
            Initialize the client.

            Args:
                base_url: URL of the environment server
            """
            self.base_url = base_url.rstrip("/")
            self.session = requests.Session()
            self._connected = False
            self._episode_id: Optional[str] = None

        def connect(self) -> None:
            """Establish connection to the server."""
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                response.raise_for_status()
                self._connected = True
            except Exception as e:
                raise ConnectionError(f"Failed to connect to environment: {e}")

        def reset(
            self,
            difficulty: str = "easy",
            seed: Optional[int] = None,
            task_id: Optional[str] = None,
            mode: str = "eval",
            **kwargs: Any,
        ) -> Dict[str, Any]:
            """
            Reset the environment and start a new episode.

            Args:
                difficulty: Difficulty level ("easy", "medium", "hard")
                seed: Optional random seed
                task_id: Optional specific task ID
                mode: Operating mode ("training" or "eval")
                **kwargs: Additional reset parameters

            Returns:
                Dictionary containing observation and metadata
            """
            response = self.session.post(
                f"{self.base_url}/reset",
                json={
                    "difficulty": difficulty,
                    "seed": seed,
                    "task_id": task_id,
                    "mode": mode,
                    **kwargs,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            self._episode_id = data.get("episode_id")
            return data

        def step(self, action: LogAction) -> Dict[str, Any]:
            """
            Execute an action in the environment.

            Args:
                action: The LogAction to execute

            Returns:
                Dictionary containing observation and reward
            """
            # Convert action to dict for HTTP request
            action_dict = {
                "action_type": action.action_type,
                "episode_id": self._episode_id,
            }

            if action.action_type == "bash":
                action_dict["command"] = action.command
            elif action.action_type == "submit":
                action_dict["anomaly_type"] = action.anomaly_type
                action_dict["component"] = action.component
                action_dict["start_time"] = action.start_time
                action_dict["end_time"] = action.end_time
                action_dict["confidence"] = action.confidence

            response = self.session.post(
                f"{self.base_url}/step",
                json=action_dict,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        def state(self) -> Dict[str, Any]:
            """
            Get the current environment state.

            Returns:
                Dictionary containing current state
            """
            response = self.session.get(f"{self.base_url}/state", timeout=5)
            response.raise_for_status()
            return response.json()

        def get_tasks(self) -> Dict[str, Any]:
            """
            Get list of available tasks.

            Returns:
                Dictionary containing task definitions
            """
            response = self.session.get(f"{self.base_url}/tasks", timeout=5)
            response.raise_for_status()
            return response.json()

        def grade(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
            """
            Get the grading result for an episode.

            Args:
                episode_id: ID of the episode to grade (uses current if None)

            Returns:
                Episode result with scores
            """
            ep_id = episode_id or self._episode_id
            if not ep_id:
                raise ValueError("No episode_id provided and no active episode")
            response = self.session.get(f"{self.base_url}/grade/{ep_id}", timeout=5)
            response.raise_for_status()
            return response.json()

        def run_baseline(
            self,
            difficulty: str = "all",
            model: str = "Qwen/Qwen3.5-2B",
            num_episodes: int = 5,
        ) -> Dict[str, Any]:
            """
            Run the baseline inference script.

            Args:
                difficulty: Difficulty level to test
                model: Model to use for inference
                num_episodes: Number of episodes to run

            Returns:
                Baseline results
            """
            response = self.session.post(
                f"{self.base_url}/baseline",
                json={
                    "difficulty": difficulty,
                    "model": model,
                    "num_episodes": num_episodes,
                },
                timeout=600,
            )
            response.raise_for_status()
            return response.json()

        def close(self) -> None:
            """Close the connection."""
            self.session.close()
            self._connected = False

        def sync(self) -> "LogAnomalyEnv":
            """Return self for compatibility with OpenEnv sync() pattern."""
            return self

        @classmethod
        def from_env(cls, repo_id: str) -> "LogAnomalyEnv":
            """
            Create client from HuggingFace Space.

            Args:
                repo_id: HuggingFace Space repository ID (e.g., "openenv/log-anomaly-env")

            Returns:
                Configured client instance
            """
            if "/" not in repo_id:
                raise ValueError(
                    f"Invalid repo_id format: {repo_id}. Expected 'username/space-name'"
                )

            username, space_name = repo_id.split("/", 1)
            space_name = space_name.replace("_", "-")
            base_url = f"https://{username}-{space_name}.hf.space"
            return cls(base_url=base_url)

        @classmethod
        def from_docker_image(cls, image_name: str, port: int = 8000) -> "LogAnomalyEnv":
            """
            Create client from Docker image (assumes container is running).

            Args:
                image_name: Docker image name
                port: Port to connect on

            Returns:
                Configured client instance
            """
            return cls(base_url=f"http://localhost:{port}")

        def __enter__(self) -> "LogAnomalyEnv":
            """Context manager entry."""
            self.connect()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            """Context manager exit."""
            self.close()


# Export main classes
__all__ = ["LogAnomalyEnv", "LogAction", "LogObservation", "LogState"]
