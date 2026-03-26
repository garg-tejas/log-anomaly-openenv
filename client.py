"""
Log Anomaly Investigation Environment - Client.

This module provides the client for connecting to the Log Anomaly
Investigation Environment server.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import requests

# Support both package and direct execution modes
if __package__:
    from .models import (
        AnomalyType,
        BashCommand,
        DifficultyLevel,
        EpisodeResult,
        InvestigationAction,
        InvestigationObservation,
        InvestigationState,
        SubmitAnswer,
    )
else:
    _parent_dir = os.path.dirname(os.path.abspath(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from models import (
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
    Client for the Log Anomaly Investigation Environment.

    This client connects to the environment server and provides methods
    for interacting with the investigation environment.

    Example:
        >>> client = LogAnomalyEnv(base_url="http://localhost:8000")
        >>> client.connect()
        >>> result = client.reset(difficulty="easy")
        >>> print(result.observation.steps_remaining)
        15
        >>> result = client.step(InvestigationAction(
        ...     action_type="bash",
        ...     bash_command=BashCommand(command="grep ERROR log.txt | head -20")
        ... ))
        >>> print(result.observation.command_output)
        2024-01-15 10:23:45 ERROR component_a: Connection timeout
        ...
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

    def connect(self) -> None:
        """Establish connection to the server."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to environment: {e}")

    def reset(
        self, difficulty: str = "easy", task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reset the environment and start a new episode.

        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            task_id: Optional specific task ID to run
            seed: Optional random seed

        Returns:
            Dictionary containing observation and metadata
        """
        response = self.session.post(
            f"{self.base_url}/reset",
            json={
                "difficulty": difficulty,
                "task_id": task_id,
                "seed": seed,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: InvestigationAction) -> Dict[str, Any]:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            Dictionary containing observation and reward
        """
        response = self.session.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
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

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of available tasks.

        Returns:
            List of task definitions
        """
        response = self.session.get(f"{self.base_url}/tasks", timeout=5)
        response.raise_for_status()
        return response.json()

    def grade(self, episode_id: str) -> EpisodeResult:
        """
        Get the grading result for an episode.

        Args:
            episode_id: ID of the episode to grade

        Returns:
            Episode result with scores
        """
        response = self.session.get(f"{self.base_url}/grade/{episode_id}", timeout=5)
        response.raise_for_status()
        return response.json()

    def run_baseline(self, difficulty: str = "all") -> Dict[str, Any]:
        """
        Run the baseline inference script.

        Args:
            difficulty: Difficulty level to test ("easy", "medium", "hard", or "all")

        Returns:
            Baseline results for all tasks
        """
        response = self.session.post(
            f"{self.base_url}/baseline",
            json={"difficulty": difficulty},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the connection."""
        self.session.close()
        self._connected = False

    @classmethod
    def from_env(cls, repo_id: str) -> "LogAnomalyEnv":
        """
        Create client from HuggingFace Space.

        Args:
            repo_id: HuggingFace Space repository ID (e.g., "username/log-anomaly-env")

        Returns:
            Configured client instance
        """
        # HuggingFace Spaces have URLs like: https://{username}-{space-name}.hf.space
        # repo_id format is "username/space-name"
        if "/" not in repo_id:
            raise ValueError(f"Invalid repo_id format: {repo_id}. Expected 'username/space-name'")

        username, space_name = repo_id.split("/", 1)
        # HuggingFace converts underscores to hyphens in Space URLs
        space_name = space_name.replace("_", "-")
        base_url = f"https://{username}-{space_name}.hf.space"
        return cls(base_url=base_url)

    @classmethod
    def from_docker_image(cls, image_name: str, port: int = 8000) -> "LogAnomalyEnv":
        """
        Create client from Docker image.

        Args:
            image_name: Docker image name
            port: Port to connect on

        Returns:
            Configured client instance
        """
        # This would typically start the container
        # For now, assume container is already running
        base_url = f"http://localhost:{port}"
        return cls(base_url=base_url)

    def __enter__(self) -> "LogAnomalyEnv":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
