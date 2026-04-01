"""
TRL-Compatible Training Client for Log Anomaly Investigation Environment.

This module provides environment factory classes that integrate with TRL's
GRPOTrainer for reinforcement learning training of log investigation agents.

Usage with TRL:
    from training_client import LogAnomalyTrainingEnv, CurriculumLogAnomalyEnv
    from trl import GRPOTrainer, GRPOConfig

    def reward_func(environments, **kwargs):
        return [env.reward for env in environments]

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B",
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=LogAnomalyTrainingEnv,  # or CurriculumLogAnomalyEnv
    )
    trainer.train()
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.abspath(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models import LogAction, LogObservation, DifficultyLevel, AnomalyType
from config import MAX_STEPS, get_logger

logger = get_logger(__name__)


class LogAnomalyTrainingEnv:
    """
    TRL-compatible environment factory for GRPO training.

    This class wraps the LogAnomalyEnvironment to expose bash commands
    and submission as tool methods that TRL's GRPOTrainer can discover
    through introspection.

    Tool Discovery:
        TRL automatically discovers public methods (not starting with _)
        other than reset() and exposes them as function-calling tools.
        Each method must have:
        - Type-annotated arguments
        - Proper docstring with Args: section

    Attributes:
        reward: Final reward for the episode (read by reward_func)
        done: Whether the episode has ended
        difficulty: Current difficulty level

    Example:
        env = LogAnomalyTrainingEnv()
        obs = env.reset(difficulty="easy")
        output = env.bash("grep ERROR log.txt | head -10")
        result = env.submit(
            anomaly_type="error_spike",
            component="service_a",
            start_time="2026-03-27T14:55:00",
            end_time="2026-03-27T14:57:00"
        )
        print(env.reward)  # 0.0 to 1.0
    """

    def __init__(self):
        """Initialize the training environment."""
        # Lazy import to avoid circular dependencies
        from server.log_anomaly_environment import LogAnomalyEnvironment

        self._env = LogAnomalyEnvironment()
        self.reward: float = 0.0
        self.done: bool = False
        self.difficulty: str = "easy"
        self._last_observation: Optional[LogObservation] = None
        self._step_count: int = 0

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None, **kwargs) -> str:
        """
        Start a new log investigation episode.

        This method is called at the beginning of each episode by TRL.
        It returns the initial observation as a string that will be
        shown to the model.

        Args:
            difficulty: Task difficulty ("easy", "medium", "hard")
            seed: Optional random seed for reproducibility
            **kwargs: Additional parameters (passed from dataset columns)

        Returns:
            Initial observation describing the task and available tools.
        """
        self.reward = 0.0
        self.done = False
        self.difficulty = difficulty
        self._step_count = 0

        # Reset the underlying environment
        obs = self._env.reset(
            difficulty=difficulty,
            seed=seed,
            mode="training",  # Enable full feedback
            **kwargs,
        )
        self._last_observation = obs

        return self._format_initial_observation(obs)

    def bash(self, command: str) -> str:
        """
        Execute a bash command to investigate the log file.

        Use standard Unix tools to search and analyze log.txt:
        - grep: Search for patterns (e.g., "grep ERROR log.txt")
        - awk: Extract fields (e.g., "awk '{print $3}' log.txt")
        - head/tail: View start/end (e.g., "head -20 log.txt")
        - sort/uniq: Find patterns (e.g., "grep ERROR log.txt | sort | uniq -c")
        - wc: Count lines (e.g., "wc -l log.txt")

        Args:
            command: The bash command to execute on log.txt

        Returns:
            The command output (stdout and stderr combined).

        Raises:
            ValueError: If the episode is already over.
        """
        if self.done:
            raise ValueError(
                "Episode is over. You already submitted an answer or ran out of steps. "
                "The episode has ended."
            )

        self._step_count += 1

        # Execute the command
        action = LogAction(action_type="bash", command=command)
        result = self._env.step(action)
        self._last_observation = result

        # Check if episode ended (ran out of steps without submitting)
        if result.steps_remaining <= 0 and not result.answer_submitted:
            self.done = True
            self.reward = 0.0  # No submission = no reward
            return (
                f"{result.command_output or '(no output)'}\n\n"
                "WARNING: You ran out of steps without submitting an answer. "
                "Episode ended with reward=0.0"
            )

        return result.command_output or "(no output)"

    def submit(
        self,
        anomaly_type: str,
        component: str,
        start_time: str,
        end_time: str,
        confidence: float = 1.0,
    ) -> str:
        """
        Submit your final answer about the detected anomaly.

        Call this when you have gathered enough evidence to identify:
        1. What type of anomaly occurred
        2. Which component was affected
        3. The time window when it happened

        Args:
            anomaly_type: Type of anomaly detected. Must be one of:
                - "error_spike": Sudden increase in error rates
                - "memory_leak": Gradual memory consumption increase
                - "latency_degradation": Response time degradation
                - "service_dropout": Service becoming unavailable
                - "cascade_failure": Failure propagating across components
            component: The affected component name (e.g., "service_a", "database")
            start_time: When the anomaly started (ISO format: "2026-03-27T14:55:00")
            end_time: When the anomaly ended (ISO format: "2026-03-27T14:57:00")
            confidence: Your confidence level (0.0 to 1.0, default 1.0)

        Returns:
            Feedback about your submission including the reward earned.

        Raises:
            ValueError: If the episode is already over.
        """
        if self.done:
            raise ValueError(
                "Episode is over. You already submitted an answer. "
                "Call reset() to start a new episode."
            )

        # Validate anomaly type
        valid_types = [t.value for t in AnomalyType]
        if anomaly_type not in valid_types:
            # Don't end episode, let them try again
            return (
                f"Invalid anomaly_type: '{anomaly_type}'. "
                f"Must be one of: {', '.join(valid_types)}. "
                "Please try again with a valid type."
            )

        self._step_count += 1

        # Submit the answer
        action = LogAction(
            action_type="submit",
            anomaly_type=anomaly_type,
            component=component,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
        )
        result = self._env.step(action)
        self._last_observation = result

        # Episode ends after submission
        self.done = True
        self.reward = result.reward if result.reward is not None else 0.0

        return self._format_submission_feedback(result)

    def _format_initial_observation(self, obs: LogObservation) -> str:
        """Format the initial observation for the model."""
        difficulty_hints = {
            "easy": "Look for ERROR patterns - this is a clear error_spike anomaly.",
            "medium": "Could be error_spike, memory_leak, latency_degradation, or service_dropout.",
            "hard": "This is a cascade_failure - find the root cause and propagation chain.",
        }

        hint = difficulty_hints.get(self.difficulty, difficulty_hints["medium"])

        return f"""LOG INVESTIGATION TASK
====================

Difficulty: {self.difficulty.upper()}
Steps available: {obs.steps_remaining}

OBJECTIVE: Investigate log.txt to identify an anomaly.
{hint}

AVAILABLE TOOLS:
1. bash(command) - Run commands like: grep ERROR log.txt | head -10
2. submit(anomaly_type, component, start_time, end_time) - Submit your answer

TIPS:
- Start with: bash("head -50 log.txt") to see log format
- Use grep to find errors: bash("grep -i error log.txt | head -20")
- Extract timestamps from ERROR lines for start_time and end_time
- Look for component names like service_a, service_b, etc.

BEGIN INVESTIGATION"""

    def _format_submission_feedback(self, result: LogObservation) -> str:
        """Format the submission feedback for the model."""
        reward = result.reward if result.reward is not None else 0.0

        # Get detailed feedback from metadata if available
        metadata = result.metadata or {}
        ground_truth = metadata.get("ground_truth", {})

        feedback_parts = [
            f"SUBMISSION RESULT",
            f"================",
            f"Reward: {reward:.4f}",
            "",
        ]

        if reward >= 0.8:
            feedback_parts.append("Excellent! You correctly identified the anomaly.")
        elif reward >= 0.5:
            feedback_parts.append("Partial credit. Some aspects were correct.")
        elif reward > 0:
            feedback_parts.append("Low score. Review the ground truth below.")
        else:
            feedback_parts.append("Incorrect. The answer did not match the anomaly.")

        # In training mode, show ground truth for learning
        if ground_truth:
            feedback_parts.extend(
                [
                    "",
                    "GROUND TRUTH (for learning):",
                    f"  Type: {ground_truth.get('anomaly_type', 'N/A')}",
                    f"  Component: {ground_truth.get('component', 'N/A')}",
                    f"  Start: {ground_truth.get('start_time', 'N/A')}",
                    f"  End: {ground_truth.get('end_time', 'N/A')}",
                ]
            )

        feedback_parts.append(f"\nEpisode complete. Steps used: {self._step_count}")

        return "\n".join(feedback_parts)


class CurriculumLogAnomalyEnv(LogAnomalyTrainingEnv):
    """
    Curriculum learning wrapper that progressively increases difficulty.

    This environment automatically selects difficulty based on the agent's
    recent performance, enabling smoother learning progression:

    - Episodes 0-20: Always easy (warmup phase)
    - Success rate < 40%: Stay on easy
    - Success rate 40-70%: Progress to medium
    - Success rate > 70%: Advance to hard

    Success is defined as reward >= 0.5.

    Attributes:
        episode_count: Total episodes played
        success_history: Deque of recent success/failure (1/0)
        window_size: Number of episodes to consider for success rate

    Example:
        env = CurriculumLogAnomalyEnv()
        # First 20 episodes are always easy
        obs = env.reset()  # difficulty auto-selected

        # As agent improves, difficulty increases
        for _ in range(100):
            obs = env.reset()  # Might be easy, medium, or hard
            # ... agent plays ...
    """

    # Curriculum thresholds
    WARMUP_EPISODES = 20
    MEDIUM_THRESHOLD = 0.4  # Success rate to advance to medium
    HARD_THRESHOLD = 0.7  # Success rate to advance to hard
    SUCCESS_THRESHOLD = 0.5  # Reward threshold to count as success
    WINDOW_SIZE = 10  # Episodes to consider for success rate

    def __init__(self):
        """Initialize the curriculum environment."""
        super().__init__()
        self.episode_count: int = 0
        self.success_history: Deque[int] = deque(maxlen=self.WINDOW_SIZE)
        self._current_auto_difficulty: str = "easy"

    def reset(self, difficulty: str = "auto", seed: Optional[int] = None, **kwargs) -> str:
        """
        Start a new episode with auto-selected or specified difficulty.

        If difficulty is "auto", it will be automatically selected
        based on the agent's recent performance (curriculum learning).

        Args:
            difficulty: Difficulty level or "auto" for curriculum selection
            seed: Optional random seed for reproducibility
            **kwargs: Additional parameters passed to parent reset

        Returns:
            Initial observation string.
        """
        # Record previous episode result if we have one
        if self._last_observation is not None and self.episode_count > 0:
            success = 1 if self.reward >= self.SUCCESS_THRESHOLD else 0
            self.success_history.append(success)

        self.episode_count += 1

        # Auto-select difficulty if not specified
        if difficulty == "auto":
            difficulty = self._select_difficulty()
            self._current_auto_difficulty = difficulty

        return super().reset(difficulty=difficulty, seed=seed, **kwargs)

    def _select_difficulty(self) -> str:
        """
        Select difficulty based on recent performance.

        Returns:
            Selected difficulty level ("easy", "medium", "hard")
        """
        # Warmup phase: always easy
        if self.episode_count <= self.WARMUP_EPISODES:
            logger.debug(f"Curriculum: Warmup phase (episode {self.episode_count}), using easy")
            return "easy"

        # Not enough history yet
        if len(self.success_history) < self.WINDOW_SIZE // 2:
            return "easy"

        # Calculate success rate
        success_rate = sum(self.success_history) / len(self.success_history)

        if success_rate >= self.HARD_THRESHOLD:
            selected = "hard"
        elif success_rate >= self.MEDIUM_THRESHOLD:
            selected = "medium"
        else:
            selected = "easy"

        logger.debug(f"Curriculum: success_rate={success_rate:.2f}, selected={selected}")
        return selected

    @property
    def current_success_rate(self) -> float:
        """Get current success rate over recent episodes."""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    @property
    def curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum learning statistics."""
        return {
            "episode_count": self.episode_count,
            "current_difficulty": self._current_auto_difficulty,
            "success_rate": self.current_success_rate,
            "history_size": len(self.success_history),
            "in_warmup": self.episode_count <= self.WARMUP_EPISODES,
        }


# Convenience function for creating reward function
def create_reward_func():
    """
    Create a reward function compatible with TRL's GRPOTrainer.

    Returns:
        A function that extracts rewards from environment instances.

    Example:
        trainer = GRPOTrainer(
            model="Qwen/Qwen3-4B",
            reward_funcs=create_reward_func(),
            environment_factory=LogAnomalyTrainingEnv,
        )
    """

    def reward_func(environments: List[LogAnomalyTrainingEnv], **kwargs) -> List[float]:
        """Extract rewards from environment instances."""
        return [env.reward for env in environments]

    return reward_func


# Export main classes
__all__ = [
    "LogAnomalyTrainingEnv",
    "CurriculumLogAnomalyEnv",
    "create_reward_func",
]
