# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Log Anomaly Investigation Environment.

A real-world OpenEnv environment for training AI agents to investigate
log anomalies using bash command exploration.

This environment simulates a realistic log investigation scenario where
an agent must identify anomalies in system logs using only read-only
bash commands.
"""
import os
import re
import subprocess
import tempfile
import uuid
import shutil
import sys
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from log_utils import (
    LogParser,
    AnomalyInjector,
    generate_synthetic_log,
)
from grader import InvestigationGrader, TaskGenerator
from models import (
    InvestigationAction,
    InvestigationObservation,
    InvestigationState,
    DifficultyLevel,
    AnomalyType,
    SubmitAnswer,
    EpisodeResult,
    LogLine,
)


class InvestigationEpisode:
    """Manages a single investigation episode."""

    MAX_STEPS = 15
    OUTPUT_TRUNCATION = 4000  # Max characters in command output

    # Allowed bash commands for security
    ALLOWED_COMMANDS = [
        "grep", "egrep", "fgrep",  # Pattern matching
        "awk", "sed",  # Text processing
        "sort", "uniq", "wc",  # Counting/aggregation
        "head", "tail", "cut",  # Selection
        "cat", "less", "more",  # Display
        "find", "xargs",  # File search
        "date", "echo",  # Utilities
        "ls", "pwd",  # Navigation (read-only)
    ]

    # Forbidden patterns (security)
    FORBIDDEN_PATTERNS = [
        r"rm\s+-",  # File deletion
        r"mv\s+",  # File move
        r"cp\s+",  # File copy
        r">\s*/",  # Writing to system dirs
        r"dd\s+",  # Direct I/O
        r"mkfs",  # Filesystem creation
        r":\(\)\{",  # Fork bomb
        r"wget.*\|",  # Download and execute
        r"curl.*\|",  # Download and execute
    ]

    def __init__(
        self,
        episode_id: str,
        difficulty: DifficultyLevel,
        log_content: List[LogLine],
        ground_truth: Dict[str, Any],
        sandbox_dir: str,
    ):
        """
        Initialize an investigation episode.

        Args:
            episode_id: Unique episode identifier
            difficulty: Task difficulty level
            log_content: Log lines for this episode
            ground_truth: Ground truth metadata
            sandbox_dir: Directory for sandboxed execution
        """
        self.episode_id = episode_id
        self.difficulty = difficulty
        self.log_lines = log_content
        self.ground_truth = ground_truth
        self.sandbox_dir = sandbox_dir
        self.log_filepath = os.path.join(sandbox_dir, "log.txt")

        # Episode state
        self.step_count = 0
        self.command_history: List[Dict[str, str]] = []
        self.answer_submitted = False
        self.predicted_answer: Optional[SubmitAnswer] = None
        self.episode_reward = 0.0

        # Initialize grader
        self.grader = InvestigationGrader()

        # Write log file
        self._write_log_file()

    def _write_log_file(self) -> None:
        """Write log content to the sandbox file."""
        os.makedirs(self.sandbox_dir, exist_ok=True)
        with open(self.log_filepath, "w", encoding="utf-8") as f:
            for line in self.log_lines:
                f.write(line.raw_line + "\n")

    def reset(self) -> InvestigationObservation:
        """
        Reset the episode state.

        Returns:
            Initial observation
        """
        self.step_count = 0
        self.command_history = []
        self.answer_submitted = False
        self.predicted_answer = None
        self.episode_reward = 0.0

        return InvestigationObservation(
            command_output="",
            command_history=[],
            steps_remaining=self.MAX_STEPS,
            total_steps=self.MAX_STEPS,
            answer_submitted=False,
            task_difficulty=self.difficulty,
            episode_reward=0.0,
            metadata={
                "log_file": "log.txt",
                "log_lines": len(self.log_lines),
                "ground_truth_hint": {
                    "component": self.ground_truth.get("component"),
                    "severity_hint": self._get_severity_hint(),
                },
            },
        )

    def _get_severity_hint(self) -> str:
        """Get a hint about the anomaly severity."""
        anomaly_type = self.ground_truth.get("anomaly_type")
        intensity = self.ground_truth.get("intensity", 0.5)

        if intensity >= 0.7:
            return "obvious"
        elif intensity >= 0.4:
            return "moderate"
        else:
            return "subtle"

    def step(self, action: InvestigationAction) -> InvestigationObservation:
        """
        Execute an action in the episode.

        Args:
            action: The action to execute

        Returns:
            Observation after the step
        """
        if self.answer_submitted:
            return InvestigationObservation(
                command_output="Episode already complete. Please reset.",
                command_history=self.command_history,
                steps_remaining=0,
                total_steps=self.MAX_STEPS,
                answer_submitted=True,
                task_difficulty=self.difficulty,
                episode_reward=self.episode_reward,
                metadata={"status": "episode_complete"},
            )

        self.step_count += 1

        if action.action_type == "submit":
            return self._handle_submit(action.answer)
        elif action.action_type == "bash":
            return self._handle_bash(action.bash_command.command if action.bash_command else "")
        else:
            return InvestigationObservation(
                command_output=f"Unknown action type: {action.action_type}",
                command_history=self.command_history,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty,
                episode_reward=self.episode_reward,
                metadata={"error": "invalid_action_type"},
            )

    def _handle_submit(
        self, answer: Optional[SubmitAnswer]
    ) -> InvestigationObservation:
        """Handle answer submission."""
        if answer is None:
            return InvestigationObservation(
                command_output="No answer provided for submission.",
                command_history=self.command_history,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty,
                episode_reward=self.episode_reward,
                metadata={"error": "no_answer"},
            )

        self.answer_submitted = True
        self.predicted_answer = answer

        # Calculate final reward
        result = self.grader.grade(
            prediction=answer,
            ground_truth=self.ground_truth,
            steps_used=self.step_count,
            total_steps=self.MAX_STEPS,
        )
        self.episode_reward = result.reward

        return InvestigationObservation(
            command_output=self._format_answer_feedback(result),
            command_history=self.command_history,
            steps_remaining=0,
            total_steps=self.MAX_STEPS,
            answer_submitted=True,
            task_difficulty=self.difficulty,
            episode_reward=self.episode_reward,
            metadata={
                "episode_complete": True,
                "scores": {
                    "component": result.component_score,
                    "type": result.type_score,
                    "window": result.window_score,
                    "efficiency": result.efficiency_score,
                },
            },
        )

    def _handle_bash(self, command: str) -> InvestigationObservation:
        """Execute a bash command in the sandbox."""
        # Validate command
        is_valid, error_msg = self._validate_command(command)
        if not is_valid:
            self.command_history.append({
                "command": command,
                "output": f"Command not allowed: {error_msg}",
                "error": True,
            })
            return InvestigationObservation(
                command_output=f"Error: {error_msg}",
                command_history=self.command_history,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty,
                episode_reward=self.episode_reward,
                metadata={"error": error_msg},
            )

        # Execute command
        output, error = self._execute_command(command)

        self.command_history.append({
            "command": command,
            "output": output,
            "error": error,
        })

        return InvestigationObservation(
            command_output=output,
            command_history=self.command_history,
            steps_remaining=self.MAX_STEPS - self.step_count,
            total_steps=self.MAX_STEPS,
            answer_submitted=False,
            task_difficulty=self.difficulty,
            episode_reward=self.episode_reward,
            metadata={
                "last_command": command,
                "output_truncated": len(output) >= self.OUTPUT_TRUNCATION,
            },
        )

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """
        Validate a bash command for security.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command or not command.strip():
            return False, "Empty command"

        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, command):
                return False, f"Forbidden pattern detected: {pattern}"

        # Extract the base command
        parts = command.strip().split()
        if not parts:
            return False, "Empty command"

        base_cmd = parts[0]

        # Check if command is in allowed list
        if base_cmd not in self.ALLOWED_COMMANDS:
            # Allow pipes, but check each command
            if "|" in command:
                pipe_cmds = [p.strip().split()[0] for p in command.split("|")]
                for pc in pipe_cmds:
                    if pc not in self.ALLOWED_COMMANDS:
                        return False, f"Command not allowed: {pc}"

        return True, ""

    def _execute_command(self, command: str) -> Tuple[str, str]:
        """
        Execute a bash command in the sandbox.

        Args:
            command: Command to execute

        Returns:
            Tuple of (stdout, stderr)
        """
        try:
            # Set up environment
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
            env["HOME"] = self.sandbox_dir

            # Execute with timeout
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )

            stdout = result.stdout
            stderr = result.stderr

            # Truncate output if too long
            if len(stdout) > self.OUTPUT_TRUNCATION:
                stdout = stdout[:self.OUTPUT_TRUNCATION] + f"\n... [truncated, total {len(result.stdout)} chars]"

            return stdout, stderr

        except subprocess.TimeoutExpired:
            return "", "Command timed out (10s limit)"
        except Exception as e:
            return "", f"Execution error: {str(e)}"

    def _format_answer_feedback(self, result: EpisodeResult) -> str:
        """Format feedback for submitted answer."""
        lines = [
            "=" * 50,
            "ANSWER SUBMITTED",
            "=" * 50,
            f"Total Score: {result.reward:.4f}",
            "-" * 50,
            "Component Identification: {:.4f}".format(result.component_score),
            "Type Classification: {:.4f}".format(result.type_score),
            "Window Precision: {:.4f}".format(result.window_score),
            "Investigation Efficiency: {:.4f}".format(result.efficiency_score),
            "-" * 50,
            "Ground Truth:",
            f"  Type: {result.ground_truth.get('anomaly_type')}",
            f"  Component: {result.ground_truth.get('component')}",
            f"  Window: {result.ground_truth.get('start_time')} to {result.ground_truth.get('end_time')}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def get_result(self) -> EpisodeResult:
        """Get the episode result."""
        return self.grader.grade(
            prediction=self.predicted_answer,
            ground_truth=self.ground_truth,
            steps_used=self.step_count,
            total_steps=self.MAX_STEPS,
        )


class LogAnomalyEnvironment:
    """
    Main environment class for log anomaly investigation.

    This environment simulates a sandboxed terminal where an agent can
    investigate log files using read-only bash commands to identify
    and characterize anomalies.
    """

    def __init__(self):
        """Initialize the environment."""
        self.episode: Optional[InvestigationEpisode] = None
        self.state = InvestigationState(
            episode_id="",
            step_count=0,
            log_file_path=None,
            ground_truth=None,
            task_id=None,
        )
        self._temp_dir: Optional[str] = None

        # Initialize components
        self.parser = LogParser()
        self.injector = AnomalyInjector()
        self.grader = InvestigationGrader()
        self.task_generator = TaskGenerator(self.grader)

        # Episode storage
        self.episodes: Dict[str, InvestigationEpisode] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "easy",
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvestigationObservation:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Random seed
            difficulty: Difficulty level ("easy", "medium", "hard")
            episode_id: Optional episode ID
            task_id: Optional specific task ID
            **kwargs: Additional options

        Returns:
            Initial observation
        """
        import random
        if seed is not None:
            random.seed(seed)

        # Parse difficulty
        try:
            difficulty_enum = DifficultyLevel(difficulty)
        except ValueError:
            difficulty_enum = DifficultyLevel.EASY

        # Generate or load log content
        log_content, ground_truth = self._generate_episode(
            difficulty=difficulty_enum,
            seed=seed,
            task_id=task_id,
        )

        # Create sandbox directory
        self._temp_dir = tempfile.mkdtemp(prefix="log_anomaly_")
        ep_id = episode_id or str(uuid.uuid4())

        # Add episode_id to ground truth
        ground_truth["episode_id"] = ep_id

        # Create episode
        self.episode = InvestigationEpisode(
            episode_id=ep_id,
            difficulty=difficulty_enum,
            log_content=log_content,
            ground_truth=ground_truth,
            sandbox_dir=self._temp_dir,
        )

        # Update state
        self.state = InvestigationState(
            episode_id=ep_id,
            step_count=0,
            log_file_path=self.episode.log_filepath,
            ground_truth=ground_truth,
            task_id=task_id,
        )

        # Store episode
        self.episodes[ep_id] = self.episode

        # Return initial observation
        return self.episode.reset()

    def _generate_episode(
        self,
        difficulty: DifficultyLevel,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Generate a new episode with injected anomaly.

        Args:
            difficulty: Difficulty level
            seed: Random seed
            task_id: Optional task ID

        Returns:
            Tuple of (log content, ground truth)
        """
        import random
        if seed is not None:
            random.seed(seed)
            self.injector = AnomalyInjector(seed=seed)
        else:
            seed = random.randint(0, 1000000)
            self.injector = AnomalyInjector(seed=seed)

        # Get difficulty settings
        config = self.task_generator.get_task_config(difficulty)

        # Generate base synthetic log
        num_lines = {
            DifficultyLevel.EASY: 500,
            DifficultyLevel.MEDIUM: 1000,
            DifficultyLevel.HARD: 2000,
        }.get(difficulty, 1000)

        logs, metadata = generate_synthetic_log(
            num_lines=num_lines,
            num_components=4,
            seed=seed,
        )

        # Select anomaly type
        anomaly_types = config["allowed_anomaly_types"]
        anomaly_type = random.choice(anomaly_types)

        # Inject anomaly
        modified_logs, ground_truth = self.injector.inject(
            logs=logs,
            anomaly_type=anomaly_type,
            intensity=config["intensity"],
            seed=seed + 1 if seed else None,
        )

        # Add metadata to ground truth
        ground_truth.update({
            "seed": seed,
            "difficulty": difficulty.value,
            "task_id": task_id or f"{difficulty.value}_{seed}",
            "num_lines": num_lines,
        })

        return modified_logs, ground_truth

    def step(
        self,
        action: InvestigationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvestigationObservation:
        """
        Execute an action in the environment.

        Args:
            action: Action to execute
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            Observation after the step
        """
        if self.episode is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Execute step
        observation = self.episode.step(action)

        # Update state
        self.state.step_count = self.episode.step_count

        return observation

    @property
    def state(self) -> InvestigationState:
        """Get current environment state."""
        return self._state

    @state.setter
    def state(self, value: InvestigationState) -> None:
        """Set current state."""
        self._state = value

    def get_result(self, episode_id: Optional[str] = None) -> EpisodeResult:
        """
        Get the result for an episode.

        Args:
            episode_id: Episode ID (uses current if not provided)

        Returns:
            Episode result with scores
        """
        if episode_id:
            episode = self.episodes.get(episode_id)
        else:
            episode = self.episode

        if episode is None:
            raise ValueError("Episode not found")

        return episode.get_result()

    def list_tasks(self) -> Dict[str, Any]:
        """
        List all available tasks.

        Returns:
            Task definitions
        """
        return self.task_generator.list_tasks()

    def grade(self, episode_id: str) -> EpisodeResult:
        """
        Grade an episode.

        Args:
            episode_id: Episode to grade

        Returns:
            Episode result
        """
        return self.get_result(episode_id)

    def close(self) -> None:
        """Clean up resources."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self.episode = None
