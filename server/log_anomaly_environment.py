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
    EnvironmentMode,
    DataSource,
    LogSource,
)
from loghub_parser import LogHubFactory, LogHubSampler


class InvestigationEpisode:
    """Manages a single investigation episode."""

    MAX_STEPS = 15
    OUTPUT_TRUNCATION = 4000  # Max characters in command output

    # Allowed bash commands for security
    ALLOWED_COMMANDS = [
        "grep",
        "egrep",
        "fgrep",  # Pattern matching
        "awk",
        "sed",  # Text processing
        "sort",
        "uniq",
        "wc",  # Counting/aggregation
        "head",
        "tail",
        "cut",  # Selection
        "cat",
        "less",
        "more",  # Display
        "find",
        "xargs",  # File search
        "date",
        "echo",  # Utilities
        "ls",
        "pwd",  # Navigation (read-only)
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
        mode: EnvironmentMode = EnvironmentMode.EVAL,
        data_source: DataSource = DataSource.SYNTHETIC,
        log_source: Optional[LogSource] = None,
    ):
        """
        Initialize an investigation episode.

        Args:
            episode_id: Unique episode identifier
            difficulty: Task difficulty level
            log_content: Log lines for this episode
            ground_truth: Ground truth metadata
            sandbox_dir: Directory for sandboxed execution
            mode: Environment operating mode (training/eval)
            data_source: Source of log data (synthetic/loghub)
            log_source: Specific LogHub source if applicable
        """
        self.episode_id = episode_id
        self.difficulty = difficulty
        self.log_lines = log_content
        self.ground_truth = ground_truth
        self.sandbox_dir = sandbox_dir
        self.log_filepath = os.path.join(sandbox_dir, "log.txt")
        self.mode = mode
        self.data_source = data_source
        self.log_source = log_source

        # Episode state
        self.step_count = 0
        self.command_history: List[Dict[str, Any]] = []
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
            Initial observation with mode-appropriate hints only.
            In EVAL mode: No ground truth hints are provided.
            In TRAINING mode: General difficulty hints only (no specific answers).
        """
        self.step_count = 0
        self.command_history = []
        self.answer_submitted = False
        self.predicted_answer = None
        self.episode_reward = 0.0

        # Build metadata with mode-appropriate hints
        metadata = {
            "log_file": "log.txt",
            "log_lines": len(self.log_lines),
        }

        # In TRAINING mode, provide general difficulty hints
        # In EVAL mode, provide NO hints about the anomaly
        if self.mode == EnvironmentMode.TRAINING:
            metadata["difficulty_hint"] = self._get_severity_hint()
            # No component or type hints - agent must discover these
        # In EVAL mode: metadata contains no hints about the anomaly

        return InvestigationObservation(
            command_output="",
            command_history=[],
            steps_remaining=self.MAX_STEPS,
            total_steps=self.MAX_STEPS,
            answer_submitted=False,
            task_difficulty=self.difficulty,
            episode_reward=0.0,
            mode=self.mode,
            data_source=self.data_source,
            log_source=self.log_source,
            metadata=metadata,
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
                mode=self.mode,
                data_source=self.data_source,
                log_source=self.log_source,
                metadata={"error": "invalid_action_type"},
            )

    def _handle_submit(self, answer: Optional[SubmitAnswer]) -> InvestigationObservation:
        """
        Handle answer submission.

        Feedback is mode-dependent:
        - TRAINING: Full feedback with component scores, type accuracy, window IoU, efficiency, AND ground truth.
        - EVAL: Scores only (no ground truth revealed to prevent learning during evaluation).
        """
        if answer is None:
            return InvestigationObservation(
                command_output="No answer provided for submission.",
                command_history=self.command_history,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty,
                episode_reward=self.episode_reward,
                mode=self.mode,
                data_source=self.data_source,
                log_source=self.log_source,
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

        # Format feedback based on mode
        if self.mode == EnvironmentMode.TRAINING:
            # Full feedback including ground truth for training
            feedback = self._format_answer_feedback(result, show_ground_truth=True)
            metadata = {
                "episode_complete": True,
                "scores": {
                    "component": result.component_score,
                    "type": result.type_score,
                    "window": result.window_score,
                    "efficiency": result.efficiency_score,
                },
                "ground_truth": {
                    "anomaly_type": result.ground_truth.get("anomaly_type"),
                    "component": result.ground_truth.get("component"),
                    "start_time": result.ground_truth.get("start_time"),
                    "end_time": result.ground_truth.get("end_time"),
                },
            }
        else:
            # EVAL mode: scores only, no ground truth
            feedback = self._format_answer_feedback(result, show_ground_truth=False)
            metadata = {
                "episode_complete": True,
                "scores": {
                    "component": result.component_score,
                    "type": result.type_score,
                    "window": result.window_score,
                    "efficiency": result.efficiency_score,
                },
                "note": "Ground truth withheld in EVAL mode",
            }

        return InvestigationObservation(
            command_output=feedback,
            command_history=self.command_history,
            steps_remaining=0,
            total_steps=self.MAX_STEPS,
            answer_submitted=True,
            task_difficulty=self.difficulty,
            episode_reward=self.episode_reward,
            mode=self.mode,
            data_source=self.data_source,
            log_source=self.log_source,
            metadata=metadata,
        )

    def _handle_bash(self, command: str) -> InvestigationObservation:
        """Execute a bash command in the sandbox."""
        # Validate command
        is_valid, error_msg = self._validate_command(command)
        if not is_valid:
            self.command_history.append(
                {
                    "command": command,
                    "output": f"Command not allowed: {error_msg}",
                    "error": error_msg,  # Store error message as string
                }
            )
            return InvestigationObservation(
                command_output=f"Error: {error_msg}",
                command_history=self.command_history,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty,
                episode_reward=self.episode_reward,
                mode=self.mode,
                data_source=self.data_source,
                log_source=self.log_source,
                metadata={"error": error_msg},
            )

        # Execute command
        output, error = self._execute_command(command)

        self.command_history.append(
            {
                "command": command,
                "output": output,
                "error": error,
            }
        )

        # Cap internal storage to last 5 entries to prevent memory bloat
        if len(self.command_history) > 5:
            self.command_history = self.command_history[-5:]

        # Compute intermediate reward signal
        intermediate_reward = self._compute_intermediate_reward(output, error)
        self.episode_reward += intermediate_reward

        return InvestigationObservation(
            command_output=output,
            command_history=self.command_history[-3:],  # Return only last 3 in observations
            steps_remaining=self.MAX_STEPS - self.step_count,
            total_steps=self.MAX_STEPS,
            answer_submitted=False,
            task_difficulty=self.difficulty,
            episode_reward=self.episode_reward,
            mode=self.mode,
            data_source=self.data_source,
            log_source=self.log_source,
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

        # Check if base command is allowed
        if base_cmd not in self.ALLOWED_COMMANDS:
            return False, f"Command not allowed: {base_cmd}"

        # If command contains pipes, validate each piped command
        if "|" in command:
            pipe_cmds = [p.strip().split()[0] for p in command.split("|") if p.strip()]
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
                stdout = (
                    stdout[: self.OUTPUT_TRUNCATION]
                    + f"\n... [truncated, total {len(result.stdout)} chars]"
                )

            return stdout, stderr

        except subprocess.TimeoutExpired:
            return "", "Command timed out (10s limit)"
        except Exception as e:
            return "", f"Execution error: {str(e)}"

    def _compute_intermediate_reward(self, stdout: str, stderr: str) -> float:
        """
        Compute small intermediate reward based on command output.

        Rewards agents for finding relevant signals without revealing ground truth.

        Args:
            stdout: Command stdout
            stderr: Command stderr

        Returns:
            Small reward delta (typically -0.05 to +0.05)
        """
        if not stdout or stderr:
            return 0.0

        reward = 0.0
        gt = self.ground_truth

        # Small positive if output contains the affected component
        if gt.get("component") and gt["component"].lower() in stdout.lower():
            reward += 0.02

        # Small positive if output contains timestamps in the anomaly window
        try:
            gt_start = self.grader._parse_timestamp(str(gt.get("start_time", "")))
            gt_end = self.grader._parse_timestamp(str(gt.get("end_time", "")))

            # Check if any timestamps in output fall within window
            import re

            timestamp_pattern = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
            for match in re.finditer(timestamp_pattern, stdout):
                try:
                    ts = self.grader._parse_timestamp(match.group(0))
                    if gt_start <= ts <= gt_end:
                        reward += 0.02
                        break  # Only reward once per command
                except:
                    pass
        except:
            pass

        # Small negative for repeating the same command pattern
        if len(self.command_history) >= 3:
            last_three = [h["command"] for h in self.command_history[-3:]]
            # Check if all three are essentially the same (ignoring minor variations)
            if (
                len(set(cmd.split()[0] for cmd in last_three if cmd.strip())) == 1
            ):  # Same base command
                reward -= 0.01

        return max(-0.05, min(0.05, reward))  # Cap between -0.05 and +0.05

    def _format_answer_feedback(self, result: EpisodeResult, show_ground_truth: bool = True) -> str:
        """
        Format feedback for submitted answer.

        Args:
            result: Episode result from grading
            show_ground_truth: Whether to include ground truth in feedback

        Returns:
            Formatted feedback string
        """
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
        ]

        if show_ground_truth:
            lines.extend(
                [
                    "-" * 50,
                    "Ground Truth:",
                    f"  Type: {result.ground_truth.get('anomaly_type')}",
                    f"  Component: {result.ground_truth.get('component')}",
                    f"  Window: {result.ground_truth.get('start_time')} to {result.ground_truth.get('end_time')}",
                ]
            )

        lines.append("=" * 50)
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

    Supports two operating modes:
    - TRAINING: Full feedback including ground truth for RL training
    - EVAL: Limited feedback for fair agent evaluation

    Supports two data sources:
    - SYNTHETIC: Generated logs with injected anomalies
    - LOGHUB: Real logs from LogHub datasets
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
            mode=EnvironmentMode.EVAL,
            data_source=DataSource.SYNTHETIC,
            log_source=None,
        )
        self._temp_dir: Optional[str] = None

        # Initialize components
        self.parser = LogParser()
        self.injector = AnomalyInjector()
        self.grader = InvestigationGrader()
        self.task_generator = TaskGenerator(self.grader)
        self.loghub_sampler = LogHubSampler()

        # LogHub data cache (lazy loaded)
        self._loghub_data: Dict[LogSource, Tuple[List[LogLine], Any]] = {}

        # Episode storage with size limit
        self.episodes: Dict[str, InvestigationEpisode] = {}
        self.MAX_EPISODES = 100  # Prevent unbounded growth

    def reset(
        self,
        seed: Optional[int] = None,
        difficulty: str = "easy",
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        mode: str = "eval",
        data_source: str = "synthetic",
        log_source: Optional[str] = None,
        **kwargs: Any,
    ) -> InvestigationObservation:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Random seed
            difficulty: Difficulty level ("easy", "medium", "hard")
            episode_id: Optional episode ID
            task_id: Optional specific task ID
            mode: Operating mode ("training" or "eval")
            data_source: Data source ("synthetic" or "loghub")
            log_source: Specific LogHub source (HDFS, BGL, OpenStack, Apache)
            **kwargs: Additional options

        Returns:
            Initial observation with mode-appropriate information
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Parse difficulty
        try:
            difficulty_enum = DifficultyLevel(difficulty)
        except ValueError:
            difficulty_enum = DifficultyLevel.EASY

        # Parse mode
        try:
            mode_enum = EnvironmentMode(mode.lower())
        except ValueError:
            mode_enum = EnvironmentMode.EVAL

        # Parse data source
        try:
            data_source_enum = DataSource(data_source.lower())
        except ValueError:
            data_source_enum = DataSource.SYNTHETIC

        # Parse log source (for LogHub data)
        log_source_enum = None
        if log_source:
            try:
                log_source_enum = LogSource(log_source.upper())
            except ValueError:
                pass

        # Generate or load log content
        log_content, ground_truth = self._generate_episode(
            difficulty=difficulty_enum,
            seed=seed,
            task_id=task_id,
            data_source=data_source_enum,
            log_source=log_source_enum,
        )

        # Create sandbox directory
        self._temp_dir = tempfile.mkdtemp(prefix="log_anomaly_")
        ep_id = episode_id or str(uuid.uuid4())

        # Add episode_id to ground truth
        ground_truth["episode_id"] = ep_id

        # Create episode with mode and data source
        self.episode = InvestigationEpisode(
            episode_id=ep_id,
            difficulty=difficulty_enum,
            log_content=log_content,
            ground_truth=ground_truth,
            sandbox_dir=self._temp_dir,
            mode=mode_enum,
            data_source=data_source_enum,
            log_source=log_source_enum,
        )

        # Update state with mode and data source
        self.state = InvestigationState(
            episode_id=ep_id,
            step_count=0,
            log_file_path=self.episode.log_filepath,
            ground_truth=ground_truth,
            task_id=task_id,
            mode=mode_enum,
            data_source=data_source_enum,
            log_source=log_source_enum,
        )

        # Store episode with size limiting
        self.episodes[ep_id] = self.episode

        # Evict oldest episode if we exceed limit
        if len(self.episodes) > self.MAX_EPISODES:
            oldest_id = min(self.episodes.keys())  # Simple FIFO
            self.episodes.pop(oldest_id, None)

        # Return initial observation
        return self.episode.reset()

    def _generate_episode(
        self,
        difficulty: DifficultyLevel,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        data_source: DataSource = DataSource.SYNTHETIC,
        log_source: Optional[LogSource] = None,
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """
        Generate a new episode with injected anomaly.

        Args:
            difficulty: Difficulty level
            seed: Random seed
            task_id: Optional task ID
            data_source: Source of log data (synthetic/loghub)
            log_source: Specific LogHub source if applicable

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

        # Try LogHub data if requested
        if data_source == DataSource.LOGHUB and log_source:
            logs, metadata = self._load_loghub_data(log_source, seed, num_lines)
            if logs:
                # Inject anomaly into real logs
                anomaly_types = config["allowed_anomaly_types"]
                anomaly_type = random.choice(anomaly_types)
                modified_logs, ground_truth = self.injector.inject(
                    logs=logs,
                    anomaly_type=anomaly_type,
                    intensity=config["intensity"],
                    seed=seed + 1 if seed else None,
                )
                ground_truth.update(
                    {
                        "seed": seed,
                        "difficulty": difficulty.value,
                        "task_id": task_id or f"{difficulty.value}_{seed}",
                        "data_source": "loghub",
                        "log_source": log_source.value if log_source else None,
                    }
                )
                return modified_logs, ground_truth

        # Default: generate synthetic logs
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
        ground_truth.update(
            {
                "seed": seed,
                "difficulty": difficulty.value,
                "task_id": task_id or f"{difficulty.value}_{seed}",
                "num_lines": num_lines,
                "data_source": "synthetic",
            }
        )

        return modified_logs, ground_truth

    def _load_loghub_data(
        self, log_source: LogSource, seed: Optional[int], max_lines: int
    ) -> Tuple[Optional[List[LogLine]], Optional[Any]]:
        """
        Load LogHub data for the specified source.

        Args:
            log_source: LogHub source to load
            seed: Random seed
            max_lines: Maximum lines to load

        Returns:
            Tuple of (logs, metadata) or (None, None) if not available
        """
        # Check for sample log file
        sample_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            f"{log_source.value.lower()}_sample.log",
        )

        if os.path.exists(sample_file):
            try:
                parser = LogHubFactory.get_parser(log_source.value, seed=seed)
                logs, metadata = parser.parse_file(sample_file, max_lines=max_lines)
                return logs, metadata
            except Exception:
                pass

        return None, None

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
