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

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import OpenEnv base classes
from openenv.core.env_server.interfaces import Environment

# Import central config
from config import (
    MAX_STEPS,
    OUTPUT_TRUNCATION,
    COMMAND_TIMEOUT,
    ALLOWED_COMMANDS,
    MAX_COMMAND_HISTORY,
    REWARD_TIMEOUT,
    REWARD_REPEAT_COMMAND_PENALTY,
    REWARD_REPEAT_WARNING_PENALTY,
    get_difficulty_config,
    get_logger,
    parse_timestamp,
)

# Set up logging for this module
logger = get_logger(__name__)

from log_utils import (
    LogParser,
    AnomalyInjector,
    generate_synthetic_log,
)
from grader import InvestigationGrader, TaskGenerator
from models import (
    LogAction,
    LogObservation,
    LogState,
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

    # Use central config values (exposed as class attributes for backwards compat)
    MAX_STEPS = MAX_STEPS
    OUTPUT_TRUNCATION = OUTPUT_TRUNCATION
    ALLOWED_COMMANDS = ALLOWED_COMMANDS

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

    def reset(self) -> LogObservation:
        """
        Reset the episode state.

        Returns:
            Initial observation with mode-appropriate hints only.
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
            "mode": self.mode.value,
            "data_source": self.data_source.value,
            "log_source": self.log_source.value if self.log_source else None,
        }

        # In TRAINING mode, provide general difficulty hints
        if self.mode == EnvironmentMode.TRAINING:
            metadata["difficulty_hint"] = self._get_severity_hint()

        return LogObservation(
            command_output="Environment ready. Use bash commands to investigate log.txt",
            stderr="",
            exit_code=0,
            steps_remaining=self.MAX_STEPS,
            total_steps=self.MAX_STEPS,
            answer_submitted=False,
            task_difficulty=self.difficulty.value,
            done=False,
            reward=0.0,
            metadata=metadata,
        )

    def _get_severity_hint(self) -> str:
        """Get a hint about the anomaly severity."""
        intensity = self.ground_truth.get("intensity", 0.5)
        if intensity >= 0.7:
            return "obvious"
        elif intensity >= 0.4:
            return "moderate"
        else:
            return "subtle"

    def step(self, action: LogAction) -> LogObservation:
        """
        Execute an action in the episode.

        Args:
            action: The action to execute (LogAction format)

        Returns:
            Observation after the step
        """
        if self.answer_submitted:
            return LogObservation(
                command_output="Episode already complete. Please reset.",
                stderr="",
                exit_code=1,
                steps_remaining=0,
                total_steps=self.MAX_STEPS,
                answer_submitted=True,
                task_difficulty=self.difficulty.value,
                done=True,
                reward=self.episode_reward,
                metadata={"status": "episode_complete"},
            )

        self.step_count += 1

        # Check for timeout (max steps reached without submission)
        if self.step_count >= self.MAX_STEPS and action.action_type != "submit":
            # Wipe accumulated rewards - clear negative signal for GRPO
            wiped_reward = self.episode_reward
            self.episode_reward = REWARD_TIMEOUT  # Strong negative final reward (-2.0)
            self.answer_submitted = True  # Mark as done

            return LogObservation(
                command_output=f"TIMEOUT: Investigation incomplete. No answer submitted. You get {REWARD_TIMEOUT} reward.",
                stderr="Episode ended without submission",
                exit_code=1,
                steps_remaining=0,
                total_steps=self.MAX_STEPS,
                answer_submitted=True,
                task_difficulty=self.difficulty.value,
                done=True,
                reward=self.episode_reward,
                metadata={
                    "timeout": True,
                    "wiped_reward": wiped_reward,
                    "steps_used": self.step_count,
                },
            )

        if action.action_type == "submit":
            return self._handle_submit(action)
        elif action.action_type == "bash":
            return self._handle_bash(action.command or "")
        else:
            return LogObservation(
                command_output=f"Unknown action type: {action.action_type}",
                stderr="Use action_type='bash' or 'submit'",
                exit_code=1,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty.value,
                done=False,
                reward=self.episode_reward,
                metadata={"error": "invalid_action_type"},
            )

    def _handle_submit(self, action: LogAction) -> LogObservation:
        """Handle answer submission."""
        if not action.anomaly_type or not action.component:
            return LogObservation(
                command_output="Submit action requires: anomaly_type, component, start_time, end_time",
                stderr="Missing required fields",
                exit_code=1,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty.value,
                done=False,
                reward=self.episode_reward,
                metadata={"error": "missing_fields"},
            )

        self.answer_submitted = True

        # Create SubmitAnswer for grading
        try:
            answer = SubmitAnswer(
                anomaly_type=AnomalyType(action.anomaly_type),
                component=action.component,
                start_time=action.start_time or "",
                end_time=action.end_time or "",
                confidence=action.confidence,
            )
            self.predicted_answer = answer
        except ValueError:
            # Invalid anomaly type
            return LogObservation(
                command_output=f"Invalid anomaly_type: {action.anomaly_type}",
                stderr=f"Valid types: {[t.value for t in AnomalyType]}",
                exit_code=1,
                steps_remaining=0,
                total_steps=self.MAX_STEPS,
                answer_submitted=True,
                task_difficulty=self.difficulty.value,
                done=True,
                reward=0.0,
                metadata={"error": "invalid_anomaly_type"},
            )

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

        return LogObservation(
            command_output=feedback,
            stderr="",
            exit_code=0,
            steps_remaining=0,
            total_steps=self.MAX_STEPS,
            answer_submitted=True,
            task_difficulty=self.difficulty.value,
            done=True,
            reward=self.episode_reward,
            metadata=metadata,
        )

    def _handle_bash(self, command: str) -> LogObservation:
        """Execute a bash command in the sandbox."""
        # Check for repeated commands (circuit breaker)
        repeat_count = sum(1 for h in self.command_history if h["command"] == command)

        if repeat_count >= 2:
            # Block the command after 2 repeats (3rd attempt blocked)
            self.episode_reward += REWARD_REPEAT_COMMAND_PENALTY  # Strong penalty (-0.3)
            return LogObservation(
                command_output=f"BLOCKED: You've run this exact command {repeat_count + 1} times. Try a different approach.",
                stderr="Repeated command blocked",
                exit_code=1,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty.value,
                done=False,
                reward=self.episode_reward,
                metadata={
                    "error": "repeat_blocked",
                    "repeat_count": repeat_count + 1,
                    "command": command,
                    "command_history": self.command_history,  # Keep history for DO NOT REPEAT
                },
            )

        # Escalating penalty for first repeat
        if repeat_count == 1:
            self.episode_reward += REWARD_REPEAT_WARNING_PENALTY  # Warning penalty (-0.1)

        # Validate command
        is_valid, error_msg = self._validate_command(command)
        if not is_valid:
            self.command_history.append(
                {
                    "command": command,
                    "output": f"Command not allowed: {error_msg}",
                    "error": error_msg,
                }
            )
            return LogObservation(
                command_output=f"Error: {error_msg}",
                stderr=error_msg,
                exit_code=1,
                steps_remaining=self.MAX_STEPS - self.step_count,
                total_steps=self.MAX_STEPS,
                answer_submitted=False,
                task_difficulty=self.difficulty.value,
                done=False,
                reward=self.episode_reward,
                metadata={
                    "error": error_msg,
                    "command": command,
                    "command_history": self.command_history,  # Keep history for DO NOT REPEAT
                },
            )

        # Execute command
        stdout, stderr = self._execute_command(command)

        self.command_history.append(
            {
                "command": command,
                "output": stdout,
                "error": stderr,
            }
        )

        # Cap history to prevent memory bloat
        if len(self.command_history) > MAX_COMMAND_HISTORY:
            self.command_history = self.command_history[-MAX_COMMAND_HISTORY:]

        # Compute intermediate reward
        intermediate_reward = self._compute_intermediate_reward(stdout, stderr)
        self.episode_reward += intermediate_reward

        return LogObservation(
            command_output=stdout,
            stderr=stderr,
            exit_code=0 if not stderr else 1,
            steps_remaining=self.MAX_STEPS - self.step_count,
            total_steps=self.MAX_STEPS,
            answer_submitted=False,
            task_difficulty=self.difficulty.value,
            done=False,
            reward=self.episode_reward,
            metadata={
                "command": command,
                "output_truncated": len(stdout) >= self.OUTPUT_TRUNCATION,
                "intermediate_reward": intermediate_reward,
                "command_history": self.command_history,  # Full history for prompt building
            },
        )

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate a bash command for security."""
        if not command or not command.strip():
            return False, "Empty command"

        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, command):
                return False, f"Forbidden pattern detected"

        # Extract the base command
        parts = command.strip().split()
        if not parts:
            return False, "Empty command"

        base_cmd = parts[0]

        # Check if base command is allowed
        if base_cmd not in self.ALLOWED_COMMANDS:
            return (
                False,
                f"Command not allowed: {base_cmd}. Use: {', '.join(self.ALLOWED_COMMANDS[:5])}...",
            )

        # If command contains pipes, validate each piped command
        # But first, we need to handle pipes inside quoted strings (e.g., grep "ERROR|WARN")
        if "|" in command:
            # Use shlex to properly parse, respecting quotes
            # We'll manually find unquoted pipes by tracking quote state
            pipe_positions = []
            in_single_quote = False
            in_double_quote = False
            for i, char in enumerate(command):
                if char == "'" and not in_double_quote:
                    in_single_quote = not in_single_quote
                elif char == '"' and not in_single_quote:
                    in_double_quote = not in_double_quote
                elif char == "|" and not in_single_quote and not in_double_quote:
                    pipe_positions.append(i)

            # Only validate if there are actual (unquoted) pipes
            if pipe_positions:
                # Split command at unquoted pipe positions
                parts = []
                prev = 0
                for pos in pipe_positions:
                    parts.append(command[prev:pos].strip())
                    prev = pos + 1
                parts.append(command[prev:].strip())

                # Validate each piped command
                for part in parts:
                    if part:
                        part_cmd = part.split()[0] if part.split() else ""
                        if part_cmd and part_cmd not in self.ALLOWED_COMMANDS:
                            return False, f"Piped command not allowed: {part_cmd}"

        return True, ""

    def _execute_command(self, command: str) -> Tuple[str, str]:
        """Execute a bash command in the sandbox."""
        try:
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
            env["HOME"] = self.sandbox_dir

            result = subprocess.run(
                command,
                shell=True,
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
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
        """Compute small intermediate reward based on command output."""
        if not stdout or stderr:
            return 0.0

        reward = 0.0
        gt = self.ground_truth

        # Small positive if output contains the affected component
        if gt.get("component") and gt["component"].lower() in stdout.lower():
            reward += 0.02

        # Check for timestamps in anomaly window
        try:
            gt_start = parse_timestamp(str(gt.get("start_time", "")))
            gt_end = parse_timestamp(str(gt.get("end_time", "")))

            if gt_start is None or gt_end is None:
                raise ValueError("Missing ground truth timestamps")

            timestamp_pattern = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
            for match in re.finditer(timestamp_pattern, stdout):
                ts = parse_timestamp(match.group(0))
                if ts and gt_start <= ts <= gt_end:
                    reward += 0.02
                    break
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed to calculate step reward: %s", e)

        # Penalize repetitive commands
        if len(self.command_history) >= 3:
            last_three = [h["command"] for h in self.command_history[-3:]]
            if len(set(cmd.split()[0] for cmd in last_three if cmd.strip())) == 1:
                reward -= 0.01

        return max(-0.05, min(0.05, reward))

    def _format_answer_feedback(self, result: EpisodeResult, show_ground_truth: bool = True) -> str:
        """Format feedback for submitted answer."""
        lines = [
            "=" * 50,
            "ANSWER SUBMITTED",
            "=" * 50,
            f"Total Score: {result.reward:.4f}",
            "-" * 50,
            f"Component Identification: {result.component_score:.4f}",
            f"Type Classification: {result.type_score:.4f}",
            f"Window Precision: {result.window_score:.4f}",
            f"Investigation Efficiency: {result.efficiency_score:.4f}",
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


class LogAnomalyEnvironment(Environment):
    """
    Main environment class for log anomaly investigation.

    This environment inherits from OpenEnv Environment base class and
    implements the standard step()/reset()/state interface.

    Supports two operating modes:
    - TRAINING: Full feedback including ground truth for RL training
    - EVAL: Limited feedback for fair agent evaluation
    """

    def __init__(self):
        """Initialize the environment."""
        self.episode: Optional[InvestigationEpisode] = None
        self._state = LogState(
            episode_id="",
            step_count=0,
            log_file_path=None,
            task_id=None,
            mode="eval",
            data_source="synthetic",
            log_source=None,
        )
        self._temp_dir: Optional[str] = None

        # Initialize components
        self.parser = LogParser()
        self.injector = AnomalyInjector()
        self.grader = InvestigationGrader()
        self.task_generator = TaskGenerator(self.grader)
        self.loghub_sampler = LogHubSampler()

        # Episode storage with size limit
        self.episodes: Dict[str, InvestigationEpisode] = {}
        self.MAX_EPISODES = 100

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LogObservation:
        """
        Reset the environment and start a new episode.

        Args:
            seed: Random seed
            episode_id: Optional episode ID
            **kwargs: Additional options (difficulty, mode, data_source, etc.)

        Returns:
            Initial observation
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Parse options from kwargs
        difficulty = kwargs.get("difficulty", "easy")
        task_id = kwargs.get("task_id")
        mode = kwargs.get("mode", "eval")
        data_source = kwargs.get("data_source", "synthetic")
        log_source = kwargs.get("log_source")

        # Parse enums
        try:
            difficulty_enum = DifficultyLevel(difficulty)
        except ValueError:
            difficulty_enum = DifficultyLevel.EASY

        try:
            mode_enum = EnvironmentMode(mode.lower())
        except ValueError:
            mode_enum = EnvironmentMode.EVAL

        try:
            data_source_enum = DataSource(data_source.lower())
        except ValueError:
            data_source_enum = DataSource.SYNTHETIC

        log_source_enum = None
        if log_source:
            try:
                log_source_enum = LogSource(log_source.upper())
            except ValueError:
                logger.debug("Invalid log source '%s', ignoring", log_source)

        # Generate episode
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
        ground_truth["episode_id"] = ep_id

        # Create episode
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

        # Update state
        self._state = LogState(
            episode_id=ep_id,
            step_count=0,
            log_file_path=self.episode.log_filepath,
            task_id=task_id,
            mode=mode_enum.value,
            data_source=data_source_enum.value,
            log_source=log_source_enum.value if log_source_enum else None,
        )

        # Store episode
        self.episodes[ep_id] = self.episode
        if len(self.episodes) > self.MAX_EPISODES:
            oldest_id = min(self.episodes.keys())
            self.episodes.pop(oldest_id, None)

        return self.episode.reset()

    def _generate_episode(
        self,
        difficulty: DifficultyLevel,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        data_source: DataSource = DataSource.SYNTHETIC,
        log_source: Optional[LogSource] = None,
    ) -> Tuple[List[LogLine], Dict[str, Any]]:
        """Generate a new episode with injected anomaly and optional decoys."""
        import random

        if seed is not None:
            random.seed(seed)
            self.injector = AnomalyInjector(seed=seed)
        else:
            seed = random.randint(0, 1000000)
            self.injector = AnomalyInjector(seed=seed)

        config = self.task_generator.get_task_config(difficulty)
        difficulty_config = get_difficulty_config(difficulty)
        num_lines = difficulty_config.num_lines
        num_decoys = difficulty_config.num_decoys  # Get decoy count from config

        # Try LogHub data if requested
        if data_source == DataSource.LOGHUB and log_source:
            logs, metadata = self._load_loghub_data(log_source, seed, num_lines)
            if logs:
                anomaly_types = config["allowed_anomaly_types"]
                anomaly_type = random.choice(anomaly_types)

                # Use inject_with_decoys for hidden state challenge
                modified_logs, ground_truth = self.injector.inject_with_decoys(
                    logs=logs,
                    primary_anomaly=anomaly_type,
                    num_decoys=num_decoys,
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

        anomaly_types = config["allowed_anomaly_types"]
        anomaly_type = random.choice(anomaly_types)

        # Use inject_with_decoys for hidden state challenge
        modified_logs, ground_truth = self.injector.inject_with_decoys(
            logs=logs,
            primary_anomaly=anomaly_type,
            num_decoys=num_decoys,
            intensity=config["intensity"],
            seed=seed + 1 if seed else None,
        )

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
        """Load LogHub data for the specified source."""
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
            except Exception as e:
                logger.debug("Failed to parse LogHub file %s: %s", sample_file, e)

        return None, None

    def step(
        self,
        action: LogAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LogObservation:
        """
        Execute an action in the environment.

        Args:
            action: Action to execute (LogAction)
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            Observation after the step
        """
        if self.episode is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Handle legacy InvestigationAction format
        if isinstance(action, InvestigationAction):
            action = action.to_log_action()

        # Execute step
        observation = self.episode.step(action)

        # Update state
        self._state.step_count = self.episode.step_count
        self._state.last_exit_code = observation.exit_code

        return observation

    @property
    def state(self) -> LogState:
        """Get current environment state."""
        return self._state

    def get_result(self, episode_id: Optional[str] = None) -> EpisodeResult:
        """Get the result for an episode."""
        if episode_id:
            episode = self.episodes.get(episode_id)
        else:
            episode = self.episode

        if episode is None:
            raise ValueError("Episode not found")

        return episode.get_result()

    def list_tasks(self) -> Dict[str, Any]:
        """List all available tasks."""
        return self.task_generator.list_tasks()

    def grade(self, episode_id: str) -> EpisodeResult:
        """Grade an episode."""
        return self.get_result(episode_id)

    def close(self) -> None:
        """Clean up resources."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self.episode = None
