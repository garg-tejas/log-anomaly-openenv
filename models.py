"""
Log Anomaly Investigation Environment - Data Models.

This module defines the typed models for actions, observations, and state
used in the Log Anomaly Investigation environment.

These models inherit from the OpenEnv base classes for spec compliance.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field, BaseModel
from enum import Enum

# Import OpenEnv base classes
from openenv.core.env_server.interfaces import Action as _BaseAction
from openenv.core.env_server.interfaces import Observation as _BaseObservation
from openenv.core.env_server.interfaces import State as _BaseState


class AnomalyType(str, Enum):
    """Types of anomalies that can be injected into logs."""

    ERROR_SPIKE = "error_spike"
    MEMORY_LEAK = "memory_leak"
    SERVICE_DROPOUT = "service_dropout"
    CASCADE_FAILURE = "cascade_failure"
    LATENCY_DEGRADATION = "latency_degradation"
    AUTH_ANOMALY = "auth_anomaly"


class DifficultyLevel(str, Enum):
    """Difficulty levels for investigation tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EnvironmentMode(str, Enum):
    """
    Operating mode for the environment.

    - TRAINING: Full feedback including ground truth after submission.
      Used for RL training where agents need complete information.
    - EVAL: Limited feedback without ground truth.
      Used for fair evaluation of agent performance.
    """

    TRAINING = "training"
    EVAL = "eval"


class DataSource(str, Enum):
    """
    Source of log data for episodes.

    - SYNTHETIC: Generated synthetic logs with injected anomalies.
    - LOGHUB: Real logs from LogHub datasets (HDFS, BGL, OpenStack, Apache).
    """

    SYNTHETIC = "synthetic"
    LOGHUB = "loghub"


class LogSource(str, Enum):
    """Specific log sources from LogHub."""

    HDFS = "HDFS"
    BGL = "BGL"
    OPENSTACK = "OpenStack"
    APACHE = "Apache"


# =============================================================================
# Action Types (inherit from OpenEnv Action)
# =============================================================================


class LogAction(_BaseAction):
    """
    Action for the Log Anomaly Investigation environment.

    Supports two action types:
    - "bash": Execute a bash command to investigate logs
    - "submit": Submit final answer with anomaly identification
    """

    action_type: str = Field(..., description="Type of action: 'bash' or 'submit'")

    # For bash actions
    command: Optional[str] = Field(default=None, description="Bash command to execute")

    # For submit actions
    anomaly_type: Optional[str] = Field(default=None, description="Type of anomaly identified")
    component: Optional[str] = Field(default=None, description="Component affected by anomaly")
    start_time: Optional[str] = Field(
        default=None, description="Start time of anomaly window (ISO format)"
    )
    end_time: Optional[str] = Field(
        default=None, description="End time of anomaly window (ISO format)"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in answer")


# =============================================================================
# Observation Types (inherit from OpenEnv Observation)
# =============================================================================


class LogObservation(_BaseObservation):
    """
    Observation returned after each step in the Log Anomaly environment.

    Inherits from OpenEnv Observation which provides:
    - done: bool - Whether the episode has terminated
    - reward: Optional[float] - Reward signal from the last action
    - metadata: Dict[str, Any] - Additional metadata
    """

    # Command execution results
    command_output: str = Field(default="", description="Output from the bash command")
    stderr: str = Field(default="", description="Standard error output")
    exit_code: int = Field(default=0, description="Command exit code")

    # Episode context (returned in metadata for cleaner interface)
    # These are commonly accessed fields that we expose directly
    steps_remaining: int = Field(default=15, description="Number of steps remaining")
    total_steps: int = Field(default=15, description="Total steps allowed")
    answer_submitted: bool = Field(default=False, description="Whether answer was submitted")
    task_difficulty: str = Field(default="easy", description="Current task difficulty")

    # Explicitly define inherited fields for type checker compatibility
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Optional[float] = Field(default=None, description="Reward signal from the last action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# State Types (inherit from OpenEnv State)
# =============================================================================


class LogState(_BaseState):
    """
    Current state of the investigation environment.

    Inherits from OpenEnv State which provides:
    - episode_id: Optional[str] - Unique identifier for the current episode
    - step_count: int - Number of steps taken in the current episode
    """

    log_file_path: Optional[str] = Field(default=None, description="Path to log file")
    task_id: Optional[str] = Field(default=None, description="Current task identifier")
    mode: str = Field(default="eval", description="Environment operating mode")
    data_source: str = Field(default="synthetic", description="Source of log data")
    log_source: Optional[str] = Field(
        default=None, description="Specific LogHub source if applicable"
    )
    last_exit_code: int = Field(default=0, description="Exit code from last command")


# =============================================================================
# Legacy Models (for backward compatibility with existing code)
# =============================================================================


class BashCommand(BaseModel):
    """A bash command to execute in the investigation environment."""

    command: str = Field(..., description="The bash command to execute")


class SubmitAnswer(BaseModel):
    """Submit the final answer for the anomaly investigation."""

    anomaly_type: AnomalyType = Field(..., description="Type of anomaly identified")
    component: str = Field(..., description="Component affected by the anomaly")
    start_time: str = Field(..., description="Start time of anomaly window (ISO format)")
    end_time: str = Field(..., description="End time of anomaly window (ISO format)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in answer")


class InvestigationAction(BaseModel):
    """Actions available to the agent during investigation (legacy format)."""

    action_type: str = Field(..., description="Type of action: 'bash' or 'submit'")
    bash_command: Optional[BashCommand] = Field(default=None, description="Bash command to execute")
    answer: Optional[SubmitAnswer] = Field(default=None, description="Final answer to submit")

    def to_log_action(self) -> LogAction:
        """Convert to OpenEnv-compliant LogAction."""
        if self.action_type == "bash" and self.bash_command:
            return LogAction(
                action_type="bash",
                command=self.bash_command.command,
            )
        elif self.action_type == "submit" and self.answer:
            return LogAction(
                action_type="submit",
                anomaly_type=self.answer.anomaly_type.value,
                component=self.answer.component,
                start_time=self.answer.start_time,
                end_time=self.answer.end_time,
                confidence=self.answer.confidence,
            )
        else:
            return LogAction(action_type=self.action_type)


class LogLine(BaseModel):
    """A single normalized log line."""

    timestamp: str = Field(..., description="Normalized timestamp (ISO format)")
    severity: str = Field(..., description="Log severity level")
    component: str = Field(..., description="Source component")
    message: str = Field(..., description="Raw log message")
    raw_line: str = Field(..., description="Original log line")


class InvestigationObservation(BaseModel):
    """Observation returned after each step (legacy format for baseline compatibility)."""

    command_output: str = Field(default="", description="Output from the bash command")
    command_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of all commands and outputs"
    )
    steps_remaining: int = Field(..., description="Number of steps remaining")
    total_steps: int = Field(default=15, description="Total steps allowed")
    answer_submitted: bool = Field(default=False, description="Whether answer was submitted")
    task_difficulty: DifficultyLevel = Field(..., description="Current task difficulty")
    episode_reward: float = Field(default=0.0, description="Cumulative reward so far")
    mode: EnvironmentMode = Field(
        default=EnvironmentMode.EVAL,
        description="Environment mode (training=full feedback, eval=limited feedback)",
    )
    data_source: DataSource = Field(
        default=DataSource.SYNTHETIC, description="Source of log data (synthetic or loghub)"
    )
    log_source: Optional[LogSource] = Field(
        default=None, description="Specific log source if using LogHub data"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_log_observation(self) -> LogObservation:
        """Convert to OpenEnv-compliant LogObservation."""
        return LogObservation(
            command_output=self.command_output,
            stderr="",
            exit_code=0,
            steps_remaining=self.steps_remaining,
            total_steps=self.total_steps,
            answer_submitted=self.answer_submitted,
            task_difficulty=self.task_difficulty.value
            if isinstance(self.task_difficulty, DifficultyLevel)
            else self.task_difficulty,
            done=self.answer_submitted,
            reward=self.episode_reward,
            metadata={
                "command_history": self.command_history,
                "mode": self.mode.value if isinstance(self.mode, EnvironmentMode) else self.mode,
                "data_source": self.data_source.value
                if isinstance(self.data_source, DataSource)
                else self.data_source,
                "log_source": self.log_source.value if self.log_source else None,
                **self.metadata,
            },
        )


class InvestigationState(BaseModel):
    """Current state of the investigation environment (legacy format)."""

    episode_id: str = Field(..., description="Unique episode identifier")
    step_count: int = Field(default=0, description="Current step number")
    log_file_path: Optional[str] = Field(default=None, description="Path to log file")
    ground_truth: Optional[Dict[str, Any]] = Field(
        default=None, description="Ground truth for grading"
    )
    task_id: Optional[str] = Field(default=None, description="Current task identifier")
    mode: EnvironmentMode = Field(
        default=EnvironmentMode.EVAL, description="Environment operating mode"
    )
    data_source: DataSource = Field(default=DataSource.SYNTHETIC, description="Source of log data")
    log_source: Optional[LogSource] = Field(
        default=None, description="Specific LogHub source if applicable"
    )


class EpisodeResult(BaseModel):
    """Result of a completed episode."""

    episode_id: str
    task_id: str
    difficulty: DifficultyLevel
    reward: float
    component_score: float
    type_score: float
    window_score: float
    efficiency_score: float
    predicted_answer: Optional[SubmitAnswer] = None
    ground_truth: Dict[str, Any]
    steps_used: int
    episode_complete: bool
    decoy_matched: bool = False  # True if agent identified a decoy instead of primary
