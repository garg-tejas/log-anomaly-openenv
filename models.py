"""
Log Anomaly Investigation Environment - Data Models.

This module defines the typed models for actions, observations, and state
used in the Log Anomaly Investigation environment.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


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


class BashCommand(BaseModel):
    """A bash command to execute in the investigation environment."""
    command: str = Field(..., description="The bash command to execute")

    class Config:
        frozen = False


class SubmitAnswer(BaseModel):
    """Submit the final answer for the anomaly investigation."""
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly identified")
    component: str = Field(..., description="Component affected by the anomaly")
    start_time: str = Field(..., description="Start time of anomaly window (ISO format)")
    end_time: str = Field(..., description="End time of anomaly window (ISO format)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in answer")


class InvestigationAction(BaseModel):
    """Actions available to the agent during investigation."""
    action_type: str = Field(..., description="Type of action: 'bash' or 'submit'")
    bash_command: Optional[BashCommand] = Field(default=None, description="Bash command to execute")
    answer: Optional[SubmitAnswer] = Field(default=None, description="Final answer to submit")

    class Config:
        frozen = False


class LogLine(BaseModel):
    """A single normalized log line."""
    timestamp: str = Field(..., description="Normalized timestamp (ISO format)")
    severity: str = Field(..., description="Log severity level")
    component: str = Field(..., description="Source component")
    message: str = Field(..., description="Raw log message")
    raw_line: str = Field(..., description="Original log line")


class InvestigationObservation(BaseModel):
    """Observation returned after each step."""
    command_output: str = Field(default="", description="Output from the bash command")
    command_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of all commands and outputs"
    )
    steps_remaining: int = Field(..., description="Number of steps remaining")
    total_steps: int = Field(default=15, description="Total steps allowed")
    answer_submitted: bool = Field(default=False, description="Whether answer was submitted")
    task_difficulty: DifficultyLevel = Field(..., description="Current task difficulty")
    episode_reward: float = Field(default=0.0, description="Cumulative reward so far")
    mode: EnvironmentMode = Field(
        default=EnvironmentMode.EVAL,
        description="Environment mode (training=full feedback, eval=limited feedback)"
    )
    data_source: DataSource = Field(
        default=DataSource.SYNTHETIC,
        description="Source of log data (synthetic or loghub)"
    )
    log_source: Optional[LogSource] = Field(
        default=None,
        description="Specific log source if using LogHub data"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InvestigationState(BaseModel):
    """Current state of the investigation environment."""
    episode_id: str = Field(..., description="Unique episode identifier")
    step_count: int = Field(default=0, description="Current step number")
    log_file_path: Optional[str] = Field(default=None, description="Path to log file")
    ground_truth: Optional[Dict[str, Any]] = Field(default=None, description="Ground truth for grading")
    task_id: Optional[str] = Field(default=None, description="Current task identifier")
    mode: EnvironmentMode = Field(default=EnvironmentMode.EVAL, description="Environment operating mode")
    data_source: DataSource = Field(default=DataSource.SYNTHETIC, description="Source of log data")
    log_source: Optional[LogSource] = Field(default=None, description="Specific LogHub source if applicable")


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
