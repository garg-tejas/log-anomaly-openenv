"""
Central Configuration for Log Anomaly Investigation Environment.

This module centralizes all configurable parameters to avoid magic numbers
scattered throughout the codebase.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

from models import AnomalyType, DifficultyLevel


# =============================================================================
# Environment Configuration
# =============================================================================

# Maximum steps allowed per episode
MAX_STEPS = int(os.getenv("LOG_ANOMALY_MAX_STEPS", "15"))

# Truncate command output to this many characters
OUTPUT_TRUNCATION = int(os.getenv("LOG_ANOMALY_OUTPUT_TRUNCATION", "4000"))

# Command execution timeout in seconds
COMMAND_TIMEOUT = int(os.getenv("LOG_ANOMALY_COMMAND_TIMEOUT", "10"))

# Minimum steps before allowing submit (to prevent premature guessing)
MIN_STEPS_BEFORE_SUBMIT = 3


# =============================================================================
# Grading Weights
# =============================================================================

COMPONENT_WEIGHT = 0.25  # Weight for correct component identification
TYPE_WEIGHT = 0.25  # Weight for correct anomaly type
WINDOW_WEIGHT = 0.35  # Weight for accurate time window
EFFICIENCY_WEIGHT = 0.15  # Weight for step efficiency


# =============================================================================
# Difficulty Configuration
# =============================================================================


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration for a difficulty level."""

    num_lines: int
    intensity: float
    window_size_factor: float
    allowed_anomaly_types: Tuple[AnomalyType, ...]

    @property
    def anomaly_type_values(self) -> List[str]:
        """Get anomaly types as string values."""
        return [t.value for t in self.allowed_anomaly_types]


DIFFICULTY_CONFIGS = {
    DifficultyLevel.EASY: DifficultyConfig(
        num_lines=500,
        intensity=0.8,
        window_size_factor=0.3,
        allowed_anomaly_types=(AnomalyType.ERROR_SPIKE,),
    ),
    DifficultyLevel.MEDIUM: DifficultyConfig(
        num_lines=1000,
        intensity=0.5,
        window_size_factor=0.2,
        allowed_anomaly_types=(
            AnomalyType.ERROR_SPIKE,
            AnomalyType.MEMORY_LEAK,
            AnomalyType.SERVICE_DROPOUT,
            AnomalyType.LATENCY_DEGRADATION,
        ),
    ),
    DifficultyLevel.HARD: DifficultyConfig(
        num_lines=2000,
        intensity=0.6,
        window_size_factor=0.15,
        allowed_anomaly_types=(AnomalyType.CASCADE_FAILURE,),
    ),
}


def get_difficulty_config(difficulty: DifficultyLevel) -> DifficultyConfig:
    """Get configuration for a difficulty level."""
    return DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS[DifficultyLevel.EASY])


# =============================================================================
# Model Configuration
# =============================================================================

DEFAULT_MODEL = os.getenv("LOG_ANOMALY_DEFAULT_MODEL", "Qwen/Qwen3.5-4B")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")


# =============================================================================
# Allowed Commands (Security)
# =============================================================================

ALLOWED_COMMANDS = [
    "grep",
    "egrep",
    "fgrep",
    "awk",
    "sed",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "sort",
    "uniq",
    "wc",
    "cut",
    "tr",
    "tee",
    "diff",
    "comm",
    "xargs",
    "find",  # Limited to current directory
    "ls",
    "stat",
    "file",
    "echo",
    "printf",
    "date",
]
