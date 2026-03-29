"""
Central Configuration for Log Anomaly Investigation Environment.

This module centralizes all configurable parameters to avoid magic numbers
scattered throughout the codebase.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from models import AnomalyType, DifficultyLevel


# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_ANOMALY_LOG_LEVEL", "WARNING").upper()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
    return logger


# =============================================================================
# Timestamp Parsing Utility
# =============================================================================

# Supported timestamp formats for parsing
TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",  # ISO with microseconds
    "%Y-%m-%dT%H:%M:%S",  # ISO without microseconds
    "%Y-%m-%d %H:%M:%S.%f",  # Space-separated with microseconds
    "%Y-%m-%d %H:%M:%S",  # Space-separated without microseconds
    "%Y-%m-%d-%H.%M.%S.%f",  # BGL format with microseconds
    "%Y-%m-%d-%H.%M.%S",  # BGL format without microseconds
]


def parse_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse a timestamp string into a datetime object.

    Tries multiple common formats used in logs. Returns None if parsing fails.

    Args:
        ts: Timestamp string to parse

    Returns:
        Parsed datetime or None if all formats fail
    """
    if not ts:
        return None

    # Clean up the timestamp
    clean_ts = ts.replace("Z", "+00:00").split("+")[0].strip()

    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(clean_ts, fmt)
        except ValueError:
            continue

    return None


def parse_timestamp_strict(ts: str) -> datetime:
    """
    Parse a timestamp string into a datetime object, raising on failure.

    Args:
        ts: Timestamp string to parse

    Returns:
        Parsed datetime

    Raises:
        ValueError: If timestamp cannot be parsed with any known format
    """
    result = parse_timestamp(ts)
    if result is None:
        raise ValueError(f"Could not parse timestamp: {ts}")
    return result


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
HF_ROUTER_URL = "https://router.huggingface.co/v1"

# LLM generation parameters
LLM_TEMPERATURE = float(os.getenv("LOG_ANOMALY_LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LOG_ANOMALY_LLM_MAX_TOKENS", "500"))

# Output preview lengths for prompts (to avoid context overflow)
OUTPUT_PREVIEW_SHORT = 400  # Short preview for recent command output
OUTPUT_PREVIEW_LONG = 800  # Longer preview for full context


# =============================================================================
# Command History Configuration
# =============================================================================

MAX_COMMAND_HISTORY = 10  # Maximum number of commands to keep in history for reward shaping


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
