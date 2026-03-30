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
# Increased from 15 to 20 to give models time to complete investigations
# (Most episodes were hitting 15-step timeout without submitting)
MAX_STEPS = int(os.getenv("LOG_ANOMALY_MAX_STEPS", "20"))

# Truncate command output to this many characters
# Increased to allow fuller log viewing (models have large context windows)
OUTPUT_TRUNCATION = int(os.getenv("LOG_ANOMALY_OUTPUT_TRUNCATION", "32000"))

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
# Reward Configuration (0.0 to 1.0 range for all difficulties)
# =============================================================================
# Simple, uniform reward function across all difficulties.
# Difficulty emerges naturally from:
# - Task complexity (log size, anomaly types)
# - Hidden state (decoys in medium/hard modes)
# - Cascade failures (hard mode only)

REWARD_TIMEOUT = 0.0  # No submission before max steps
REWARD_WRONG_TYPE_PENALTY = -0.1  # Small penalty for incorrect anomaly type
REWARD_WRONG_COMPONENT_PENALTY = -0.1  # Small penalty for incorrect component
REWARD_DECOY_PENALTY = -0.1  # Penalty for identifying decoy as primary anomaly
REWARD_PERFECT_FAST_BONUS = 0.2  # Bonus for perfect + fast answer
REWARD_REPEAT_COMMAND_PENALTY = -0.3  # Penalty for repeating same command (blocked)
REWARD_REPEAT_WARNING_PENALTY = -0.1  # Warning penalty for first repeat


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
    num_decoys: int = 0  # Number of decoy anomalies to inject (hidden state)

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
        num_decoys=0,  # No decoys for easy mode
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
        num_decoys=1,  # 1 decoy for medium mode
    ),
    DifficultyLevel.HARD: DifficultyConfig(
        num_lines=2000,
        intensity=0.6,
        window_size_factor=0.15,
        allowed_anomaly_types=(AnomalyType.CASCADE_FAILURE,),
        num_decoys=2,  # 2 decoys for hard mode
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
# Optimized for Qwen3.5 native thinking mode (recommended by Qwen team)
LLM_TEMPERATURE = float(os.getenv("LOG_ANOMALY_LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LOG_ANOMALY_LLM_MAX_TOKENS", "8192"))
LLM_TOP_P = float(os.getenv("LOG_ANOMALY_LLM_TOP_P", "0.95"))
LLM_PRESENCE_PENALTY = float(os.getenv("LOG_ANOMALY_LLM_PRESENCE_PENALTY", "1.5"))

# Output preview lengths for prompts
# Increased significantly since modern LLMs have large context windows (262K for Qwen3.5)
OUTPUT_PREVIEW_SHORT = 8192  # Recent command output preview
OUTPUT_PREVIEW_LONG = 16384  # Full context preview

# =============================================================================
# Command History Configuration
# =============================================================================

# Maximum number of commands to keep in history
MAX_COMMAND_HISTORY = 25


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
