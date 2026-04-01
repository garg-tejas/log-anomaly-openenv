"""
Training Prompts for Log Anomaly Investigation Environment.

This module provides curated system and user prompts optimized for
GRPO training at different difficulty levels.

The prompts are designed to:
1. Clearly explain the task and available tools
2. Provide hints appropriate to the difficulty level
3. Guide the agent toward systematic investigation
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TrainingPrompt:
    """A training prompt with system and user components."""

    system: str
    user: str
    difficulty: str

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to message format for chat models."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def to_user_only(self) -> List[Dict[str, str]]:
        """Convert to user-only format (system embedded in user message)."""
        combined = f"{self.system}\n\n---\n\n{self.user}"
        return [{"role": "user", "content": combined}]


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are an expert DevOps engineer and SRE specialist.
Your task is to investigate log files to identify and diagnose system anomalies.

You have access to two tools:
1. `bash(command)` - Execute bash commands to analyze log.txt
2. `submit(anomaly_type, component, start_time, end_time)` - Submit your findings

INVESTIGATION METHODOLOGY:
1. Start by examining the log structure (head, tail, wc -l)
2. Search for error patterns (grep ERROR, grep -i error)
3. Identify affected components (look for service names)
4. Determine the time window (extract timestamps from errors)
5. Submit your answer with all required fields

IMPORTANT:
- Extract EXACT timestamps from the logs (format: YYYY-MM-DDTHH:MM:SS)
- Identify the PRIMARY affected component
- Choose the correct anomaly type based on the patterns you observe"""

SYSTEM_PROMPT_EASY = (
    SYSTEM_PROMPT_BASE
    + """

DIFFICULTY: EASY
- The anomaly type is ERROR_SPIKE
- Look for sudden increases in ERROR log messages
- The pattern should be obvious and concentrated in a time window"""
)

SYSTEM_PROMPT_MEDIUM = (
    SYSTEM_PROMPT_BASE
    + """

DIFFICULTY: MEDIUM
- The anomaly could be: error_spike, memory_leak, latency_degradation, or service_dropout
- memory_leak: Look for increasing heap/memory usage over time
- latency_degradation: Look for increasing response times or timeout patterns
- service_dropout: Look for service unavailability or connection refused patterns
- error_spike: Look for sudden bursts of ERROR messages"""
)

SYSTEM_PROMPT_HARD = (
    SYSTEM_PROMPT_BASE
    + """

DIFFICULTY: HARD
- This is a CASCADE_FAILURE scenario
- Multiple components are affected in sequence
- Your task is to identify the ROOT CAUSE and the propagation chain
- Look for: "caused by", "due to", "affected by", "circuit breaker" patterns
- The primary component is where the cascade STARTED, not where it ended"""
)


# =============================================================================
# User Prompts (Task Descriptions)
# =============================================================================

USER_PROMPT_EASY = """TASK: Investigate log.txt for an ERROR SPIKE anomaly.

An error spike is a sudden, concentrated burst of ERROR-level log messages.
Find when it started, when it ended, and which component was affected.

STEPS:
1. Run: bash("grep ERROR log.txt | wc -l") to count errors
2. Run: bash("grep ERROR log.txt | head -20") to see error patterns
3. Run: bash("grep ERROR log.txt | awk '{print $1, $3}' | sort | uniq -c") to find the component
4. Extract timestamps and submit your answer

BEGIN INVESTIGATION NOW."""

USER_PROMPT_MEDIUM = """TASK: Investigate log.txt for a system anomaly.

The anomaly could be one of:
- error_spike: Sudden burst of errors
- memory_leak: Gradual memory increase (look for heap/MB patterns)
- latency_degradation: Increasing response times (look for ms/latency patterns)
- service_dropout: Service unavailability (look for unavailable/connection refused)

INVESTIGATION STEPS:
1. bash("head -30 log.txt") - Understand the log format
2. bash("grep -iE 'error|warn|fatal' log.txt | head -20") - Find issues
3. bash("grep -iE 'memory|heap|mb' log.txt | head -10") - Check for memory issues
4. bash("grep -iE 'latency|timeout|ms' log.txt | head -10") - Check for latency
5. Identify the pattern and submit

BEGIN INVESTIGATION NOW."""

USER_PROMPT_HARD = """TASK: Investigate log.txt for a CASCADE FAILURE.

A cascade failure is when one component's failure causes other components to fail.
Your goal is to find:
1. The ROOT CAUSE component (where it started)
2. The exact time window of the initial failure
3. The cascade_failure anomaly type

LOOK FOR:
- "initial failure", "root cause", "originated from"
- "caused by [component]", "due to [component] failure"
- "circuit breaker", "degraded mode", "fallback"
- Timestamps showing the propagation sequence

INVESTIGATION STRATEGY:
1. bash("grep -iE 'cascade|failure|caused by|due to' log.txt | head -20")
2. bash("grep -iE 'initial|root|origin|started' log.txt | head -10")
3. bash("grep ERROR log.txt | awk '{print $1, $3}' | sort | uniq -c | sort -rn")
4. Trace back to the FIRST component that failed

The PRIMARY component is where the cascade STARTED.

BEGIN INVESTIGATION NOW."""


# =============================================================================
# Prompt Collections
# =============================================================================

TRAINING_PROMPTS = {
    "easy": TrainingPrompt(
        system=SYSTEM_PROMPT_EASY,
        user=USER_PROMPT_EASY,
        difficulty="easy",
    ),
    "medium": TrainingPrompt(
        system=SYSTEM_PROMPT_MEDIUM,
        user=USER_PROMPT_MEDIUM,
        difficulty="medium",
    ),
    "hard": TrainingPrompt(
        system=SYSTEM_PROMPT_HARD,
        user=USER_PROMPT_HARD,
        difficulty="hard",
    ),
}


def get_prompt(difficulty: str) -> TrainingPrompt:
    """
    Get the training prompt for a difficulty level.

    Args:
        difficulty: One of "easy", "medium", "hard"

    Returns:
        TrainingPrompt for the specified difficulty
    """
    return TRAINING_PROMPTS.get(difficulty, TRAINING_PROMPTS["easy"])


def get_prompt_messages(difficulty: str, include_system: bool = True) -> List[Dict[str, str]]:
    """
    Get prompt messages ready for dataset creation.

    Args:
        difficulty: One of "easy", "medium", "hard"
        include_system: Whether to include system message separately

    Returns:
        List of message dicts for chat format
    """
    prompt = get_prompt(difficulty)
    if include_system:
        return prompt.to_messages()
    return prompt.to_user_only()


# =============================================================================
# Variations for Dataset Diversity
# =============================================================================

USER_PROMPT_VARIATIONS = {
    "easy": [
        USER_PROMPT_EASY,
        """Analyze log.txt. There's an error_spike - find the component and time window.
Use bash commands like grep, awk, head to investigate. Submit when ready.""",
        """Your mission: Detect the ERROR SPIKE in log.txt.
1. Find all ERROR entries
2. Identify which component has the most errors
3. Determine the time range
4. Submit your findings""",
    ],
    "medium": [
        USER_PROMPT_MEDIUM,
        """Investigate log.txt for anomalies. It could be error_spike, memory_leak, 
latency_degradation, or service_dropout. Use grep patterns to identify the type,
then find the affected component and time window.""",
        """System anomaly detected in log.txt. Your task:
1. Determine the anomaly type (error/memory/latency/dropout)
2. Find the affected component
3. Extract the time window
4. Submit a complete diagnosis""",
    ],
    "hard": [
        USER_PROMPT_HARD,
        """CASCADE FAILURE investigation required. Multiple components are failing in sequence.
Trace the failure back to its origin. The root cause component is your target.
Look for dependency patterns and failure propagation timestamps.""",
        """A cascade failure has occurred. Find where it STARTED (root cause).
Search for: "caused by", "affected by", "circuit breaker" patterns.
The PRIMARY component is the origin point, not the final victim.
Submit: anomaly_type="cascade_failure", component=ROOT_CAUSE""",
    ],
}


def get_diverse_prompts(difficulty: str, count: int = 10) -> List[List[Dict[str, str]]]:
    """
    Get diverse prompt variations for dataset creation.

    This helps prevent overfitting to specific prompt wording.

    Args:
        difficulty: One of "easy", "medium", "hard"
        count: Number of prompts to generate

    Returns:
        List of message lists, cycling through variations
    """
    system = TRAINING_PROMPTS[difficulty].system
    variations = USER_PROMPT_VARIATIONS.get(difficulty, [USER_PROMPT_EASY])

    prompts = []
    for i in range(count):
        user = variations[i % len(variations)]
        prompts.append(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    return prompts


# =============================================================================
# Dataset Creation Helpers
# =============================================================================


def create_training_dataset_dict(
    num_easy: int = 100,
    num_medium: int = 100,
    num_hard: int = 100,
    include_system: bool = True,
) -> Dict[str, List]:
    """
    Create a dataset dictionary ready for HuggingFace datasets.

    Args:
        num_easy: Number of easy samples
        num_medium: Number of medium samples
        num_hard: Number of hard samples
        include_system: Whether to include system messages

    Returns:
        Dict with "prompt" and "difficulty" keys
    """
    prompts = []
    difficulties = []

    for diff, count in [("easy", num_easy), ("medium", num_medium), ("hard", num_hard)]:
        diverse = get_diverse_prompts(diff, count)
        prompts.extend(diverse)
        difficulties.extend([diff] * count)

    return {
        "prompt": prompts,
        "difficulty": difficulties,
    }


__all__ = [
    "TrainingPrompt",
    "TRAINING_PROMPTS",
    "get_prompt",
    "get_prompt_messages",
    "get_diverse_prompts",
    "create_training_dataset_dict",
    "SYSTEM_PROMPT_EASY",
    "SYSTEM_PROMPT_MEDIUM",
    "SYSTEM_PROMPT_HARD",
]
