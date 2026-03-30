"""
Grader System for Log Anomaly Investigation.

This module provides the grading functionality for evaluating agent
performance on log anomaly investigation tasks.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Support both package and direct execution modes
if __package__:
    from .models import AnomalyType, DifficultyLevel, SubmitAnswer, EpisodeResult
    from .config import (
        COMPONENT_WEIGHT,
        TYPE_WEIGHT,
        WINDOW_WEIGHT,
        EFFICIENCY_WEIGHT,
        REWARD_TIMEOUT,
        REWARD_WRONG_TYPE_PENALTY,
        REWARD_WRONG_COMPONENT_PENALTY,
        REWARD_DECOY_PENALTY,
        REWARD_PERFECT_FAST_BONUS,
        get_difficulty_config,
        get_logger,
        parse_timestamp_strict,
    )
else:
    # Direct execution mode
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import AnomalyType, DifficultyLevel, SubmitAnswer, EpisodeResult
    from config import (
        COMPONENT_WEIGHT,
        TYPE_WEIGHT,
        WINDOW_WEIGHT,
        EFFICIENCY_WEIGHT,
        REWARD_TIMEOUT,
        REWARD_WRONG_TYPE_PENALTY,
        REWARD_WRONG_COMPONENT_PENALTY,
        REWARD_DECOY_PENALTY,
        REWARD_PERFECT_FAST_BONUS,
        get_difficulty_config,
        get_logger,
        parse_timestamp_strict,
    )

# Set up logging for this module
logger = get_logger(__name__)


class InvestigationGrader:
    """
    Grades agent performance on log anomaly investigation tasks.

    The grader evaluates predictions against ground truth using multiple
    metrics with uniform 0.0-1.0 scoring:
    - Component identification (0.25 weight)
    - Anomaly type classification (0.25 weight)
    - Time window precision (0.35 weight)
    - Investigation efficiency (0.15 weight)

    Difficulty emerges from task complexity (decoys, log size, anomaly types)
    rather than different reward ranges.
    """

    # Reward weights - import from central config
    COMPONENT_WEIGHT = COMPONENT_WEIGHT
    TYPE_WEIGHT = TYPE_WEIGHT
    WINDOW_WEIGHT = WINDOW_WEIGHT
    EFFICIENCY_WEIGHT = EFFICIENCY_WEIGHT

    def grade(
        self,
        prediction: Optional[SubmitAnswer],
        ground_truth: Dict[str, Any],
        steps_used: int,
        total_steps: int = 15,
    ) -> EpisodeResult:
        """
        Grade an episode with uniform reward range (0.0 to 1.0).

        Reward calculation:
        - Base: weighted sum of component, type, window, efficiency scores (0-1)
        - Small penalties: wrong type/component (-0.1 each), decoy match (-0.1)
        - Small bonus: perfect answer + high efficiency (up to +0.2)
        - All outputs clamped to [0.0, 1.0] range

        Args:
            prediction: Agent's submitted answer
            ground_truth: Ground truth metadata
            steps_used: Number of steps taken
            total_steps: Total available steps

        Returns:
            EpisodeResult with detailed scores
        """
        difficulty = DifficultyLevel(ground_truth.get("difficulty", "easy"))

        if prediction is None:
            # No answer submitted
            return EpisodeResult(
                episode_id=ground_truth.get("episode_id", "unknown"),
                task_id=ground_truth.get("task_id", "unknown"),
                difficulty=difficulty,
                reward=REWARD_TIMEOUT,
                component_score=0.0,
                type_score=0.0,
                window_score=0.0,
                efficiency_score=0.0,
                predicted_answer=None,
                ground_truth=ground_truth,
                steps_used=steps_used,
                episode_complete=False,
            )

        # Calculate component score
        component_score = self._grade_component(
            prediction.component,
            str(ground_truth.get("component", "")),
        )

        # Calculate type score
        type_score = self._grade_type(
            prediction.anomaly_type,
            AnomalyType(ground_truth.get("anomaly_type", "error_spike")),
        )

        # Calculate window score
        window_score = self._grade_window(
            prediction.start_time,
            prediction.end_time,
            str(ground_truth.get("start_time", "")),
            str(ground_truth.get("end_time", "")),
        )

        # Calculate efficiency score
        efficiency_score = self._grade_efficiency(steps_used, total_steps)

        # Check if prediction matches a decoy instead of primary anomaly
        decoy_matched = self._check_decoy_match(prediction, ground_truth)

        # === REWARD CALCULATION (0.0 to 1.0) ===
        # Start with weighted base score
        base_reward = (
            component_score * self.COMPONENT_WEIGHT
            + type_score * self.TYPE_WEIGHT
            + window_score * self.WINDOW_WEIGHT
            + efficiency_score * self.EFFICIENCY_WEIGHT
        )

        # Apply small penalties for wrong answers
        penalties = 0.0
        answer_given = (
            prediction.component
            or prediction.anomaly_type
            or prediction.start_time
            or prediction.end_time
        )
        if answer_given and type_score == 0.0:
            penalties += REWARD_WRONG_TYPE_PENALTY  # -0.1
        if answer_given and component_score == 0.0:
            penalties += REWARD_WRONG_COMPONENT_PENALTY  # -0.1
        if decoy_matched:
            penalties += REWARD_DECOY_PENALTY  # -0.1

        # Apply small bonus for perfect + fast answers
        bonus = 0.0
        is_perfect = (
            component_score == 1.0
            and type_score == 1.0
            and window_score >= 0.8  # Allow some window tolerance
        )
        if is_perfect:
            # Small efficiency bonus (max +0.2)
            bonus = efficiency_score * REWARD_PERFECT_FAST_BONUS

        # Final reward: base + bonus + penalties
        total_reward = base_reward + bonus + penalties

        # Clamp to [0.0, 1.0] range
        total_reward = max(0.0, min(1.0, total_reward))

        return EpisodeResult(
            episode_id=ground_truth.get("episode_id", "unknown"),
            task_id=ground_truth.get("task_id", "unknown"),
            difficulty=difficulty,
            reward=round(total_reward, 4),
            component_score=round(component_score, 4),
            type_score=round(type_score, 4),
            window_score=round(window_score, 4),
            efficiency_score=round(efficiency_score, 4),
            predicted_answer=prediction,
            ground_truth=ground_truth,
            steps_used=steps_used,
            episode_complete=True,
            decoy_matched=decoy_matched,
        )

        # Calculate component score
        component_score = self._grade_component(
            prediction.component,
            str(ground_truth.get("component", "")),
        )

        # Calculate type score
        type_score = self._grade_type(
            prediction.anomaly_type,
            AnomalyType(ground_truth.get("anomaly_type", "error_spike")),
        )

        # Calculate window score
        window_score = self._grade_window(
            prediction.start_time,
            prediction.end_time,
            str(ground_truth.get("start_time", "")),
            str(ground_truth.get("end_time", "")),
        )

        # Calculate efficiency score
        efficiency_score = self._grade_efficiency(steps_used, total_steps)

        # Check if prediction matches a decoy instead of primary anomaly
        decoy_matched = self._check_decoy_match(prediction, ground_truth)

        # === GRADUATED REWARD CALCULATION ===
        # Start with weighted base score (0-1 range)
        base_reward = (
            component_score * self.COMPONENT_WEIGHT
            + type_score * self.TYPE_WEIGHT
            + window_score * self.WINDOW_WEIGHT
            + efficiency_score * self.EFFICIENCY_WEIGHT
        )

        # Apply penalties for wrong answers (using difficulty-specific penalties)
        penalties = 0.0
        answer_given = (
            prediction.component
            or prediction.anomaly_type
            or prediction.start_time
            or prediction.end_time
        )
        if answer_given and type_score == 0.0:
            penalties += reward_config.wrong_type_penalty
        if answer_given and component_score == 0.0:
            penalties += reward_config.wrong_component_penalty
        if decoy_matched:
            penalties += reward_config.decoy_penalty

        # Apply bonus for perfect + fast answers
        bonus = 0.0
        is_perfect = (
            component_score == 1.0
            and type_score == 1.0
            and window_score >= 0.8  # Allow some window tolerance
        )
        if is_perfect:
            # Scale bonus by efficiency using difficulty-specific multiplier
            # Perfect efficiency (1.0) = full bonus, low efficiency (0.0) = no bonus
            bonus = efficiency_score * reward_config.efficiency_bonus_multiplier

        # Final reward: base + bonus + penalties
        total_reward = base_reward + bonus + penalties

        # Clamp to difficulty-specific range
        total_reward = max(reward_config.min_reward, min(reward_config.max_reward, total_reward))

        return EpisodeResult(
            episode_id=ground_truth.get("episode_id", "unknown"),
            task_id=ground_truth.get("task_id", "unknown"),
            difficulty=difficulty,
            reward=round(total_reward, 4),
            component_score=round(component_score, 4),
            type_score=round(type_score, 4),
            window_score=round(window_score, 4),
            efficiency_score=round(efficiency_score, 4),
            predicted_answer=prediction,
            ground_truth=ground_truth,
            steps_used=steps_used,
            episode_complete=True,
            decoy_matched=decoy_matched,
        )

    def _grade_component(
        self,
        predicted: str,
        actual: str,
    ) -> float:
        """
        Grade component identification.

        Returns 1.0 for exact match, partial credit for partial matches.

        Args:
            predicted: Predicted component name
            actual: Actual component name

        Returns:
            Score between 0.0 and 1.0
        """
        if not predicted or not actual:
            return 0.0

        predicted = predicted.lower().strip()
        actual = actual.lower().strip()

        if predicted == actual:
            return 1.0

        # Check for substring match
        if predicted in actual or actual in predicted:
            return 0.75

        # Check for common prefix
        if predicted.split("_")[-1] == actual.split("_")[-1]:
            return 0.5

        return 0.0

    def _grade_type(
        self,
        predicted: AnomalyType,
        actual: AnomalyType,
    ) -> float:
        """
        Grade anomaly type classification.

        Returns 1.0 for exact match, 0.0 otherwise (categorical).

        Args:
            predicted: Predicted anomaly type
            actual: Actual anomaly type

        Returns:
            1.0 or 0.0
        """
        if predicted == actual:
            return 1.0
        return 0.0

    def _grade_window(
        self,
        pred_start: str,
        pred_end: str,
        actual_start: str,
        actual_end: str,
    ) -> float:
        """
        Grade time window precision using IoU.

        Args:
            pred_start: Predicted window start
            pred_end: Predicted window end
            actual_start: Actual window start
            actual_end: Actual window end

        Returns:
            Score between 0.0 and 1.0 based on IoU
        """
        try:
            pred_start_dt = self._parse_timestamp(pred_start)
            pred_end_dt = self._parse_timestamp(pred_end)
            actual_start_dt = self._parse_timestamp(actual_start)
            actual_end_dt = self._parse_timestamp(actual_end)
        except Exception as e:
            logger.debug("Failed to parse timestamps for window grading: %s", e)
            return 0.0

        # Calculate intersection over union
        intersection_start = max(pred_start_dt, actual_start_dt)
        intersection_end = min(pred_end_dt, actual_end_dt)

        if intersection_start >= intersection_end:
            # No overlap
            return 0.0

        intersection = (intersection_end - intersection_start).total_seconds()
        union = (
            max(pred_end_dt, actual_end_dt) - min(pred_start_dt, actual_start_dt)
        ).total_seconds()

        if union <= 0:
            return 0.0

        iou = intersection / union

        # Bonus for precision (narrow windows that still cover the anomaly)
        pred_duration = (pred_end_dt - pred_start_dt).total_seconds()
        actual_duration = (actual_end_dt - actual_start_dt).total_seconds()

        if pred_duration > 0:
            precision = actual_duration / pred_duration
            # Score is IoU weighted by how well the prediction captures the anomaly
            return iou * (0.7 + 0.3 * min(precision, 1.0))

        return iou

    def _grade_efficiency(
        self,
        steps_used: int,
        total_steps: int = 15,
    ) -> float:
        """
        Grade investigation efficiency.

        More efficient solutions (using fewer steps) get higher scores.

        Args:
            steps_used: Number of steps taken
            total_steps: Total available steps

        Returns:
            Score between 0.0 and 1.0, linear decay from max steps to min
        """
        if steps_used <= 0:
            return 1.0  # Perfect efficiency (impossible case)

        if steps_used >= total_steps:
            return 0.0  # Used all steps, minimum efficiency

        # Linear decay: 1.0 at 1 step, 0.0 at total_steps
        efficiency = 1.0 - ((steps_used - 1) / (total_steps - 1))
        return max(0.0, min(1.0, efficiency))

    def _check_decoy_match(
        self,
        prediction: SubmitAnswer,
        ground_truth: Dict[str, Any],
    ) -> bool:
        """
        Check if the prediction matches a decoy anomaly instead of the primary.

        Decoys are stored in ground_truth["decoys"] as a list of dicts with:
        - anomaly_type: str
        - component: str
        - start_time: str
        - end_time: str

        Args:
            prediction: Agent's submitted answer
            ground_truth: Ground truth with optional decoys list

        Returns:
            True if prediction matches a decoy, False otherwise
        """
        decoys = ground_truth.get("decoys", [])
        if not decoys:
            return False

        pred_type = prediction.anomaly_type.value
        pred_component = prediction.component.lower().strip()

        for decoy in decoys:
            decoy_type = decoy.get("anomaly_type", "")
            decoy_component = decoy.get("component", "").lower().strip()

            # Check if prediction matches this decoy
            type_match = pred_type == decoy_type
            component_match = (
                pred_component == decoy_component
                or pred_component in decoy_component
                or decoy_component in pred_component
            )

            if type_match and component_match:
                logger.debug(
                    "Prediction matches decoy: type=%s, component=%s",
                    decoy_type,
                    decoy_component,
                )
                return True

        return False

    def _parse_timestamp(self, ts: str) -> datetime:
        """Parse various timestamp formats using central utility."""
        return parse_timestamp_strict(ts)


class TaskGenerator:
    """
    Generates investigation tasks with varying difficulty.

    Tasks are created by combining:
    - Log data (real or synthetic)
    - Injected anomalies
    - Difficulty settings (from central config)
    """

    def __init__(self, grader: InvestigationGrader):
        """
        Initialize the task generator.

        Args:
            grader: Grader instance for evaluation
        """
        self.grader = grader

    def get_task_config(self, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """
        Get configuration for a difficulty level.

        Args:
            difficulty: Difficulty level

        Returns:
            Task configuration dictionary
        """
        config = get_difficulty_config(difficulty)
        return {
            "intensity": config.intensity,
            "window_size_factor": config.window_size_factor,
            "allowed_anomaly_types": list(config.allowed_anomaly_types),
        }

    def list_tasks(self) -> Dict[str, Any]:
        """
        List all available tasks with metadata.

        Returns:
            Dictionary of task definitions
        """
        tasks = {}
        for difficulty in DifficultyLevel:
            config = self.get_task_config(difficulty)
            tasks[difficulty.value] = {
                "difficulty": difficulty.value,
                "description": self._get_task_description(difficulty),
                "intensity_range": (
                    config["intensity"] * 0.8,
                    min(1.0, config["intensity"] * 1.2),
                ),
                "anomaly_types": [t.value for t in config["allowed_anomaly_types"]],
                "action_schema": {
                    "action_type": "string (bash | submit)",
                    "bash_command": {
                        "command": "string (bash command)",
                    },
                    "answer": {
                        "anomaly_type": "string (error_spike | memory_leak | service_dropout | cascade_failure | latency_degradation | auth_anomaly)",
                        "component": "string",
                        "start_time": "ISO timestamp string",
                        "end_time": "ISO timestamp string",
                    },
                },
            }
        return tasks

    def _get_task_description(self, difficulty: DifficultyLevel) -> str:
        """Get human-readable task description."""
        descriptions = {
            DifficultyLevel.EASY: (
                "Find and identify obvious error spikes in system logs. "
                "The anomaly is clearly visible with high error rates."
            ),
            DifficultyLevel.MEDIUM: (
                "Investigate logs to identify anomalies of various types. "
                "Requires correlation across multiple log dimensions."
            ),
            DifficultyLevel.HARD: (
                "Investigate cascade failures across multiple components. "
                "Requires temporal reasoning to identify root cause and propagation chain."
            ),
        }
        return descriptions.get(difficulty, "")


def calculate_summary_stats(results: list, include_by_difficulty: bool = True) -> Dict[str, Any]:
    """
    Calculate summary statistics from multiple episode results.

    Args:
        results: List of EpisodeResult objects
        include_by_difficulty: Whether to include breakdown by difficulty

    Returns:
        Summary statistics dictionary
    """
    if not results:
        return {
            "count": 0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_component": 0.0,
            "mean_type": 0.0,
            "mean_window": 0.0,
            "mean_efficiency": 0.0,
        }

    rewards = [r.reward for r in results]
    component_scores = [r.component_score for r in results]
    type_scores = [r.type_score for r in results]
    window_scores = [r.window_score for r in results]
    efficiency_scores = [r.efficiency_score for r in results]

    import statistics

    stats = {
        "count": len(results),
        "mean_reward": round(statistics.mean(rewards), 4),
        "std_reward": round(statistics.stdev(rewards) if len(rewards) > 1 else 0, 4),
        "mean_component": round(statistics.mean(component_scores), 4),
        "mean_type": round(statistics.mean(type_scores), 4),
        "mean_window": round(statistics.mean(window_scores), 4),
        "mean_efficiency": round(statistics.mean(efficiency_scores), 4),
    }

    # Only add by_difficulty at top level to prevent infinite recursion
    if include_by_difficulty:
        stats["by_difficulty"] = {
            str(d.value): calculate_summary_stats(
                [r for r in results if r.difficulty == d], include_by_difficulty=False
            )
            for d in set(r.difficulty for r in results)
        }

    return stats
