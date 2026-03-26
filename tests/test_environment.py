# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for Log Anomaly Investigation Environment.

Run with: pytest tests/ -v
"""
import pytest
import os
import tempfile
import shutil
from datetime import datetime

from models import (
    InvestigationAction,
    InvestigationObservation,
    InvestigationState,
    DifficultyLevel,
    AnomalyType,
    SubmitAnswer,
    BashCommand,
    LogLine,
)
from log_utils import LogParser, AnomalyInjector, generate_synthetic_log
from grader import InvestigationGrader, TaskGenerator, calculate_summary_stats
from server.log_anomaly_environment import LogAnomalyEnvironment, InvestigationEpisode


class TestLogParser:
    """Tests for LogParser class."""

    def test_parse_hdfs_line(self):
        """Test parsing HDFS-style log format."""
        parser = LogParser("hdfs")
        line = "2024-01-15T10:23:45.123Z INFO BlockManager Starting block recovery"
        result = parser.parse_line(line)
        assert result is not None
        assert result.severity == "INFO"
        assert "BlockManager" in result.component

    def test_parse_generic_line(self):
        """Test generic log parsing fallback."""
        parser = LogParser("unknown")
        line = "[2024-01-15 10:23:45] [service_a] ERROR: Connection timeout"
        result = parser.parse_line(line)
        assert result is not None
        assert result.severity == "ERROR"

    def test_parse_file(self):
        """Test parsing a complete log file."""
        parser = LogParser("generic")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("2024-01-15 INFO component_a Message 1\n")
            f.write("2024-01-15 WARN component_b Message 2\n")
            f.write("2024-01-15 ERROR component_a Message 3\n")
            temp_path = f.name

        try:
            result = parser.parse_file(temp_path)
            assert len(result.lines) == 3
            assert result.start_time is not None
            assert result.end_time is not None
            assert len(result.components) >= 1
        finally:
            os.unlink(temp_path)


class TestAnomalyInjector:
    """Tests for AnomalyInjector class."""

    def test_inject_error_spike(self):
        """Test error spike injection."""
        injector = AnomalyInjector(seed=42)
        logs, _ = generate_synthetic_log(num_lines=100, seed=42)
        modified, gt = injector.inject(logs, AnomalyType.ERROR_SPIKE, intensity=0.5, seed=42)
        assert gt["anomaly_type"] == "error_spike"
        assert "component" in gt
        assert "start_time" in gt

    def test_inject_memory_leak(self):
        """Test memory leak injection."""
        injector = AnomalyInjector(seed=123)
        logs, _ = generate_synthetic_log(num_lines=100, seed=123)
        modified, gt = injector.inject(logs, AnomalyType.MEMORY_LEAK, intensity=0.5, seed=123)
        assert gt["anomaly_type"] == "memory_leak"
        assert "peak_memory" in gt

    def test_inject_service_dropout(self):
        """Test service dropout injection."""
        injector = AnomalyInjector(seed=456)
        logs, _ = generate_synthetic_log(num_lines=100, seed=456)
        original_count = len(logs)
        modified, gt = injector.inject(logs, AnomalyType.SERVICE_DROPOUT, intensity=0.5, seed=456)
        assert len(modified) < original_count  # Some logs should be removed
        assert gt["anomaly_type"] == "service_dropout"

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        logs, _ = generate_synthetic_log(num_lines=50, seed=100)
        injector1 = AnomalyInjector(seed=200)
        _, gt1 = injector1.inject(logs, AnomalyType.ERROR_SPIKE, intensity=0.5, seed=200)
        injector2 = AnomalyInjector(seed=200)
        _, gt2 = injector2.inject(logs, AnomalyType.ERROR_SPIKE, intensity=0.5, seed=200)
        assert gt1["component"] == gt2["component"]


class TestGrader:
    """Tests for InvestigationGrader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.grader = InvestigationGrader()

    def test_grade_exact_match(self):
        """Test grading with exact match."""
        prediction = SubmitAnswer(
            anomaly_type=AnomalyType.ERROR_SPIKE,
            component="service_a",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:30:00",
        )
        ground_truth = {
            "anomaly_type": "error_spike",
            "component": "service_a",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T10:30:00",
            "episode_id": "test_1",
            "task_id": "test_task",
            "difficulty": "easy",
        }
        result = self.grader.grade(prediction, ground_truth, steps_used=5, total_steps=15)
        assert result.reward > 0.9  # Near perfect score
        assert result.component_score == 1.0
        assert result.type_score == 1.0

    def test_grade_partial_component(self):
        """Test grading with partial component match."""
        prediction = SubmitAnswer(
            anomaly_type=AnomalyType.ERROR_SPIKE,
            component="service",  # Partial match (actual is "service_a")
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:30:00",
        )
        ground_truth = {
            "anomaly_type": "error_spike",
            "component": "service_a",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T10:30:00",
            "episode_id": "test_2",
            "task_id": "test_task",
            "difficulty": "easy",
        }
        result = self.grader.grade(prediction, ground_truth, steps_used=5)
        assert result.component_score > 0.0
        assert result.component_score < 1.0

    def test_grade_wrong_type(self):
        """Test grading with wrong anomaly type."""
        prediction = SubmitAnswer(
            anomaly_type=AnomalyType.MEMORY_LEAK,  # Wrong type
            component="service_a",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:30:00",
        )
        ground_truth = {
            "anomaly_type": "error_spike",
            "component": "service_a",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T10:30:00",
            "episode_id": "test_3",
            "task_id": "test_task",
            "difficulty": "easy",
        }
        result = self.grader.grade(prediction, ground_truth, steps_used=5)
        assert result.type_score == 0.0

    def test_grade_no_answer(self):
        """Test grading when no answer submitted."""
        ground_truth = {
            "anomaly_type": "error_spike",
            "component": "service_a",
            "start_time": "2024-01-15T10:00:00",
            "end_time": "2024-01-15T10:30:00",
            "episode_id": "test_4",
            "task_id": "test_task",
            "difficulty": "easy",
        }
        result = self.grader.grade(None, ground_truth, steps_used=15)
        assert result.reward == 0.0
        assert result.episode_complete is False


class TestTaskGenerator:
    """Tests for TaskGenerator class."""

    def test_get_task_config(self):
        """Test getting task configuration."""
        grader = InvestigationGrader()
        generator = TaskGenerator(grader)
        config = generator.get_task_config(DifficultyLevel.EASY)
        assert config["intensity"] == 0.8
        assert AnomalyType.ERROR_SPIKE in config["allowed_anomaly_types"]

    def test_list_tasks(self):
        """Test listing all tasks."""
        grader = InvestigationGrader()
        generator = TaskGenerator(grader)
        tasks = generator.list_tasks()
        assert "easy" in tasks
        assert "medium" in tasks
        assert "hard" in tasks
        assert "action_schema" in tasks["easy"]


class TestLogAnomalyEnvironment:
    """Tests for LogAnomalyEnvironment class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.env = LogAnomalyEnvironment()

    def teardown_method(self):
        """Clean up after tests."""
        self.env.close()

    def test_reset_easy(self):
        """Test environment reset with easy difficulty."""
        obs = self.env.reset(difficulty="easy", seed=42)
        assert obs.steps_remaining == 15
        assert obs.task_difficulty == DifficultyLevel.EASY
        assert obs.answer_submitted is False
        assert self.env.state.episode_id is not None

    def test_reset_medium(self):
        """Test environment reset with medium difficulty."""
        obs = self.env.reset(difficulty="medium", seed=123)
        assert obs.steps_remaining == 15
        assert obs.task_difficulty == DifficultyLevel.MEDIUM

    def test_reset_hard(self):
        """Test environment reset with hard difficulty."""
        obs = self.env.reset(difficulty="hard", seed=456)
        assert obs.steps_remaining == 15
        assert obs.task_difficulty == DifficultyLevel.HARD

    def test_step_bash_command(self):
        """Test executing a bash command."""
        self.env.reset(difficulty="easy", seed=42)
        action = InvestigationAction(
            action_type="bash",
            bash_command=BashCommand(command="cat log.txt | head -5")
        )
        obs = self.env.step(action)
        assert obs.steps_remaining == 14
        assert len(obs.command_history) == 1
        assert obs.command_history[0]["command"] == "cat log.txt | head -5"

    def test_step_forbidden_command(self):
        """Test that forbidden commands are rejected."""
        self.env.reset(difficulty="easy", seed=42)
        action = InvestigationAction(
            action_type="bash",
            bash_command=BashCommand(command="rm -rf /")
        )
        obs = self.env.step(action)
        assert "not allowed" in obs.command_output.lower() or "forbidden" in obs.command_output.lower()

    def test_submit_answer(self):
        """Test submitting an answer."""
        self.env.reset(difficulty="easy", seed=42)
        gt = self.env.state.ground_truth

        action = InvestigationAction(
            action_type="submit",
            answer=SubmitAnswer(
                anomaly_type=AnomalyType(gt["anomaly_type"]),
                component=gt["component"],
                start_time=gt["start_time"],
                end_time=gt["end_time"],
            )
        )
        obs = self.env.step(action)
        assert obs.answer_submitted is True
        assert obs.steps_remaining == 0
        assert obs.episode_reward > 0

    def test_multiple_steps(self):
        """Test multiple sequential steps."""
        self.env.reset(difficulty="easy", seed=42)
        for _ in range(5):
            action = InvestigationAction(
                action_type="bash",
                bash_command=BashCommand(command="head -10 log.txt")
            )
            self.env.step(action)
        assert self.env.state.step_count == 5

    def test_list_tasks(self):
        """Test listing available tasks."""
        tasks = self.env.list_tasks()
        assert "easy" in tasks
        assert "medium" in tasks
        assert "hard" in tasks


class TestCalculateSummaryStats:
    """Tests for calculate_summary_stats function."""

    def test_empty_results(self):
        """Test summary with no results."""
        stats = calculate_summary_stats([])
        assert stats["count"] == 0
        assert stats["mean_reward"] == 0.0

    def test_single_result(self):
        """Test summary with single result."""
        from models import EpisodeResult, DifficultyLevel
        result = EpisodeResult(
            episode_id="test",
            task_id="test",
            difficulty=DifficultyLevel.EASY,
            reward=0.75,
            component_score=0.8,
            type_score=1.0,
            window_score=0.7,
            efficiency_score=0.5,
            steps_used=5,
            episode_complete=True,
            ground_truth={},
        )
        stats = calculate_summary_stats([result])
        assert stats["count"] == 1
        assert stats["mean_reward"] == 0.75


class TestIntegration:
    """Integration tests for full episode workflow."""

    def test_full_episode(self):
        """Test a complete episode from reset to grading."""
        env = LogAnomalyEnvironment()

        # Reset
        obs = env.reset(difficulty="easy", seed=999)
        episode_id = env.state.episode_id
        gt = env.state.ground_truth

        # Execute some commands
        commands = [
            "head -20 log.txt",
            "grep ERROR log.txt | head -10",
            "awk '{print $3}' log.txt | sort | uniq -c | sort -rn",
        ]
        for cmd in commands:
            action = InvestigationAction(
                action_type="bash",
                bash_command=BashCommand(command=cmd)
            )
            env.step(action)

        # Submit answer
        action = InvestigationAction(
            action_type="submit",
            answer=SubmitAnswer(
                anomaly_type=AnomalyType(gt["anomaly_type"]),
                component=gt["component"],
                start_time=gt["start_time"],
                end_time=gt["end_time"],
            )
        )
        env.step(action)

        # Get result
        result = env.get_result(episode_id)
        assert result is not None
        assert result.episode_id == episode_id
        assert result.reward >= 0
        assert result.episode_complete is True

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
