"""
Quick test script to verify the environment works.

Run this after installation to validate everything is set up correctly.
"""
import sys
import traceback


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from models import (
            InvestigationAction,
            InvestigationObservation,
            DifficultyLevel,
            AnomalyType,
            SubmitAnswer,
            BashCommand,
        )
        from log_utils import LogParser, AnomalyInjector, generate_synthetic_log
        from grader import InvestigationGrader, TaskGenerator
        from server.log_anomaly_environment import LogAnomalyEnvironment
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_log_generation():
    """Test synthetic log generation."""
    print("Testing log generation...")
    try:
        from log_utils import generate_synthetic_log
        logs, metadata = generate_synthetic_log(num_lines=100, seed=42)
        assert len(logs) == 100
        assert "num_lines" in metadata
        assert "components" in metadata
        print(f"  ✓ Generated {len(logs)} log lines with {len(metadata['components'])} components")
        return True
    except Exception as e:
        print(f"  ✗ Log generation failed: {e}")
        traceback.print_exc()
        return False


def test_anomaly_injection():
    """Test anomaly injection."""
    print("Testing anomaly injection...")
    try:
        from log_utils import generate_synthetic_log, AnomalyInjector
        from models import AnomalyType

        injector = AnomalyInjector(seed=123)
        logs, _ = generate_synthetic_log(num_lines=100, seed=123)
        modified, gt = injector.inject(logs, AnomalyType.ERROR_SPIKE, intensity=0.6, seed=123)

        assert gt["anomaly_type"] == "error_spike"
        assert "component" in gt
        assert "start_time" in gt
        print(f"  ✓ Injected {gt['anomaly_type']} anomaly in component {gt['component']}")
        return True
    except Exception as e:
        print(f"  ✗ Anomaly injection failed: {e}")
        traceback.print_exc()
        return False


def test_grader():
    """Test grading system."""
    print("Testing grader...")
    try:
        from grader import InvestigationGrader
        from models import SubmitAnswer, AnomalyType

        grader = InvestigationGrader()
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
            "episode_id": "test",
            "task_id": "test",
            "difficulty": "easy",
        }
        result = grader.grade(prediction, ground_truth, steps_used=5)
        assert result.reward > 0.9
        print(f"  ✓ Grading works (score: {result.reward:.4f})")
        return True
    except Exception as e:
        print(f"  ✗ Grading failed: {e}")
        traceback.print_exc()
        return False


def test_environment():
    """Test the main environment."""
    print("Testing environment...")
    try:
        from server.log_anomaly_environment import LogAnomalyEnvironment

        env = LogAnomalyEnvironment()

        # Test reset
        obs = env.reset(difficulty="easy", seed=42)
        assert obs.steps_remaining == 15
        assert obs.answer_submitted is False
        print(f"  ✓ Environment reset successful")

        # Test step
        from models import InvestigationAction, BashCommand
        action = InvestigationAction(
            action_type="bash",
            bash_command=BashCommand(command="head -5 log.txt")
        )
        obs = env.step(action)
        assert obs.steps_remaining == 14
        assert len(obs.command_history) == 1
        print(f"  ✓ Environment step successful")

        # Test submit
        gt = env.state.ground_truth
        action = InvestigationAction(
            action_type="submit",
            answer=SubmitAnswer(
                anomaly_type=AnomalyType(gt["anomaly_type"]),
                component=gt["component"],
                start_time=gt["start_time"],
                end_time=gt["end_time"],
            )
        )
        obs = env.step(action)
        assert obs.answer_submitted is True
        assert obs.episode_reward > 0
        print(f"  ✓ Environment submit successful (reward: {obs.episode_reward:.4f})")

        env.close()
        return True
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Log Anomaly Investigation Environment - Quick Test")
    print("=" * 50)
    print()

    results = [
        ("Imports", test_imports()),
        ("Log Generation", test_log_generation()),
        ("Anomaly Injection", test_anomaly_injection()),
        ("Grader", test_grader()),
        ("Environment", test_environment()),
    ]

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Environment is ready.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
