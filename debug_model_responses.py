#!/usr/bin/env python3
"""
Debug script to capture and analyze raw model responses.

This script runs a small set of episodes and logs:
- Raw model output (before parsing)
- Parsed action (what we extracted)
- Fallback triggers (when/why fallback was used)
- Command history at each step

Usage:
    python debug_model_responses.py --model Qwen/Qwen3.5-4B --episodes 3 --difficulty easy
    python debug_model_responses.py --model Qwen/Qwen3.5-4B --episodes 2 --difficulty hard --output debug_hard.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_inference import ReactAgent
from models import (
    LogAction,
    InvestigationAction,
    InvestigationObservation,
    InvestigationState,
    DifficultyLevel,
    AnomalyType,
)


class DebugReactAgent(ReactAgent):
    """
    Extended ReactAgent that captures raw model responses for debugging.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.debug_log: List[
            List[Dict[str, Any]]
        ] = []  # List of episodes (each episode is a list of step dicts)
        self.current_episode_log: List[Dict[str, Any]] = []  # Current episode's step dicts

    def think(self, observation: InvestigationObservation, state: InvestigationState) -> str:
        """Override to capture raw model output."""
        # Call parent to get raw model response
        raw_response = super().think(observation, state)

        # Log the raw response
        self.current_episode_log.append(
            {
                "step": len(observation.command_history or []) + 1,
                "raw_model_output": raw_response,
                "prompt_context": {
                    "steps_remaining": observation.steps_remaining,
                    "command_history_length": len(observation.command_history or []),
                    "last_command": (
                        observation.command_history[-1].get("command", "")
                        if observation.command_history
                        else None
                    ),
                    "last_output_length": len(observation.command_output),
                    "last_output_preview": observation.command_output[:200],
                },
            }
        )

        return raw_response

    def parse_action(
        self, thought: str, observation: InvestigationObservation
    ) -> InvestigationAction:
        """Override to capture parsing details."""
        # Get the step log entry we just created
        if self.current_episode_log:
            step_log = self.current_episode_log[-1]
        else:
            step_log = {}

        # Try to parse
        action = super().parse_action(thought, observation)

        # Check if fallback was used (set by parent's parse_action)
        used_fallback = getattr(self, "_last_parse_used_fallback", False)

        # Build parsing info
        parsing_info = {
            "action_type": action.action_type,
            "used_fallback": used_fallback,
            "fallback_reason": "No code block found" if used_fallback else None,
            "extracted_command": None,
            "extracted_submit": None,
        }

        if action.action_type == "bash" and action.bash_command:
            parsing_info["extracted_command"] = action.bash_command.command

        elif action.action_type == "submit" and action.answer:
            parsing_info["extracted_submit"] = {
                "anomaly_type": action.answer.anomaly_type.value,
                "component": action.answer.component,
                "start_time": action.answer.start_time,
                "end_time": action.answer.end_time,
                "confidence": action.answer.confidence,
            }

        # Add parsing info to step log
        step_log["parsing"] = parsing_info

        return action

    def start_episode(self, episode_id: str, difficulty: str) -> None:
        """Start tracking a new episode."""
        self.current_episode_log = []
        self.current_episode_log.append(
            {
                "episode_id": episode_id,
                "difficulty": difficulty,
                "started_at": datetime.now().isoformat(),
            }
        )

    def end_episode(self, result: Dict[str, Any]) -> None:
        """Finish episode and save debug log."""
        self.current_episode_log.append(
            {
                "episode_complete": True,
                "final_reward": result.get("reward", 0.0),
                "component_score": result.get("component_score", 0.0),
                "type_score": result.get("type_score", 0.0),
                "window_score": result.get("window_score", 0.0),
                "efficiency_score": result.get("efficiency_score", 0.0),
                "steps_used": result.get("steps_used", 0),
            }
        )

        # Save to main debug log
        self.debug_log.append(self.current_episode_log.copy())
        self.current_episode_log = []


def analyze_debug_log(debug_log: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze collected debug logs to identify patterns.
    """
    total_steps = 0
    fallback_steps = 0
    parsing_failures = []
    successful_parses = []

    for episode_log in debug_log:
        for entry in episode_log:
            if "raw_model_output" in entry:
                total_steps += 1

                parsing = entry.get("parsing", {})
                if parsing.get("used_fallback"):
                    fallback_steps += 1
                    parsing_failures.append(
                        {
                            "episode": episode_log[0].get("episode_id", "unknown"),
                            "step": entry.get("step", 0),
                            "difficulty": episode_log[0].get("difficulty", "unknown"),
                            "raw_output": entry["raw_model_output"],
                            "fallback_reason": parsing.get("fallback_reason"),
                            "context": entry.get("prompt_context", {}),
                        }
                    )
                else:
                    successful_parses.append(
                        {
                            "episode": episode_log[0].get("episode_id", "unknown"),
                            "step": entry.get("step", 0),
                            "difficulty": episode_log[0].get("difficulty", "unknown"),
                            "raw_output": entry["raw_model_output"],
                            "action_type": parsing.get("action_type"),
                            "extracted": parsing.get("extracted_command")
                            or parsing.get("extracted_submit"),
                        }
                    )

    analysis = {
        "summary": {
            "total_steps": total_steps,
            "fallback_steps": fallback_steps,
            "successful_parses": total_steps - fallback_steps,
            "fallback_rate": (
                round(fallback_steps / total_steps * 100, 2) if total_steps > 0 else 0
            ),
        },
        "parsing_failures": parsing_failures,
        "successful_parses": successful_parses[:10],  # First 10 examples
        "common_failure_reasons": {},
    }

    # Count failure reasons
    for failure in parsing_failures:
        reason = failure.get("fallback_reason", "unknown")
        analysis["common_failure_reasons"][reason] = (
            analysis["common_failure_reasons"].get(reason, 0) + 1
        )

    return analysis


def print_debug_report(analysis: Dict[str, Any]) -> None:
    """
    Print a human-readable debug report.
    """
    print("\n" + "=" * 80)
    print("MODEL RESPONSE DEBUG REPORT")
    print("=" * 80)

    summary = analysis["summary"]
    print(f"\n📊 SUMMARY:")
    print(f"  Total steps executed: {summary['total_steps']}")
    print(f"  Successful parses: {summary['successful_parses']}")
    print(f"  Fallback parses: {summary['fallback_steps']}")
    print(f"  Fallback rate: {summary['fallback_rate']}%")

    print(f"\n🚨 COMMON FAILURE REASONS:")
    for reason, count in sorted(
        analysis["common_failure_reasons"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  - {reason}: {count} times")

    print(f"\n❌ PARSING FAILURES (First 5):")
    for i, failure in enumerate(analysis["parsing_failures"][:5], 1):
        print(f"\n  --- Failure #{i} ---")
        print(f"  Episode: {failure['episode']}")
        print(f"  Difficulty: {failure['difficulty']}")
        print(f"  Step: {failure['step']}")
        print(f"  Reason: {failure['fallback_reason']}")
        print(f"  Steps remaining: {failure['context'].get('steps_remaining', 'N/A')}")
        print(f"\n  Raw model output:")
        print(f"  {failure['raw_output'][:300]}")
        if len(failure["raw_output"]) > 300:
            print(f"  ... [truncated, total {len(failure['raw_output'])} chars]")

    print(f"\n✅ SUCCESSFUL PARSES (First 3):")
    for i, success in enumerate(analysis["successful_parses"][:3], 1):
        print(f"\n  --- Success #{i} ---")
        print(f"  Episode: {success['episode']}")
        print(f"  Difficulty: {success['difficulty']}")
        print(f"  Step: {success['step']}")
        print(f"  Action type: {success['action_type']}")
        print(f"\n  Raw model output:")
        print(f"  {success['raw_output'][:300]}")
        if len(success["raw_output"]) > 300:
            print(f"  ... [truncated]")
        print(f"\n  Extracted:")
        print(f"  {success['extracted']}")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug model responses during baseline inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Model to use for inference",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Difficulty level to test",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="debug_model_responses.json",
        help="Output file for debug logs",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL",
    )

    args = parser.parse_args()

    # Setup - ReactAgent reads API key from environment, not constructor
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")

    # Ensure OPENAI_API_KEY is set in environment if not already
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "dummy"

    print(f"🔍 Debug Model Responses")
    print(f"Model: {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes: {args.episodes}")
    print(f"API Base URL: {base_url}")
    print(f"Output: {args.output}\n")

    # Create debug agent (api_key read from environment by ReactAgent.__post_init__)
    agent = DebugReactAgent(base_url=base_url, model=args.model)

    # Import environment (direct environment, not client)
    from server.log_anomaly_environment import LogAnomalyEnvironment
    from models import EnvironmentMode, DifficultyLevel

    # Helper function to convert LogState to InvestigationState
    def log_state_to_investigation(log_state):
        return InvestigationState(
            episode_id=log_state.episode_id or "",
            step_count=log_state.step_count,
            log_file_path=log_state.log_file_path,
            task_id=log_state.task_id,
            mode=EnvironmentMode(log_state.mode)
            if isinstance(log_state.mode, str)
            else log_state.mode,
        )

    # Helper function to convert LogObservation to InvestigationObservation
    def log_obs_to_investigation(log_obs, log_state):
        # Extract command history from metadata
        cmd_history = []
        if hasattr(log_obs, "metadata") and log_obs.metadata:
            cmd_history = log_obs.metadata.get("command_history", [])

        return InvestigationObservation(
            command_output=log_obs.command_output,
            command_history=cmd_history,
            steps_remaining=log_obs.steps_remaining,
            total_steps=log_obs.total_steps,
            answer_submitted=log_obs.answer_submitted,
            task_difficulty=DifficultyLevel(log_obs.task_difficulty)
            if isinstance(log_obs.task_difficulty, str)
            else log_obs.task_difficulty,
            episode_reward=log_obs.reward,
            mode=EnvironmentMode(log_state.mode)
            if isinstance(log_state.mode, str)
            else log_state.mode,
        )

    env = LogAnomalyEnvironment()

    # Run episodes
    for ep in range(args.episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'=' * 60}")

        # Reset environment
        log_observation = env.reset(difficulty=args.difficulty, seed=ep)
        log_state = env.state

        # Convert to Investigation types
        observation = log_obs_to_investigation(log_observation, log_state)
        state = log_state_to_investigation(log_state)

        # Start episode tracking
        agent.start_episode(state.episode_id, args.difficulty)

        # Run episode
        step = 0
        max_steps = 15

        while not observation.answer_submitted and step < max_steps:
            step += 1

            # Agent thinks and acts
            thought = agent.think(observation, state)
            action = agent.parse_action(thought, observation)

            print(f"  Step {step}: ", end="")
            if action.action_type == "bash" and action.bash_command:
                print(f"Executing: {action.bash_command.command}")
            elif action.action_type == "submit":
                print(f"Submitting answer")

            # Execute action
            log_action = action.to_log_action()
            log_observation = env.step(log_action)
            log_state = env.state

            # Convert to Investigation types
            observation = log_obs_to_investigation(log_observation, log_state)
            state = log_state_to_investigation(log_state)

        # Get final result
        episode_result = env.get_result()

        print(f"  Result: reward={episode_result.reward:.4f}")

        # End episode tracking - convert to dict
        result_dict = {
            "reward": episode_result.reward,
            "component_score": episode_result.component_score,
            "type_score": episode_result.type_score,
            "window_score": episode_result.window_score,
            "efficiency_score": episode_result.efficiency_score,
            "steps_used": episode_result.steps_used,
        }
        agent.end_episode(result_dict)

    # Analyze and report
    print("\n" + "=" * 80)
    print("ANALYZING DEBUG LOGS...")
    print("=" * 80)

    analysis = analyze_debug_log(agent.debug_log)
    print_debug_report(analysis)

    # Save full debug logs
    output_data = {
        "metadata": {
            "model": args.model,
            "difficulty": args.difficulty,
            "episodes": args.episodes,
            "generated_at": datetime.now().isoformat(),
        },
        "analysis": analysis,
        "raw_debug_logs": agent.debug_log,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Full debug logs saved to: {output_path}")
    print(f"   To view raw model outputs, check the 'raw_debug_logs' section in the JSON file.")


if __name__ == "__main__":
    main()
