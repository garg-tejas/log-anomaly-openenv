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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_log: List[Dict[str, Any]] = []
        self.current_episode_log: List[Dict[str, Any]] = []

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

        # Analyze what happened
        parsing_info = {
            "action_type": action.action_type,
            "used_fallback": False,
            "fallback_reason": None,
            "extracted_command": None,
            "extracted_submit": None,
        }

        if action.action_type == "bash":
            if action.bash_command:
                parsing_info["extracted_command"] = action.bash_command.command

                # Check if this is a fallback command
                fallback_commands = [
                    "wc -l log.txt && head -10 log.txt",
                    "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
                    "grep -iE 'cascade|dependency|affected",
                    "grep -iE 'latency|timeout|[0-9]+ms",
                    "grep -iE 'memory|heap|gc|[0-9]+mb",
                ]

                for fallback_cmd in fallback_commands:
                    if fallback_cmd in action.bash_command.command:
                        parsing_info["used_fallback"] = True
                        parsing_info["fallback_reason"] = "Command matches default fallback pattern"
                        break

                # Check if model output had "Command:" prefix
                if "command:" not in thought.lower() and "submit:" not in thought.lower():
                    parsing_info["used_fallback"] = True
                    parsing_info["fallback_reason"] = "No 'Command:' or 'Submit:' prefix found"

        elif action.action_type == "submit":
            if action.answer:
                parsing_info["extracted_submit"] = {
                    "anomaly_type": action.answer.anomaly_type.value,
                    "component": action.answer.component,
                    "start_time": action.answer.start_time,
                    "end_time": action.answer.end_time,
                    "confidence": action.answer.confidence,
                }

                # Check if this looks like a genuine submission
                if not action.answer.start_time or not action.answer.end_time:
                    parsing_info["used_fallback"] = True
                    parsing_info["fallback_reason"] = "Missing timestamps in submission"

                if action.answer.component == "unknown":
                    parsing_info["used_fallback"] = True
                    parsing_info["fallback_reason"] = "Component is 'unknown'"

        # Add parsing info to step log
        step_log["parsing"] = parsing_info

        return action

    def start_episode(self, episode_id: str, difficulty: str):
        """Start tracking a new episode."""
        self.current_episode_log = []
        self.current_episode_log.append(
            {
                "episode_id": episode_id,
                "difficulty": difficulty,
                "started_at": datetime.now().isoformat(),
            }
        )

    def end_episode(self, result: Dict[str, Any]):
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


def print_debug_report(analysis: Dict[str, Any]):
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


def main():
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

    # Setup
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")

    print(f"🔍 Debug Model Responses")
    print(f"Model: {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes: {args.episodes}")
    print(f"API Base URL: {base_url}")
    print(f"Output: {args.output}\n")

    # Create debug agent
    agent = DebugReactAgent(base_url=base_url, api_key=api_key, model=args.model)

    # Import environment client
    from client import LogAnomalyEnv

    env = LogAnomalyEnv(base_url="http://localhost:8000")

    # Run episodes
    for ep in range(args.episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'=' * 60}")

        # Reset environment
        result = env.reset(difficulty=args.difficulty, mode="eval")
        obs_dict = result["observation"]
        episode_id = result["episode_id"]

        # Convert to InvestigationObservation
        from models import InvestigationObservation, DifficultyLevel, EnvironmentMode

        obs = InvestigationObservation(
            command_output=obs_dict.get("command_output", ""),
            command_history=[],
            steps_remaining=obs_dict.get("steps_remaining", 15),
            total_steps=obs_dict.get("total_steps", 15),
            answer_submitted=obs_dict.get("answer_submitted", False),
            task_difficulty=DifficultyLevel(obs_dict.get("task_difficulty", "easy")),
            episode_reward=0.0,
            mode=EnvironmentMode.EVAL,
        )

        # Start episode tracking
        agent.start_episode(episode_id, args.difficulty)

        # Get state
        state_result = env.state()
        state = InvestigationState(
            episode_id=state_result.get("episode_id", ""),
            step_count=state_result.get("step_count", 0),
            log_file_path=state_result.get("log_file_path"),
        )

        # Run episode
        step = 0
        done = False
        max_steps = obs.total_steps

        while not done and step < max_steps:
            step += 1

            # Agent thinks and acts
            thought = agent.think(obs, state)
            action = agent.parse_action(thought, obs)

            print(f"  Step {step}: ", end="")
            if action.action_type == "bash" and action.bash_command:
                print(f"Executing: {action.bash_command.command}")
            elif action.action_type == "submit":
                print(f"Submitting answer")

            # Convert to LogAction and execute
            log_action = action.to_log_action()
            result = env.step(log_action)

            obs_dict = result["observation"]
            done = result.get("done", False)

            # Update observation
            obs = InvestigationObservation(
                command_output=obs_dict.get("command_output", ""),
                command_history=obs_dict.get("metadata", {}).get("command_history", []),
                steps_remaining=obs_dict.get("steps_remaining", 0),
                total_steps=obs_dict.get("total_steps", 15),
                answer_submitted=obs_dict.get("answer_submitted", False),
                task_difficulty=DifficultyLevel(obs_dict.get("task_difficulty", "easy")),
                episode_reward=result.get("reward", 0.0),
                mode=EnvironmentMode.EVAL,
            )

            state.step_count = step

        # Get final grading
        grade_result = env.grade(episode_id)

        print(f"  Result: reward={grade_result.get('reward', 0.0):.4f}")

        # End episode tracking
        agent.end_episode(grade_result)

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
