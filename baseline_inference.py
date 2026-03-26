"""
Baseline Inference Script for Log Anomaly Investigation.

This script provides baseline performance using ReAct prompting with Qwen
via HuggingFace router.

Usage:
    # Run from command line
    python baseline_inference.py --difficulty easy --episodes 5

    # Or import and use programmatically
    from baseline_inference import run_baseline_inference, ReactAgent

    results = run_baseline_inference(
        environment=env,
        difficulty="all",
        num_episodes=3,
    )
"""

import os
import json
import argparse
import re
from typing import Any, Dict, List
from dataclasses import dataclass, field

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars only

from openai import OpenAI

from models import (
    InvestigationAction,
    InvestigationObservation,
    InvestigationState,
    DifficultyLevel,
    AnomalyType,
    SubmitAnswer,
    BashCommand,
)
from grader import InvestigationGrader, calculate_summary_stats


@dataclass
class ReactAgent:
    """
    ReAct-style agent for log anomaly investigation.

    Uses chain-of-thought reasoning between bash commands to
    investigate log anomalies systematically.
    """

    model: str = "Qwen/Qwen3.5-2B"
    max_steps: int = 15
    base_url: str = ""  # Empty means auto-detect
    client: OpenAI = field(init=False)
    _api_model: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the OpenAI-compatible client."""
        # Determine base URL and API key
        # Priority: explicit base_url > OPENAI_BASE_URL env > HuggingFace router
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL", "")

        if base_url:
            # Using custom endpoint (local LLM, vLLM, Ollama, etc.)
            api_key = os.environ.get("OPENAI_API_KEY", "local")
            self._api_model = self.model
        else:
            # Default to HuggingFace router
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key found. Set HF_TOKEN for HuggingFace or OPENAI_API_KEY for other providers."
                )
            base_url = "https://router.huggingface.co/v1"
            # Use :novita suffix for HuggingFace router
            self._api_model = f"{self.model}:novita" if ":novita" not in self.model else self.model

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def think(self, observation: InvestigationObservation, state: InvestigationState) -> str:
        """
        Generate reasoning about the current observation.

        Args:
            observation: Current environment observation
            state: Current environment state

        Returns:
            Reasoning about what to do next
        """
        prompt = self._build_thinking_prompt(observation, state)

        response = self.client.chat.completions.create(
            model=self._api_model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        content = response.choices[0].message.content
        if content is None:
            return "Error: No response from model"
        return content

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are an expert DevOps engineer investigating system logs for anomalies.
You have access to a sandboxed bash environment with the following constraints:

AVAILABLE COMMANDS:
- grep, egrep, fgrep: Pattern matching
- awk, sed: Text processing
- sort, uniq, wc: Counting/aggregation
- head, tail, cut: Line selection
- cat, less: File display
- find, xargs: File search
- date, echo, ls, pwd: Utilities

TASK:
Investigate the log file (log.txt) to identify anomalies. For each step:
1. Analyze the command output from previous steps
2. Decide on the next command to run OR submit your final answer

SUBMIT FORMAT (when you have enough evidence):
{
    "action_type": "submit",
    "answer": {
        "anomaly_type": "error_spike|memory_leak|service_dropout|cascade_failure|latency_degradation",
        "component": "component_name",
        "start_time": "ISO timestamp",
        "end_time": "ISO timestamp"
    }
}

INVESTIGATION STRATEGY:
1. Start with broad commands to understand the log structure
2. Look for patterns: error rates, severity distribution, timing anomalies
3. Narrow down to specific components and time windows
4. Correlate findings across multiple dimensions
5. Submit when you have high confidence

Be thorough but efficient. You have limited steps."""

    def _build_thinking_prompt(
        self,
        observation: InvestigationObservation,
        state: InvestigationState,
    ) -> str:
        """Build the prompt for the current step."""
        prompt_parts = [
            f"Episode: {state.episode_id}",
            f"Steps remaining: {observation.steps_remaining}/{observation.total_steps}",
            f"Task difficulty: {observation.task_difficulty.value}",
            f"Answer submitted: {observation.answer_submitted}",
            "",
        ]

        if observation.command_history:
            prompt_parts.append("=== COMMAND HISTORY ===")
            for i, cmd_entry in enumerate(observation.command_history[-5:], 1):
                prompt_parts.append(f"[{i}] Command: {cmd_entry['command']}")
                if cmd_entry.get("error"):
                    prompt_parts.append(f"    Error: {cmd_entry['output']}")
                else:
                    output = cmd_entry["output"]
                    if len(output) > 500:
                        output = (
                            output[:500]
                            + f"\n... [truncated, {len(cmd_entry['output'])} total chars]"
                        )
                    prompt_parts.append(f"    Output:\n{output}")
            prompt_parts.append("")

        if observation.command_output:
            prompt_parts.append("=== LAST OUTPUT ===")
            prompt_parts.append(observation.command_output[:1000])

        prompt_parts.extend(
            [
                "",
                "What is your next action? Provide your reasoning and the command to execute.",
                "If you have identified the anomaly with sufficient confidence, submit your answer.",
            ]
        )

        return "\n".join(prompt_parts)

    def parse_action(
        self, thought: str, observation: InvestigationObservation
    ) -> InvestigationAction:
        """
        Parse the agent's thought into an action.

        Args:
            thought: The agent's reasoning text
            observation: Current observation (for fallback command selection)

        Returns:
            InvestigationAction to execute
        """
        # Check if submitting
        if "submit" in thought.lower() or '{"action_type": "submit"' in thought:
            # Try to extract answer from JSON
            json_match = re.search(r"\{[^}]+\}", thought, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    answer_data = data.get("answer", {})
                    return InvestigationAction(
                        action_type="submit",
                        answer=SubmitAnswer(
                            anomaly_type=AnomalyType(
                                answer_data.get("anomaly_type", "error_spike")
                            ),
                            component=answer_data.get("component", "unknown"),
                            start_time=answer_data.get("start_time", ""),
                            end_time=answer_data.get("end_time", ""),
                        ),
                    )
                except (json.JSONDecodeError, ValueError, KeyError):
                    pass

        # Extract bash command
        command_match = re.search(r"(?:Command|command):\s*([^\n]+)", thought)
        if command_match:
            command = command_match.group(1).strip()
            if command:
                return InvestigationAction(
                    action_type="bash",
                    bash_command=BashCommand(command=command),
                )

        # Default to viewing more log content
        step = len(observation.command_history)
        commands = [
            "cat log.txt | head -50",
            "grep ERROR log.txt | head -20",
            "grep WARN log.txt | head -20",
            "awk '{print $3}' log.txt | sort | uniq -c | sort -rn | head -10",
            "grep -i 'memory\\|heap\\|gc' log.txt | head -20",
        ]
        command = commands[min(step, len(commands) - 1)]

        return InvestigationAction(
            action_type="bash",
            bash_command=BashCommand(command=command),
        )


def run_baseline_inference(
    environment: Any,
    difficulty: str = "all",
    model: str = "Qwen/Qwen3.5-2B",
    num_episodes: int = 3,
    max_steps: int = 15,
    base_url: str = "",
) -> Dict[str, Any]:
    """
    Run baseline inference on the environment.

    Args:
        environment: LogAnomalyEnvironment instance
        difficulty: Difficulty level ("easy", "medium", "hard", or "all")
        model: Model to use for baseline
        num_episodes: Number of episodes per difficulty
        max_steps: Maximum steps per episode
        base_url: OpenAI-compatible API base URL (empty for auto-detect)

    Returns:
        Baseline results dictionary
    """
    # Check for API key (either HF_TOKEN or OPENAI_API_KEY)
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    has_base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    if not api_key and not has_base_url:
        return {
            "status": "error",
            "message": "No API key found. Set HF_TOKEN, OPENAI_API_KEY, or use --base-url for local LLMs",
            "baseline_scores": {},
        }

    # Create agent
    agent = ReactAgent(model=model, max_steps=max_steps, base_url=base_url)

    # Determine difficulties to test
    if difficulty == "all":
        difficulties = [d.value for d in DifficultyLevel]
    else:
        difficulties = [difficulty]

    all_results: List[Any] = []

    for diff in difficulties:
        print(f"\n{'=' * 50}")
        print(f"Testing difficulty: {diff.upper()}")
        print(f"{'=' * 50}")

        for episode_num in range(num_episodes):
            print(f"\nEpisode {episode_num + 1}/{num_episodes}")

            # Reset environment
            obs_result = environment.reset(difficulty=diff, seed=episode_num)
            # Handle both dict and object returns
            if isinstance(obs_result, dict):
                observation = _dict_to_observation(obs_result.get("observation", obs_result))
            elif isinstance(obs_result, InvestigationObservation):
                observation = obs_result
            else:
                # Try to convert from model
                observation = _dict_to_observation(
                    obs_result.model_dump() if hasattr(obs_result, "model_dump") else obs_result
                )
            state = environment.state

            # Run agent
            for step in range(agent.max_steps):
                if observation.answer_submitted:
                    break

                print(f"  Step {step + 1}: ", end="")

                # Agent thinks
                thought = agent.think(observation, state)

                # Parse action (pass observation for fallback)
                action = agent.parse_action(thought, observation)

                if action.action_type == "bash" and action.bash_command:
                    print(f"Executing: {action.bash_command.command}")
                else:
                    print("Submitting answer")

                # Execute action
                result = environment.step(action)
                # Handle both dict and object returns
                if isinstance(result, dict):
                    observation = _dict_to_observation(result.get("observation", result))
                elif isinstance(result, InvestigationObservation):
                    observation = result
                else:
                    observation = _dict_to_observation(
                        result.model_dump() if hasattr(result, "model_dump") else result
                    )
                state = environment.state

            # Grade episode
            episode_result = environment.get_result()
            all_results.append(episode_result)

            print(f"  Result: reward={episode_result.reward:.4f}")

    # Calculate summary
    summary = calculate_summary_stats(all_results)

    return {
        "status": "success",
        "model": model,
        "total_episodes": len(all_results),
        "baseline_scores": summary,
        "detailed_results": [r.model_dump() for r in all_results],
    }


def _dict_to_observation(d: Dict[str, Any]) -> InvestigationObservation:
    """Convert dictionary to InvestigationObservation."""
    return InvestigationObservation(
        command_output=d.get("command_output", ""),
        command_history=d.get("command_history", []),
        steps_remaining=d.get("steps_remaining", 15),
        total_steps=d.get("total_steps", 15),
        answer_submitted=d.get("answer_submitted", False),
        task_difficulty=DifficultyLevel(d.get("task_difficulty", "easy")),
        episode_reward=d.get("episode_reward", 0.0),
        metadata=d.get("metadata", {}),
    )


def main() -> None:
    """Command-line interface for baseline inference."""
    parser = argparse.ArgumentParser(
        description="Run baseline inference on Log Anomaly Investigation Environment"
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level to test",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen3.5-2B",
        help="Model to use (default: Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=3,
        help="Number of episodes per difficulty",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--api-key",
        help="API key (or set HF_TOKEN/OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL (e.g., http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum steps per episode (default: 15)",
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Set base URL if provided
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    # Import environment
    from server.log_anomaly_environment import LogAnomalyEnvironment

    # Create environment
    env = LogAnomalyEnvironment()

    # Run baseline
    print(f"Running baseline with {args.model}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes per difficulty: {args.episodes}")
    print()

    results = run_baseline_inference(
        environment=env,
        difficulty=args.difficulty,
        model=args.model,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )

    # Print results
    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)
    print(json.dumps(results, indent=2))

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
