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
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
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
from config import (
    MAX_STEPS,
    MIN_STEPS_BEFORE_SUBMIT,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    get_difficulty_config,
)


@dataclass
class ReactAgent:
    """
    ReAct-style agent for log anomaly investigation.

    Uses chain-of-thought reasoning between bash commands to
    investigate log anomalies systematically.
    """

    model: str = DEFAULT_MODEL
    max_steps: int = MAX_STEPS
    base_url: str = DEFAULT_BASE_URL
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
        difficulty = observation.task_difficulty.value if observation.task_difficulty else "easy"

        response = self.client.chat.completions.create(
            model=self._api_model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(difficulty),
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

    def _get_system_prompt(self, difficulty: str = "easy") -> str:
        """Get the system prompt for the agent."""
        # Different anomaly types are valid for different difficulties
        if difficulty == "easy":
            anomaly_hint = (
                "ANOMALY TYPE: error_spike (this is the ONLY valid type for easy difficulty)"
            )
        elif difficulty == "hard":
            anomaly_hint = (
                "ANOMALY TYPE: cascade_failure (this is the ONLY valid type for hard difficulty)"
            )
        else:  # medium
            anomaly_hint = (
                "ANOMALY TYPES: error_spike, memory_leak, latency_degradation, service_dropout"
            )

        return f"""You are a DevOps engineer investigating log.txt for anomalies.

{anomaly_hint}

To run a command, reply with ONLY a bash code block:
```bash
grep ERROR log.txt | head -10
```

To submit your answer, reply with ONLY a json code block:
```json
{"anomaly_type": "TYPE_HERE", "component": "SERVICE_HERE", "start_time": "TIMESTAMP_FROM_LOGS", "end_time": "TIMESTAMP_FROM_LOGS"}
```

RULES:
1. Output exactly ONE code block per response
2. No text before or after the code block
3. start_time and end_time MUST be real timestamps copied from grep output (format: 2026-03-27T14:55:00)
4. Investigate at least 3 commands before submitting"""

    def _build_thinking_prompt(
        self,
        observation: InvestigationObservation,
        state: InvestigationState,
    ) -> str:
        """Build the prompt for the current step."""
        prompt_parts = [
            f"Steps: {observation.total_steps - observation.steps_remaining}/{observation.total_steps}",
        ]

        if observation.command_history:
            prompt_parts.append("\n=== HISTORY ===")
            for cmd_entry in observation.command_history[-5:]:
                prompt_parts.append(f"$ {cmd_entry['command']}")
                output = cmd_entry["output"]
                if len(output) > 400:
                    output = output[:400] + "\n[truncated]"
                prompt_parts.append(output)
            prompt_parts.append("")

        if observation.command_output and not observation.command_history:
            prompt_parts.append("=== OUTPUT ===")
            prompt_parts.append(observation.command_output[:800])

        prompt_parts.append("\nReply with one code block:")

        return "\n".join(prompt_parts)

    def parse_action(
        self, thought: str, observation: InvestigationObservation
    ) -> InvestigationAction:
        """
        Parse the agent's thought into an action.

        Simplified parser that ONLY looks for fenced code blocks:
        - ```bash ... ``` -> bash command
        - ```json ... ``` -> submit answer
        - ```  ... ```    -> try to detect type from content

        Args:
            thought: The agent's reasoning text
            observation: Current observation (for fallback command selection)

        Returns:
            InvestigationAction to execute
        """
        # Track whether we used fallback (for debugging)
        self._last_parse_used_fallback = False

        # Handle Qwen thinking mode - extract content after </think>
        if "</think>" in thought:
            thought = thought.split("</think>")[-1].strip()

        # Normalize line endings and whitespace
        thought = thought.replace("\r\n", "\n").replace("\r", "\n")

        # === STEP 1: Look for fenced code blocks ===
        # Match ```bash, ```json, or plain ``` code blocks
        # More flexible: optional whitespace after language, handles edge cases
        code_block_pattern = r"```(\w*)\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, thought, re.DOTALL | re.IGNORECASE)

        for lang, content in matches:
            content = content.strip()
            if not content:
                continue

            lang = lang.lower()

            # JSON block = submit
            if lang == "json" or content.startswith("{"):
                answer_data = self._extract_submit_json(content)
                if answer_data:
                    try:
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
                    except (ValueError, KeyError):
                        pass

            # Bash block or unlabeled block = command
            if lang in ("bash", "sh", "") and not content.startswith("{"):
                # Take first line as command (in case model adds comments)
                command = content.split("\n")[0].strip()
                if command:
                    return InvestigationAction(
                        action_type="bash",
                        bash_command=BashCommand(command=command),
                    )

        # === STEP 2: Fallback - use default investigation commands ===
        self._last_parse_used_fallback = True
        return self._get_fallback_action(observation)

    def _get_fallback_action(self, observation: InvestigationObservation) -> InvestigationAction:
        """Get a fallback investigation command based on current step."""
        recent_commands = [
            entry.get("command", "") for entry in (observation.command_history or [])[-3:]
        ]

        step = len(observation.command_history or [])
        default_commands = [
            # Step 0: Get error counts by component
            "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
            # Step 1: Check for CASCADE patterns
            "grep -iE 'cascade|dependency|affected by|circuit breaker' log.txt | head -15",
            # Step 2: Check for LATENCY patterns
            "grep -iE 'latency|timeout|[0-9]+ms|slow' log.txt | head -15",
            # Step 3: Check for MEMORY patterns
            "grep -iE 'memory|heap|gc|[0-9]+mb' log.txt | head -15",
            # Step 4: Get ERROR timestamps
            "grep ERROR log.txt | head -5 && echo '---' && grep ERROR log.txt | tail -5",
            # Step 5: Full error details
            "grep -E 'ERROR|FATAL' log.txt | head -20",
        ]

        # Find next command that wasn't recently used
        for cmd in default_commands[step:] + default_commands[:step]:
            if cmd not in recent_commands:
                return InvestigationAction(
                    action_type="bash",
                    bash_command=BashCommand(command=cmd),
                )

        return InvestigationAction(
            action_type="bash",
            bash_command=BashCommand(command=default_commands[0]),
        )

    def _extract_submit_json(self, thought: str) -> Optional[Dict[str, Any]]:
        """Extract submit answer JSON from model output."""
        # Pattern 1: Nested format {"action_type": "submit", "answer": {...}}
        nested_match = re.search(r'"answer"\s*:\s*(\{[^}]+\})', thought)
        if nested_match:
            try:
                return json.loads(nested_match.group(1))
            except json.JSONDecodeError:
                pass

        # Pattern 2: Flat format {"anomaly_type": "...", "component": "...", ...}
        flat_match = re.search(r'\{[^}]*"anomaly_type"\s*:\s*"[^"]+[^}]*\}', thought)
        if flat_match:
            try:
                return json.loads(flat_match.group(0))
            except json.JSONDecodeError:
                pass

        # Pattern 3: Any JSON object
        json_match = re.search(r"\{[^}]+\}", thought)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                # Check if it has required fields
                if "anomaly_type" in data or "answer" in data:
                    return data.get("answer", data)
            except json.JSONDecodeError:
                pass

        return None


def _extract_timestamps(output: str) -> List[str]:
    """Extract ISO-like timestamps from command output."""
    return re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?", output)


def _extract_error_timestamps(output: str) -> List[str]:
    """Extract timestamps specifically from ERROR/WARN lines."""
    error_timestamps = []
    for line in output.split("\n"):
        if "ERROR" in line or "WARN" in line:
            ts_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?", line)
            if ts_match:
                error_timestamps.append(ts_match.group(0))
    return error_timestamps


def _extract_component_from_errors(output: str) -> Optional[str]:
    """Extract most common component from ERROR/WARN lines."""
    component_counts: Dict[str, int] = {}
    for line in output.split("\n"):
        if "ERROR" in line or "WARN" in line or "FATAL" in line:
            # Component is typically the 3rd field in our log format
            comp_match = re.search(r"(service_[a-d])", line.lower())
            if comp_match:
                comp = comp_match.group(1)
                component_counts[comp] = component_counts.get(comp, 0) + 1

    if component_counts:
        return max(component_counts, key=lambda k: component_counts[k])
    return None


def _detect_anomaly_type(output: str) -> AnomalyType:
    """Detect anomaly type from command output patterns."""
    output_lower = output.lower()

    # Check for cascade failure patterns (highest priority for hard tasks)
    # These match the exact phrases generated by log_utils._generate_cascade_error()
    cascade_keywords = [
        "cascaded failure",
        "cascade",
        "dependency failure",
        "affected by",
        "waiting for",
        "circuit breaker",
        "degraded mode",
        "initial failure",
        "due to",
        "upstream",
    ]
    cascade_count = sum(1 for kw in cascade_keywords if kw in output_lower)
    if cascade_count >= 2:  # Need multiple signals for cascade
        return AnomalyType.CASCADE_FAILURE

    # Check for memory leak patterns
    memory_keywords = ["heap", "gc", "memory", "outofmemory", "mb"]
    memory_pattern = re.search(r"(\d{3,4})\s*mb", output_lower)
    if memory_pattern or sum(1 for kw in memory_keywords if kw in output_lower) >= 2:
        return AnomalyType.MEMORY_LEAK

    # Check for latency degradation
    latency_keywords = ["latency", "timeout", "response time", "slow"]
    latency_pattern = re.search(r"latency[:\s]+(\d{3,})", output_lower)
    if latency_pattern or sum(1 for kw in latency_keywords if kw in output_lower) >= 2:
        return AnomalyType.LATENCY_DEGRADATION

    # Check for service dropout (absence pattern - harder to detect)
    if "no lines" in output_lower or "0 matches" in output_lower:
        return AnomalyType.SERVICE_DROPOUT

    # Default to error_spike (most common in easy tasks)
    return AnomalyType.ERROR_SPIKE


def _find_error_time_window(timestamps: List[str]) -> Tuple[str, str]:
    """
    Find the time window where errors are concentrated.
    Uses the middle portion of timestamps to avoid log boundary timestamps.
    """
    if not timestamps:
        return ("", "")
    if len(timestamps) == 1:
        return (timestamps[0], timestamps[0])

    # Sort timestamps
    sorted_ts = sorted(set(timestamps))  # Deduplicate and sort

    if len(sorted_ts) <= 3:
        return (sorted_ts[0], sorted_ts[-1])

    # Use middle 60% of timestamps to avoid boundary artifacts
    # Anomalies are injected in the middle of logs, not at boundaries
    trim = len(sorted_ts) // 5  # Remove 20% from each end
    start_idx = trim
    end_idx = len(sorted_ts) - trim - 1

    return (sorted_ts[start_idx], sorted_ts[end_idx])


def _guess_submit_answer(observation: InvestigationObservation) -> SubmitAnswer:
    """Build a best-effort submission from command history."""
    history = observation.command_history or []
    combined_output = "\n".join(str(entry.get("output", "")) for entry in history)

    # Detect anomaly type from patterns in output
    anomaly_type = _detect_anomaly_type(combined_output)

    # Extract component - prioritize ERROR-associated components
    component = _extract_component_from_errors(combined_output)
    if not component:
        # Fallback: count all component mentions
        components = re.findall(r"\b(service_[a-d])\b", combined_output.lower())
        component = Counter(components).most_common(1)[0][0] if components else "service_a"

    # Extract time window - prioritize ERROR-line timestamps with clustering
    error_timestamps = _extract_error_timestamps(combined_output)
    if len(error_timestamps) >= 2:
        # Use clustered error timestamp range
        start_time, end_time = _find_error_time_window(error_timestamps)
    else:
        # Fall back to all timestamps with clustering
        all_timestamps = _extract_timestamps(combined_output)
        if len(all_timestamps) >= 2:
            start_time, end_time = _find_error_time_window(all_timestamps)
        elif all_timestamps:
            start_time = all_timestamps[0]
            end_time = all_timestamps[0]
        else:
            # Last resort - no timestamps found
            start_time = ""
            end_time = ""

    # Truncate to seconds for cleaner output
    if start_time and "." in start_time:
        start_time = start_time.split(".")[0]
    if end_time and "." in end_time:
        end_time = end_time.split(".")[0]

    return SubmitAnswer(
        anomaly_type=anomaly_type,
        component=component,
        start_time=start_time,
        end_time=end_time,
        confidence=0.5,
    )


def run_baseline_inference(
    environment: Any,
    difficulty: str = "all",
    model: str = "Qwen/Qwen3.5-2B",
    num_episodes: int = 3,
    max_steps: int = 15,
    base_url: Optional[str] = None,
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
    agent = ReactAgent(model=model, max_steps=max_steps, base_url=base_url or "")

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

                is_last_step = step == agent.max_steps - 1
                if is_last_step and not observation.answer_submitted:
                    # Force a final submission to avoid null episodes
                    forced_answer = _guess_submit_answer(observation)
                    action = InvestigationAction(action_type="submit", answer=forced_answer)
                    print("Submitting answer (forced fallback)")
                else:
                    # Agent thinks
                    thought = agent.think(observation, state)

                    # Parse action (pass observation for fallback)
                    action = agent.parse_action(thought, observation)

                    # Fix 4: Minimum steps before allowing submit
                    # Ensure we gather enough evidence before submitting
                    if action.action_type == "submit" and step < MIN_STEPS_BEFORE_SUBMIT:
                        # Force more investigation - use fallback command
                        print(f"(deferring submit, need more evidence) ", end="")
                        fallback_commands = [
                            "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
                            "grep ERROR log.txt | head -10 && echo '---' && grep ERROR log.txt | tail -10",
                            "grep -iE 'cascade|dependency|affected|latency|memory|heap' log.txt | head -15",
                        ]
                        cmd = fallback_commands[min(step, len(fallback_commands) - 1)]
                        action = InvestigationAction(
                            action_type="bash",
                            bash_command=BashCommand(command=cmd),
                        )

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
