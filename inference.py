"""
Inference Script for Log Anomaly Investigation Environment.
===================================
MANDATORY ENVIRONMENT VARIABLES (set these before running):
    API_BASE_URL   The API endpoint for the LLM (default: HuggingFace router)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face API key

Usage:
    # Run against HuggingFace Space (default)
    uv run python inference.py --difficulty easy --episodes 3

    # Run against local server
    uv run python inference.py --local --difficulty easy --episodes 1

    # Run against custom URL
    uv run python inference.py --url http://localhost:8000 --difficulty all --episodes 3

    # Save verbose output to JSON
    uv run python inference.py -d all -e 3 -o results.json
"""

import asyncio
import json
import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import os

load_dotenv()

from models import (
    LogAction,
    LogObservation,
    DifficultyLevel,
    AnomalyType,
    EpisodeResult,
)
from grader import calculate_summary_stats
from config import (
    MAX_STEPS,
    MIN_STEPS_BEFORE_SUBMIT,
    DEFAULT_MODEL,
    HF_ROUTER_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_PRESENCE_PENALTY,
    OUTPUT_PREVIEW_SHORT,
    OUTPUT_PREVIEW_LONG,
    get_logger,
)
from client import LogAnomalyEnvClient, LocalEnvWrapper, DEFAULT_ENV_URL

# Set up logging
logger = get_logger(__name__)

# =============================================================================
# Environment Variables (as required by hackathon)
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", HF_ROUTER_URL)
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)


@dataclass
class VerboseLog:
    """Container for verbose episode logs."""

    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def add_episode(
        self,
        episode_id: str,
        difficulty: str,
        steps: List[Dict[str, Any]],
        result: Optional[EpisodeResult] = None,
    ):
        """Add an episode log."""
        self.episodes.append(
            {
                "episode_id": episode_id,
                "difficulty": difficulty,
                "timestamp": datetime.now().isoformat(),
                "steps": steps,
                "result": result.model_dump() if result else None,
            }
        )

    def save(self, path: str):
        """Save verbose logs to JSON file."""
        with open(path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_episodes": len(self.episodes),
                    "episodes": self.episodes,
                },
                f,
                indent=2,
            )


@dataclass
class ReactAgent:
    """
    ReAct-style agent for log anomaly investigation.

    Uses chain-of-thought reasoning between bash commands to
    investigate log anomalies systematically.
    """

    model: str = MODEL_NAME
    max_steps: int = MAX_STEPS
    base_url: str = API_BASE_URL
    _http_client: Optional[httpx.AsyncClient] = field(default=None, init=False)
    _api_model: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the API model name."""
        # For HuggingFace router, append :novita suffix
        if "huggingface" in self.base_url.lower():
            self._api_model = f"{self.model}:novita" if ":novita" not in self.model else self.model
        else:
            self._api_model = self.model

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0, connect=30.0),
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def think(
        self,
        command_output: str,
        command_history: List[Dict[str, Any]],
        steps_remaining: int,
        total_steps: int,
        difficulty: str,
    ) -> str:
        """
        Generate reasoning about the current observation.

        Args:
            command_output: Output from last command
            command_history: History of commands and outputs
            steps_remaining: Steps remaining in episode
            total_steps: Total steps allowed
            difficulty: Current difficulty level

        Returns:
            Reasoning about what to do next
        """
        prompt = self._build_thinking_prompt(
            command_output, command_history, steps_remaining, total_steps
        )

        client = await self._get_client()

        response = await client.post(
            "/chat/completions",
            json={
                "model": self._api_model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt(difficulty)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
                "top_p": LLM_TOP_P,
                "presence_penalty": LLM_PRESENCE_PENALTY,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": 20,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        if content is None:
            return "Error: No response from model"
        return content

    def _get_system_prompt(self, difficulty: str = "easy") -> str:
        """Get the system prompt for the agent."""
        if difficulty == "easy":
            anomaly_hint = (
                "ANOMALY TYPE: error_spike (this is the ONLY valid type for easy difficulty)"
            )
            search_hint = "Look for: ERROR, error spike patterns"
        elif difficulty == "hard":
            anomaly_hint = (
                "ANOMALY TYPE: cascade_failure (this is the ONLY valid type for hard difficulty)"
            )
            search_hint = "Look for: cascade, dependency, affected by, circuit breaker"
        else:
            anomaly_hint = (
                "ANOMALY TYPES: error_spike, memory_leak, latency_degradation, service_dropout"
            )
            search_hint = "Look for: ERROR, heap/GC/memory (memory_leak), latency/timeout (latency), unavailable/dropout (service_dropout)"

        return f"""You are a DevOps engineer investigating log.txt for anomalies.

{anomaly_hint}
{search_hint}

To run a command, reply with ONLY a bash code block:
```bash
grep -i error log.txt | head -10
```

To submit your answer, reply with ONLY a json code block:
```json
{{"anomaly_type": "TYPE_HERE", "component": "SERVICE_HERE", "start_time": "TIMESTAMP_FROM_LOGS", "end_time": "TIMESTAMP_FROM_LOGS"}}
```

RULES:
1. Output exactly ONE code block per response
2. No text before or after the code block
3. start_time and end_time MUST be real timestamps copied from grep output (format: 2026-03-27T14:55:00)
4. Investigate at least 3 commands before submitting
5. If grep returns empty, try -i flag (case insensitive) or different keywords"""

    def _build_thinking_prompt(
        self,
        command_output: str,
        command_history: List[Dict[str, Any]],
        steps_remaining: int,
        total_steps: int,
    ) -> str:
        """Build the prompt for the current step."""
        steps_used = total_steps - steps_remaining

        prompt_parts = [f"Step {steps_used}/{total_steps}"]

        if steps_remaining <= 3 and steps_remaining > 0:
            prompt_parts.append(
                f"\n!!! FINAL {steps_remaining} STEPS - Submit your answer NOW or get 0 reward !!!"
            )

        if command_history:
            unique_commands = list(
                dict.fromkeys(cmd_entry["command"] for cmd_entry in command_history)
            )
            if unique_commands:
                prompt_parts.append(
                    f"\n*** DO NOT REPEAT THESE COMMANDS (you will be penalized): ***\n"
                    + "\n".join(f"- {c}" for c in unique_commands[-7:])
                )

        if command_history:
            prompt_parts.append("\n=== RECENT OUTPUT ===")
            empty_count = 0
            for cmd_entry in command_history[-3:]:
                prompt_parts.append(f"$ {cmd_entry['command']}")
                output = cmd_entry["output"]
                if not output or not output.strip():
                    prompt_parts.append("(no output - try different keywords)")
                    empty_count += 1
                elif len(output) > OUTPUT_PREVIEW_SHORT:
                    prompt_parts.append(output[:OUTPUT_PREVIEW_SHORT] + "\n[truncated]")
                else:
                    prompt_parts.append(output)

            if empty_count >= 2:
                prompt_parts.append(
                    "\n*** Many commands returned empty. Try: grep -i (case insensitive), or check 'head -50 log.txt' for log format ***"
                )
            prompt_parts.append("")

        if command_output and not command_history:
            prompt_parts.append("=== OUTPUT ===")
            prompt_parts.append(command_output[:OUTPUT_PREVIEW_LONG])

        prompt_parts.append("\nReply with one code block:")

        return "\n".join(prompt_parts)

    def parse_action(
        self,
        thought: str,
        command_history: List[Dict[str, Any]],
    ) -> LogAction:
        """
        Parse the agent's thought into a LogAction.

        Args:
            thought: The agent's reasoning text
            command_history: History of commands for fallback selection

        Returns:
            LogAction to execute
        """
        self._last_parse_used_fallback = False

        # Handle Qwen thinking mode - extract content after </think>
        if "</think>" in thought:
            thought = thought.split("</think>")[-1].strip()

        thought = thought.replace("\r\n", "\n").replace("\r", "\n")

        # Look for fenced code blocks
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
                        return LogAction(
                            action_type="submit",
                            anomaly_type=answer_data.get("anomaly_type", "error_spike"),
                            component=answer_data.get("component", "unknown"),
                            start_time=answer_data.get("start_time", ""),
                            end_time=answer_data.get("end_time", ""),
                        )
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Failed to parse JSON submit answer: {e}")

            # Bash block = command
            if lang in ("bash", "sh", "") and not content.startswith("{"):
                command = content.split("\n")[0].strip()
                if command:
                    return LogAction(action_type="bash", command=command)

        # Fallback
        self._last_parse_used_fallback = True
        return self._get_fallback_action(command_history)

    def _get_fallback_action(self, command_history: List[Dict[str, Any]]) -> LogAction:
        """Get a fallback investigation command based on current step."""
        recent_commands = [entry.get("command", "") for entry in (command_history or [])[-3:]]
        step = len(command_history or [])

        default_commands = [
            "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
            "grep -iE 'cascade|dependency|affected by|circuit breaker' log.txt | head -15",
            "grep -iE 'latency|timeout|[0-9]+ms|slow' log.txt | head -15",
            "grep -iE 'memory|heap|gc|[0-9]+mb' log.txt | head -15",
            "grep ERROR log.txt | head -5 && echo '---' && grep ERROR log.txt | tail -5",
            "grep -E 'ERROR|FATAL' log.txt | head -20",
        ]

        for cmd in default_commands[step:] + default_commands[:step]:
            if cmd not in recent_commands:
                return LogAction(action_type="bash", command=cmd)

        return LogAction(action_type="bash", command=default_commands[0])

    def _extract_submit_json(self, thought: str) -> Optional[Dict[str, Any]]:
        """Extract submit answer JSON from model output."""
        # Pattern 1: Nested format
        nested_match = re.search(r'"answer"\s*:\s*(\{[^}]+\})', thought)
        if nested_match:
            try:
                return json.loads(nested_match.group(1))
            except json.JSONDecodeError:
                pass

        # Pattern 2: Flat format
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
            comp_match = re.search(r"(service_[a-d])", line.lower())
            if comp_match:
                comp = comp_match.group(1)
                component_counts[comp] = component_counts.get(comp, 0) + 1

    if component_counts:
        return max(component_counts, key=lambda k: component_counts[k])
    return None


def _detect_anomaly_type(output: str) -> str:
    """Detect anomaly type from command output patterns."""
    output_lower = output.lower()

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
    if cascade_count >= 2:
        return "cascade_failure"

    memory_keywords = ["heap", "gc", "memory", "outofmemory", "mb"]
    memory_pattern = re.search(r"(\d{3,4})\s*mb", output_lower)
    if memory_pattern or sum(1 for kw in memory_keywords if kw in output_lower) >= 2:
        return "memory_leak"

    latency_keywords = ["latency", "timeout", "response time", "slow"]
    latency_pattern = re.search(r"latency[:\s]+(\d{3,})", output_lower)
    if latency_pattern or sum(1 for kw in latency_keywords if kw in output_lower) >= 2:
        return "latency_degradation"

    if "no lines" in output_lower or "0 matches" in output_lower:
        return "service_dropout"

    return "error_spike"


def _find_error_time_window(timestamps: List[str]) -> Tuple[str, str]:
    """Find the time window where errors are concentrated."""
    if not timestamps:
        return ("", "")
    if len(timestamps) == 1:
        return (timestamps[0], timestamps[0])

    sorted_ts = sorted(set(timestamps))
    if len(sorted_ts) <= 3:
        return (sorted_ts[0], sorted_ts[-1])

    trim = len(sorted_ts) // 5
    start_idx = trim
    end_idx = len(sorted_ts) - trim - 1

    return (sorted_ts[start_idx], sorted_ts[end_idx])


def _guess_submit_answer(command_history: List[Dict[str, Any]]) -> LogAction:
    """Build a best-effort submission from command history."""
    combined_output = "\n".join(str(entry.get("output", "")) for entry in command_history)

    anomaly_type = _detect_anomaly_type(combined_output)
    component = _extract_component_from_errors(combined_output)
    if not component:
        components = re.findall(r"\b(service_[a-d])\b", combined_output.lower())
        component = Counter(components).most_common(1)[0][0] if components else "service_a"

    error_timestamps = _extract_error_timestamps(combined_output)
    if len(error_timestamps) >= 2:
        start_time, end_time = _find_error_time_window(error_timestamps)
    else:
        all_timestamps = _extract_timestamps(combined_output)
        if len(all_timestamps) >= 2:
            start_time, end_time = _find_error_time_window(all_timestamps)
        elif all_timestamps:
            start_time = end_time = all_timestamps[0]
        else:
            start_time = end_time = ""

    if start_time and "." in start_time:
        start_time = start_time.split(".")[0]
    if end_time and "." in end_time:
        end_time = end_time.split(".")[0]

    return LogAction(
        action_type="submit",
        anomaly_type=anomaly_type,
        component=component,
        start_time=start_time,
        end_time=end_time,
        confidence=0.5,
    )


async def run_episode(
    env: Union[LogAnomalyEnvClient, LocalEnvWrapper],
    agent: ReactAgent,
    difficulty: str,
    episode_num: int,
    verbose_log: Optional[VerboseLog] = None,
    pbar: Optional[tqdm] = None,
) -> Optional[EpisodeResult]:
    """
    Run a single episode.

    Args:
        env: Environment client (WebSocket or local)
        agent: ReAct agent
        difficulty: Difficulty level
        episode_num: Episode number (for seed)
        verbose_log: Optional verbose log container
        pbar: Optional progress bar

    Returns:
        EpisodeResult or None if error
    """
    episode_id = f"{difficulty}_{episode_num}"
    step_logs: List[Dict[str, Any]] = []
    command_history: List[Dict[str, Any]] = []

    try:
        # Reset environment
        result = await env.reset(difficulty=difficulty, seed=episode_num)
        obs = result.observation

        if pbar:
            pbar.set_description(f"{difficulty} ep{episode_num + 1}")

        for step in range(agent.max_steps):
            if result.done or obs.answer_submitted:
                break

            is_last_step = step == agent.max_steps - 1

            if is_last_step and not obs.answer_submitted:
                # Force submission
                action = _guess_submit_answer(command_history)
                action_desc = "submit (forced)"
            else:
                # Agent thinks
                thought = await agent.think(
                    command_output=obs.command_output,
                    command_history=command_history,
                    steps_remaining=obs.steps_remaining,
                    total_steps=obs.total_steps,
                    difficulty=difficulty,
                )

                action = agent.parse_action(thought, command_history)

                # Minimum steps before submit
                if action.action_type == "submit" and step < MIN_STEPS_BEFORE_SUBMIT:
                    fallback_commands = [
                        "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
                        "grep ERROR log.txt | head -10 && echo '---' && grep ERROR log.txt | tail -10",
                        "grep -iE 'cascade|dependency|affected|latency|memory|heap' log.txt | head -15",
                    ]
                    cmd = fallback_commands[min(step, len(fallback_commands) - 1)]
                    action = LogAction(action_type="bash", command=cmd)
                    action_desc = f"bash (deferred submit): {cmd}"
                elif action.action_type == "bash":
                    action_desc = f"bash: {action.command}"
                else:
                    action_desc = "submit"

            # Execute action
            result = await env.step(action)
            obs = result.observation

            # Update command history for bash actions
            if action.action_type == "bash":
                command_history.append(
                    {
                        "command": action.command,
                        "output": obs.command_output,
                    }
                )

            # Log step
            step_log = {
                "step": step + 1,
                "action_type": action.action_type,
                "action": action.model_dump(exclude_none=True),
                "output": obs.command_output[:500] if obs.command_output else "",
                "reward": result.reward,
                "done": result.done,
            }
            step_logs.append(step_log)

            if pbar:
                pbar.update(0)  # Refresh display

        # Get episode result
        if hasattr(env, "get_result"):
            episode_result = env.get_result()
        else:
            # For WebSocket client, build result from final observation
            episode_result = EpisodeResult(
                episode_id=episode_id,
                task_id=episode_id,
                difficulty=DifficultyLevel(difficulty),
                reward=result.reward or 0.0,
                component_score=0.0,  # Not available from WebSocket
                type_score=0.0,
                window_score=0.0,
                efficiency_score=0.0,
                steps_used=len(step_logs),
                episode_complete=result.done,
                ground_truth={},
            )

        if verbose_log:
            verbose_log.add_episode(episode_id, difficulty, step_logs, episode_result)

        return episode_result

    except Exception as e:
        logger.error(f"Episode {episode_id} failed: {e}")
        if verbose_log:
            verbose_log.add_episode(episode_id, difficulty, step_logs, None)
        return None


async def run_baseline_inference(
    env_url: str,
    use_local: bool,
    difficulty: str,
    num_episodes: int,
    model: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run baseline inference on the environment.

    Args:
        env_url: Environment URL (HF Space or local server)
        use_local: If True, use local environment import
        difficulty: Difficulty level ("easy", "medium", "hard", or "all")
        num_episodes: Number of episodes per difficulty
        model: Model to use for inference
        output_path: Optional path for verbose JSON output

    Returns:
        Baseline results dictionary
    """
    if not API_KEY:
        return {
            "status": "error",
            "message": "No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.",
            "baseline_scores": {},
        }

    # Create agent
    agent = ReactAgent(model=model, max_steps=MAX_STEPS, base_url=API_BASE_URL)
    verbose_log = VerboseLog() if output_path else None

    # Determine difficulties
    if difficulty == "all":
        difficulties = [d.value for d in DifficultyLevel]
    else:
        difficulties = [difficulty]

    total_episodes = len(difficulties) * num_episodes
    all_results: List[EpisodeResult] = []

    try:
        # Create environment
        if use_local:
            from server.log_anomaly_environment import LogAnomalyEnvironment

            env = LocalEnvWrapper(LogAnomalyEnvironment())
        else:
            env = LogAnomalyEnvClient(base_url=env_url)

        async with env:
            with tqdm(total=total_episodes, desc="Running episodes", unit="ep") as pbar:
                for diff in difficulties:
                    for ep_num in range(num_episodes):
                        result = await run_episode(
                            env=env,
                            agent=agent,
                            difficulty=diff,
                            episode_num=ep_num,
                            verbose_log=verbose_log,
                            pbar=pbar,
                        )
                        if result:
                            all_results.append(result)
                            pbar.set_postfix(
                                reward=f"{result.reward:.2f}",
                                diff=diff,
                            )
                        pbar.update(1)

    finally:
        await agent.close()

    # Calculate summary
    summary = calculate_summary_stats(all_results) if all_results else {}

    # Build results
    results = {
        "status": "success",
        "model": model,
        "env_url": env_url if not use_local else "local",
        "total_episodes": len(all_results),
        "baseline_scores": summary,
        "detailed_results": [r.model_dump() for r in all_results],
    }

    # Save verbose output
    if output_path and verbose_log:
        verbose_log.save(output_path)
        results["verbose_output"] = output_path

    return results


async def main_async() -> None:
    """Async main entry point."""
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
        default=None,
        help=f"Model to use (default: MODEL_NAME env var or {DEFAULT_MODEL})",
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
        help="Output file for verbose results (JSON)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_ENV_URL,
        help=f"Environment URL (default: {DEFAULT_ENV_URL})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local environment instead of WebSocket connection",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum steps per episode (default: {MAX_STEPS})",
    )

    args = parser.parse_args()
    model = args.model or MODEL_NAME

    # Print configuration
    print(f"Running inference with model: {model}")
    print(f"API Base URL: {API_BASE_URL}")
    if args.local:
        print("Environment: LOCAL (direct import)")
    else:
        print(f"Environment URL: {args.url}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Episodes per difficulty: {args.episodes}")
    print()

    results = await run_baseline_inference(
        env_url=args.url,
        use_local=args.local,
        difficulty=args.difficulty,
        num_episodes=args.episodes,
        model=model,
        output_path=args.output,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)

    if results["status"] == "success":
        print(f"\nModel: {results['model']}")
        print(f"Environment: {results['env_url']}")
        print(f"Total episodes: {results['total_episodes']}")

        baseline = results.get("baseline_scores", {})
        if baseline:
            print(f"\nOverall: mean_reward={baseline.get('mean_reward', 0):.4f}")
            print("\nBy Difficulty:")
            for diff, scores in baseline.get("by_difficulty", {}).items():
                if isinstance(scores, dict):
                    print(f"  {diff}: mean_reward={scores.get('mean_reward', 0):.4f}")

        if args.output:
            print(f"\nVerbose output saved to: {args.output}")
    else:
        print(f"\nError: {results.get('message', 'Unknown error')}")


def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
