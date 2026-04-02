"""
Inference Script for Log Anomaly Investigation Environment.
===================================
MANDATORY ENVIRONMENT VARIABLES (set these before running):
    API_BASE_URL   The API endpoint for the LLM (default: HuggingFace router)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face API key

STDOUT FORMAT
- The script emits exactly three line types to stdout:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    python inference.py --difficulty all --episodes 2
    python inference.py --url https://ggtejas-log-anomaly-env.hf.space -d easy -e 1
"""

import asyncio
import json
import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

from models import (
    LogAction,
    DifficultyLevel,
)
from config import (
    MAX_STEPS,
    MIN_STEPS_BEFORE_SUBMIT,
    DEFAULT_MODEL,
    HF_ROUTER_URL,
    OUTPUT_PREVIEW_SHORT,
    OUTPUT_PREVIEW_LONG,
)
from client import LogAnomalyEnvClient, DEFAULT_ENV_URL

# =============================================================================
# Environment Variables (as required by hackathon)
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL") or HF_ROUTER_URL
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or DEFAULT_MODEL

# Benchmark identifier
BENCHMARK = "log-anomaly"

# Reproducibility settings
BASELINE_TEMPERATURE = 0.1
BASELINE_SEED = 42

# Success threshold (minimum score to be considered successful)
SUCCESS_SCORE_THRESHOLD = 0.3


# =============================================================================
# Structured Logging Functions (MANDATORY FORMAT)
# =============================================================================


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in required format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log step in required format."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Escape action string to be single-line safe
    action_safe = action.replace("\n", " ").replace("\r", "")[:100]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in required format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =============================================================================
# ReAct Agent
# =============================================================================


@dataclass
class ReactAgent:
    """ReAct-style agent for log anomaly investigation."""

    model: str = MODEL_NAME
    max_steps: int = MAX_STEPS
    base_url: str = API_BASE_URL
    temperature: float = BASELINE_TEMPERATURE
    _client: OpenAI = field(init=False)
    _api_model: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the OpenAI client."""
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=API_KEY,
            timeout=120.0,
        )
        # For HuggingFace router, append provider suffix
        if "huggingface" in self.base_url.lower():
            self._api_model = f"{self.model}:novita" if ":novita" not in self.model else self.model
        else:
            self._api_model = self.model

    def think(
        self,
        command_output: str,
        command_history: List[Dict[str, Any]],
        steps_remaining: int,
        total_steps: int,
        difficulty: str,
    ) -> str:
        """Generate next action based on current state."""
        prompt = self._build_prompt(command_output, command_history, steps_remaining, total_steps)

        try:
            completion = self._client.chat.completions.create(
                model=self._api_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(difficulty)},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=4096,
                seed=BASELINE_SEED,
            )
            content = completion.choices[0].message.content
            return content if content else "Error: No response"
        except Exception as e:
            return f"Error: {e}"

    def _get_system_prompt(self, difficulty: str = "easy") -> str:
        """Get system prompt based on difficulty."""
        if difficulty == "easy":
            anomaly_hint = "ANOMALY TYPE: error_spike (ONLY valid type for easy)"
            search_hint = "Look for: ERROR patterns"
        elif difficulty == "hard":
            anomaly_hint = "ANOMALY TYPE: cascade_failure (ONLY valid type for hard)"
            search_hint = "Look for: cascade, dependency, circuit breaker"
        else:
            anomaly_hint = "TYPES: error_spike, memory_leak, latency_degradation, service_dropout"
            search_hint = "Look for: ERROR, memory/heap, latency/timeout, unavailable"

        return f"""You are a DevOps engineer investigating log.txt for anomalies.

{anomaly_hint}
{search_hint}

To run a command, reply with ONLY a bash code block:
```bash
grep -i error log.txt | head -10
```

To submit your answer, reply with ONLY a json code block:
```json
{{"anomaly_type": "TYPE", "component": "SERVICE", "start_time": "TIMESTAMP", "end_time": "TIMESTAMP"}}
```

RULES:
1. Output exactly ONE code block per response
2. No text before or after the code block
3. Timestamps MUST be from grep output (format: 2026-03-27T14:55:00)
4. Investigate at least 3 commands before submitting"""

    def _build_prompt(
        self,
        command_output: str,
        command_history: List[Dict[str, Any]],
        steps_remaining: int,
        total_steps: int,
    ) -> str:
        """Build the user prompt."""
        steps_used = total_steps - steps_remaining
        parts = [f"Step {steps_used}/{total_steps}"]

        if steps_remaining <= 3:
            parts.append(f"\n!!! FINAL {steps_remaining} STEPS - Submit NOW !!!")

        if command_history:
            parts.append("\n=== RECENT OUTPUT ===")
            for entry in command_history[-3:]:
                parts.append(f"$ {entry['command']}")
                output = entry["output"]
                if not output.strip():
                    parts.append("(no output)")
                elif len(output) > OUTPUT_PREVIEW_SHORT:
                    parts.append(output[:OUTPUT_PREVIEW_SHORT] + "\n[truncated]")
                else:
                    parts.append(output)

        if command_output and not command_history:
            parts.append("=== OUTPUT ===")
            parts.append(command_output[:OUTPUT_PREVIEW_LONG])

        parts.append("\nReply with one code block:")
        return "\n".join(parts)

    def parse_action(self, thought: str, command_history: List[Dict[str, Any]]) -> LogAction:
        """Parse agent thought into LogAction."""
        # Handle Qwen thinking mode
        if "</think>" in thought:
            thought = thought.split("</think>")[-1].strip()

        thought = thought.replace("\r\n", "\n").replace("\r", "\n")

        # Look for fenced code blocks
        code_pattern = r"```(\w*)\s*\n(.*?)```"
        matches = re.findall(code_pattern, thought, re.DOTALL | re.IGNORECASE)

        for lang, content in matches:
            content = content.strip()
            if not content:
                continue

            lang = lang.lower()

            # JSON = submit
            if lang == "json" or content.startswith("{"):
                data = self._extract_json(content)
                if data:
                    return LogAction(
                        action_type="submit",
                        anomaly_type=data.get("anomaly_type", "error_spike"),
                        component=data.get("component", "unknown"),
                        start_time=data.get("start_time", ""),
                        end_time=data.get("end_time", ""),
                    )

            # Bash = command
            if lang in ("bash", "sh", "") and not content.startswith("{"):
                command = content.split("\n")[0].strip()
                if command:
                    return LogAction(action_type="bash", command=command)

        # Fallback
        return self._get_fallback(command_history)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text."""
        try:
            match = re.search(r"\{[^}]+\}", text)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        return None

    def _get_fallback(self, command_history: List[Dict[str, Any]]) -> LogAction:
        """Get fallback command."""
        recent = [e.get("command", "") for e in (command_history or [])[-3:]]
        step = len(command_history or [])

        commands = [
            "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
            "grep -iE 'cascade|dependency|circuit breaker' log.txt | head -15",
            "grep -iE 'latency|timeout|slow' log.txt | head -15",
            "grep -iE 'memory|heap|gc' log.txt | head -15",
            "grep ERROR log.txt | head -10",
        ]

        for cmd in commands[step:] + commands[:step]:
            if cmd not in recent:
                return LogAction(action_type="bash", command=cmd)

        return LogAction(action_type="bash", command=commands[0])


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_timestamps(output: str) -> List[str]:
    """Extract ISO timestamps from output."""
    return re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", output)


def _extract_component(output: str) -> str:
    """Extract most common component from output."""
    components = re.findall(r"\b(service_[a-d])\b", output.lower())
    if components:
        return Counter(components).most_common(1)[0][0]
    return "service_a"


def _detect_anomaly_type(output: str, difficulty: str) -> str:
    """Detect anomaly type based on output and difficulty."""
    if difficulty == "easy":
        return "error_spike"
    if difficulty == "hard":
        return "cascade_failure"

    lower = output.lower()
    if any(kw in lower for kw in ["cascade", "dependency", "circuit breaker"]):
        return "cascade_failure"
    if any(kw in lower for kw in ["heap", "gc", "memory"]):
        return "memory_leak"
    if any(kw in lower for kw in ["latency", "timeout"]):
        return "latency_degradation"
    return "error_spike"


def _guess_submit(command_history: List[Dict[str, Any]], difficulty: str) -> LogAction:
    """Build best-effort submission from history."""
    combined = "\n".join(str(e.get("output", "")) for e in command_history)

    anomaly_type = _detect_anomaly_type(combined, difficulty)
    component = _extract_component(combined)
    timestamps = _extract_timestamps(combined)

    start_time = timestamps[0] if timestamps else ""
    end_time = timestamps[-1] if timestamps else ""

    return LogAction(
        action_type="submit",
        anomaly_type=anomaly_type,
        component=component,
        start_time=start_time,
        end_time=end_time,
    )


# =============================================================================
# Main Episode Runner (async to match OpenEnv pattern)
# =============================================================================


async def run_episode(
    env,
    agent: ReactAgent,
    task_name: str,
    difficulty: str,
    episode_num: int,
    model_name: str,
) -> Tuple[float, int, List[float]]:
    """
    Run a single episode with structured logging.

    Returns:
        Tuple of (final_score, steps_taken, rewards_list)
    """
    rewards: List[float] = []
    steps_taken = 0
    command_history: List[Dict[str, Any]] = []
    score = 0.0
    success = False

    # Log start
    log_start(task=task_name, env=BENCHMARK, model=model_name)

    try:
        # Reset environment (async)
        seed = BASELINE_SEED + episode_num
        result = await env.reset(difficulty=difficulty, seed=seed)
        obs = result.observation
        done = result.done

        for step in range(1, agent.max_steps + 1):
            if done or obs.answer_submitted:
                break

            is_last = step == agent.max_steps

            # Get action (LLM call is sync as required by hackathon)
            if is_last and not obs.answer_submitted:
                action = _guess_submit(command_history, difficulty)
            else:
                thought = agent.think(
                    command_output=obs.command_output or "",
                    command_history=command_history,
                    steps_remaining=obs.steps_remaining,
                    total_steps=obs.total_steps,
                    difficulty=difficulty,
                )
                action = agent.parse_action(thought, command_history)

                # Enforce minimum investigation
                if action.action_type == "submit" and step < MIN_STEPS_BEFORE_SUBMIT:
                    fallback_cmds = [
                        "grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn",
                        "grep ERROR log.txt | head -10",
                        "grep -iE 'cascade|latency|memory' log.txt | head -15",
                    ]
                    action = LogAction(action_type="bash", command=fallback_cmds[min(step - 1, 2)])

            # Build action string for logging
            if action.action_type == "bash":
                action_str = f"bash({action.command})"
            else:
                action_str = f"submit({action.anomaly_type},{action.component})"

            # Execute action (async)
            result = await env.step(action)
            obs = result.observation
            done = result.done or obs.answer_submitted or obs.steps_remaining <= 0

            # Get reward (clamp to 0-1)
            reward = min(max(result.reward or 0.0, 0.0), 1.0)
            rewards.append(reward)
            steps_taken = step

            # Update history
            if action.action_type == "bash":
                command_history.append(
                    {
                        "command": action.command,
                        "output": obs.command_output or "",
                    }
                )

            # Log step
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        # Calculate final score (average reward, clamped to 0-1)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # Log error case
        score = 0.0
        success = False
        if not rewards:
            rewards = [0.0]
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        # Always log end
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, steps_taken, rewards


async def main() -> None:
    """Main entry point (async)."""
    parser = argparse.ArgumentParser(description="Log Anomaly Investigation Inference")
    parser.add_argument(
        "--difficulty",
        "-d",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty level",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=2,
        help="Episodes per difficulty (default: 2)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model to use",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_ENV_URL,
        help=f"Environment URL (default: {DEFAULT_ENV_URL})",
    )

    args = parser.parse_args()
    model = args.model or MODEL_NAME

    # Create WebSocket client to HF Space
    env = LogAnomalyEnvClient(base_url=args.url)

    # Create agent
    agent = ReactAgent(model=model, max_steps=MAX_STEPS, base_url=API_BASE_URL)

    # Determine tasks (difficulties)
    if args.difficulty == "all":
        difficulties = [d.value for d in DifficultyLevel]
    else:
        difficulties = [args.difficulty]

    # Run episodes
    all_scores: List[float] = []

    try:
        async with env:
            for difficulty in difficulties:
                for ep_num in range(args.episodes):
                    task_name = f"{difficulty}_{ep_num + 1}"
                    score, steps, rewards = await run_episode(
                        env=env,
                        agent=agent,
                        task_name=task_name,
                        difficulty=difficulty,
                        episode_num=ep_num,
                        model_name=model,
                    )
                    all_scores.append(score)
    except Exception as e:
        print(f"[DEBUG] Environment error: {e}", flush=True)

    # Print summary (separate from structured logs)
    if all_scores:
        print("\n" + "=" * 50, flush=True)
        print("SUMMARY", flush=True)
        print("=" * 50, flush=True)
        print(f"Total episodes: {len(all_scores)}", flush=True)
        print(f"Mean score: {sum(all_scores) / len(all_scores):.3f}", flush=True)
        print("=" * 50, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
