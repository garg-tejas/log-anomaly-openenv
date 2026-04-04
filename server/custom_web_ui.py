"""
Custom OpenEnv web tab for log anomaly investigation.

This builder is mounted by OpenEnv as an extra tab alongside the default
Playground when `gradio_builder` is passed to `create_app`.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _obs_to_dict(observation: Any) -> Dict[str, Any]:
    """Best-effort observation serialization for UI display."""
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return {
        "command_output": str(getattr(observation, "command_output", "")),
        "reward": getattr(observation, "reward", None),
        "done": getattr(observation, "done", False),
        "steps_remaining": getattr(observation, "steps_remaining", None),
        "task_difficulty": str(getattr(observation, "task_difficulty", "easy")),
    }


def build_log_anomaly_tab(
    web_manager: Any,
    action_fields: Dict[str, Any],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> Any:
    """Build a custom visualization tab for the OpenEnv /web interface."""
    import gradio as gr

    del action_fields, is_chat_env, title, quick_start_md  # Not needed in this tab.

    def reset_episode(difficulty: str, seed_text: str) -> tuple[str, str, str]:
        """Reset episode with difficulty/seed controls."""
        reset_kwargs: Dict[str, Any] = {"difficulty": difficulty}
        cleaned_seed = seed_text.strip()
        if cleaned_seed:
            try:
                reset_kwargs["seed"] = int(cleaned_seed)
            except ValueError:
                return "", "Invalid seed. Provide an integer or leave blank.", "{}"

        observation = web_manager.env.reset(**reset_kwargs)
        obs = _obs_to_dict(observation)
        state = web_manager.get_state()
        summary = (
            f"difficulty={obs.get('task_difficulty')} "
            f"steps_remaining={obs.get('steps_remaining')} done={obs.get('done')}"
        )
        return (
            summary,
            str(obs.get("command_output", "")),
            json.dumps(state, indent=2, sort_keys=True),
        )

    def run_command(command: str) -> tuple[str, str, str]:
        """Execute a bash investigation command."""
        cmd = command.strip()
        if not cmd:
            return "", "Command is required.", "{}"
        observation = web_manager.env.step(web_manager.action_cls(action_type="bash", command=cmd))
        obs = _obs_to_dict(observation)
        state = web_manager.get_state()
        summary = (
            f"reward={obs.get('reward', 0.0)} done={obs.get('done')} "
            f"steps_remaining={obs.get('steps_remaining')}"
        )
        return (
            summary,
            str(obs.get("command_output", "")),
            json.dumps(state, indent=2, sort_keys=True),
        )

    def submit_answer(
        anomaly_type: str, component: str, start_time: str, end_time: str, confidence: float
    ) -> tuple[str, str, str]:
        """Submit a final answer and return graded result."""
        action = web_manager.action_cls(
            action_type="submit",
            anomaly_type=anomaly_type.strip(),
            component=component.strip(),
            start_time=start_time.strip(),
            end_time=end_time.strip(),
            confidence=confidence,
        )
        observation = web_manager.env.step(action)
        obs = _obs_to_dict(observation)
        state = web_manager.get_state()
        summary = (
            f"reward={obs.get('reward', 0.0)} done={obs.get('done')} "
            f"answer_submitted={obs.get('answer_submitted')}"
        )
        return (
            summary,
            str(obs.get("command_output", "")),
            json.dumps(state, indent=2, sort_keys=True),
        )

    with gr.Blocks() as tab:
        display_name = getattr(metadata, "title", None) or getattr(metadata, "name", "Environment")
        gr.Markdown(f"### {display_name} - Custom Investigation View")
        gr.Markdown(
            "Use this tab for quick manual debugging with controlled reset/step flows. "
            "Core OpenEnv APIs remain available at `/reset`, `/step`, and `/state`."
        )

        with gr.Row():
            difficulty = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Difficulty",
            )
            seed = gr.Textbox(
                value="",
                label="Seed (optional integer)",
                placeholder="42",
            )
            reset_btn = gr.Button("Reset Episode", variant="primary")

        command_input = gr.Textbox(
            label="Bash Command",
            placeholder="grep ERROR log.txt | head -20",
            lines=2,
        )
        run_btn = gr.Button("Run Command")

        with gr.Row():
            anomaly_type = gr.Dropdown(
                choices=[
                    "error_spike",
                    "memory_leak",
                    "service_dropout",
                    "latency_degradation",
                    "cascade_failure",
                    "auth_anomaly",
                ],
                value="error_spike",
                label="Anomaly Type",
            )
            component = gr.Textbox(label="Component", value="service_a")

        with gr.Row():
            start_time = gr.Textbox(label="Start Time", placeholder="2024-01-15T10:00:00")
            end_time = gr.Textbox(label="End Time", placeholder="2024-01-15T10:15:00")
            confidence = gr.Slider(
                minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Confidence"
            )

        submit_btn = gr.Button("Submit Answer", variant="secondary")

        status = gr.Textbox(label="Status", interactive=False)
        command_output = gr.Textbox(label="Command Output", lines=12, interactive=False)
        state_json = gr.Code(label="Current State", language="json", interactive=False)

        reset_btn.click(reset_episode, [difficulty, seed], [status, command_output, state_json])
        run_btn.click(run_command, [command_input], [status, command_output, state_json])
        submit_btn.click(
            submit_answer,
            [anomaly_type, component, start_time, end_time, confidence],
            [status, command_output, state_json],
        )

    return tab
