---
title: Log Anomaly Investigation Environment
emoji: "🔍"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - log-analysis
  - anomaly-detection
  - real-world-evaluation
---

# Log Anomaly Investigation Environment

A real-world OpenEnv environment for evaluating AI agents on log anomaly investigation with bash command exploration.

## Overview

This environment simulates a realistic log investigation scenario where an agent must identify anomalies in system logs using only read-only bash commands. It provides a sandboxed environment for multi-turn agent evaluation and baseline end-to-end validation.

### Key Features

- **Realistic Task**: Simulates actual DevOps/SRE log investigation workflows
- **Sandboxed Execution**: Safe bash command execution with security controls
- **Multiple Difficulty Levels**: Easy, Medium, and Hard tasks
- **Multi-axis Grading**: Component identification, type classification, window precision, efficiency
- **Baseline Inference**: ReAct + Qwen baseline included

## Round 1 Scope

For OpenEnv Round 1, this repository focuses on:

- A working OpenEnv-compliant environment (`reset`, `step`, `state`)
- At least three graded tasks (easy/medium/hard)
- A reproducible baseline `inference.py` to demonstrate agent interaction

RL training is optional and not required for Round 1 submission.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv

### From Source

```bash
# Clone the repository
git clone https://github.com/garg-tejas/log-anomaly-openenv.git
cd log-anomaly-openenv

# Install dependencies
pip install -e ".[all]"

# Or with uv
uv sync
```

## Quick Start

### Using the Python API

```python
from log_anomaly_env import LogAnomalyEnv, InvestigationAction, BashCommand

# Connect to environment
env = LogAnomalyEnv(base_url="http://localhost:8000")

# Reset and start episode
result = env.reset(difficulty="easy")
print(f"Steps remaining: {result['observation']['steps_remaining']}")

# Execute bash commands
action = InvestigationAction(
    action_type="bash",
    bash_command=BashCommand(command="grep ERROR log.txt | head -20")
)
result = env.step(action)
print(result['observation']['command_output'][:500])

# Submit answer when ready
action = InvestigationAction(
    action_type="submit",
    answer={
        "anomaly_type": "error_spike",
        "component": "service_a",
        "start_time": "2024-01-15T10:00:00",
        "end_time": "2024-01-15T10:15:00"
    }
)
result = env.step(action)

# Get grading
grade_result = env.grade(result['episode_id'])
print(f"Total Score: {grade_result['reward']}")
```

### Running the Server

```bash
# Start the server
cd log_anomaly_env
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or with Docker
docker build -t log-anomaly-env:latest -f server/Dockerfile .
docker run -p 8000:8000 log-anomaly-env:latest
```

To enable the OpenEnv web UI:

```bash
export ENABLE_WEB_INTERFACE=1
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/web
```

The web app includes the default Playground and a custom Visualization tab for investigation-specific controls.

### Using HTTP Endpoints

```bash
# Start episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Execute step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "bash",
    "command": "grep ERROR log.txt | head -20"
  }'

# Submit answer
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "submit",
    "anomaly_type": "error_spike",
    "component": "service_a",
    "start_time": "2024-01-15T10:00:00",
    "end_time": "2024-01-15T10:15:00"
  }'
```

## Environment Details

### Task Description

The agent must investigate a log file to identify anomalies. An anomaly is characterized by:

1. **Type**: What kind of anomaly (error spike, memory leak, service dropout)
2. **Component**: Which system component is affected
3. **Time Window**: When the anomaly occurred

### Action Space

| Action Type | Description                                     |
| ----------- | ----------------------------------------------- |
| `bash`      | Execute a read-only bash command in the sandbox |
| `submit`    | Submit the final answer                         |

### Allowed Bash Commands

- `grep`, `egrep`, `fgrep` - Pattern matching
- `awk`, `sed` - Text processing
- `sort`, `uniq`, `wc` - Counting/aggregation
- `head`, `tail`, `cut` - Line selection
- `cat`, `less`, `more` - File display
- `find`, `xargs` - File search
- `date`, `echo`, `ls`, `pwd` - Utilities

### Observation Space

| Field              | Description                               |
| ------------------ | ----------------------------------------- |
| `command_output`   | Output from the last bash command         |
| `command_history`  | List of all previous commands and outputs |
| `steps_remaining`  | Number of steps left (max 15)             |
| `answer_submitted` | Whether the answer has been submitted     |
| `task_difficulty`  | Current difficulty level                  |
| `episode_reward`   | Cumulative reward so far                  |

### Grading

Episodes are graded on four components:

| Component                | Weight | Description                               |
| ------------------------ | ------ | ----------------------------------------- |
| Component Identification | 0.25   | Correctly identify the affected component |
| Type Classification      | 0.25   | Correctly identify anomaly type           |
| Window Precision         | 0.35   | IoU of predicted vs actual time window    |
| Investigation Efficiency | 0.15   | Fewer steps = higher score                |

### Difficulty Levels

| Level  | Intensity      | Anomaly Types    | Log Size    |
| ------ | -------------- | ---------------- | ----------- |
| Easy   | High (0.8)     | Error Spike only | ~500 lines  |
| Medium | Moderate (0.5) | All types        | ~1000 lines |
| Hard   | Low (0.6)      | Cascade Failure  | ~2000 lines |

## Baseline Scores

Baseline results using `Qwen/Qwen3.5-9B` with ReAct prompting (9 episodes):

| Difficulty  | Mean Reward | Component | Type | Window | Efficiency |
| ----------- | ----------- | --------- | ---- | ------ | ---------- |
| **Easy**    | 0.87        | 1.00      | 1.00 | 0.75   | 0.71       |
| **Medium**  | 0.64        | 1.00      | 0.67 | 0.55   | 0.21       |
| **Hard**    | 0.47        | 0.33      | 1.00 | 0.11   | 0.64       |
| **Overall** | **0.66**    | 0.78      | 0.89 | 0.47   | 0.52       |

_9 episodes (3 per difficulty). Hard tasks require identifying the root cause component in cascade failures, which remains challenging._

### Highlights

- **Best episode**: 0.94 reward (easy) - near-perfect detection
- **Easy tasks**: 100% component accuracy, 100% type accuracy
- **Medium tasks**: 100% component accuracy, varied anomaly types (memory_leak, latency_degradation, service_dropout)
- **Hard tasks**: Cascade failure type detection is strong (100%), but root cause identification is challenging (33%)

## Baseline Inference

Run the ReAct + Qwen baseline. The script is environment-variable driven and works for both local and deployed environments.

### Option 1: HuggingFace Router (Cloud)

```bash
# Set HuggingFace token
export HF_TOKEN="your-huggingface-token"
export MODEL_NAME="Qwen/Qwen3.5-4B"  # optional, has default
export BASE_URL="https://your-space-name.hf.space"

# Batch baseline across easy/medium/hard (default mode)
uv run python inference.py --mode batch --difficulty all --episodes 2
```

### Option 2: Local LLM with Ollama

```bash
# Install and start Ollama
ollama pull qwen2.5:7b

# Set environment variables
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="qwen2.5:7b"
export BASE_URL="http://localhost:8000"

# Run baseline
uv run python inference.py --mode batch --difficulty all --episodes 2
```

### Option 3: Local LLM with vLLM

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --port 8080

# Set environment variables
export API_BASE_URL="http://localhost:8080/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export BASE_URL="http://localhost:8000"

# Run baseline
uv run python inference.py --mode batch --difficulty all --episodes 2
```

### Single-Task Execution (Evaluator-Compatible)

```bash
# One task, one episode
uv run python inference.py --mode single --difficulty medium --url http://localhost:8000
```

### Optional Aggregate Summary

```bash
uv run python inference.py --mode batch --difficulty all --episodes 2 --summary-to-stderr
```

## Optional: Training with TRL/GRPO

This environment supports reinforcement learning training with TRL's GRPOTrainer.
Train your own log investigation agent using Group Relative Policy Optimization.

### Training Installation

```bash
# Install training dependencies
pip install -e ".[training]"

# Or with vLLM for faster inference
pip install -e ".[training-vllm]"
```

### Quick Start Training

```bash
# Basic training with curriculum learning (recommended)
python train_grpo.py --model Qwen/Qwen3-4B --curriculum

# Quick test with small model
python train_grpo.py --model Qwen/Qwen3-0.6B --num-samples 20 --no-vllm

# Train on specific difficulty
python train_grpo.py --model Qwen/Qwen3-4B --difficulty easy --num-samples 100
```

### Curriculum Learning

The environment supports progressive difficulty training:

| Phase       | Episodes | Success Rate | Difficulty |
| ----------- | -------- | ------------ | ---------- |
| Warmup      | 0-20     | -            | Easy       |
| Learning    | 20+      | < 40%        | Easy       |
| Progression | 20+      | 40-70%       | Medium     |
| Advanced    | 20+      | > 70%        | Hard       |

Enable with `--curriculum` flag or use `CurriculumLogAnomalyEnv` directly.

### vLLM Integration

For faster training, use vLLM in colocate or server mode:

```bash
# Colocate mode (single GPU, recommended for getting started)
python train_grpo.py --model Qwen/Qwen3-4B --vllm-mode colocate

# Server mode (multi-GPU setup)
# Terminal 1: Start vLLM server
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-4B --port 8000
# Terminal 2: Run training
CUDA_VISIBLE_DEVICES=1 python train_grpo.py --vllm-mode server --vllm-server-url http://localhost:8000
```

### Custom Training Scripts

Use the environment factory directly in your training code:

```python
from training_client import LogAnomalyTrainingEnv, CurriculumLogAnomalyEnv
from training_prompts import create_training_dataset_dict
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Create dataset
data = create_training_dataset_dict(num_easy=100, num_medium=100, num_hard=100)
dataset = Dataset.from_dict(data)

# Define reward function
def reward_func(environments, **kwargs):
    return [env.reward for env in environments]

# Create trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    environment_factory=CurriculumLogAnomalyEnv,  # or LogAnomalyTrainingEnv
    args=GRPOConfig(
        output_dir="./outputs",
        num_generations=4,
        max_completion_length=4096,
    ),
)

trainer.train()
```

### Training Environment API

The training environment exposes two tool methods for TRL:

| Method                                                 | Description                       |
| ------------------------------------------------------ | --------------------------------- |
| `bash(command)`                                        | Execute bash command on log.txt   |
| `submit(anomaly_type, component, start_time, end_time)` | Submit final answer               |

After each episode, read `env.reward` (0.0-1.0) for the grading result.

## API Reference

### Endpoints

| Endpoint              | Method | Description            |
| --------------------- | ------ | ---------------------- |
| `/health`             | GET    | Health check           |
| `/reset`              | POST   | Start new episode      |
| `/step`               | POST   | Execute action         |
| `/state`              | GET    | Get current state      |
| `/tasks`              | GET    | List available tasks   |
| `/grade/{episode_id}` | GET    | Get episode grading    |
| `/baseline`           | POST   | Run baseline inference |

### Models

#### LogAction

```python
{
    "action_type": "bash" | "submit",
    "command": "string",  # for bash
    "anomaly_type": "error_spike" | "memory_leak" | "service_dropout" | "latency_degradation" | "cascade_failure" | "auth_anomaly",
    "component": "string",  # for submit
    "start_time": "ISO timestamp",  # for submit
    "end_time": "ISO timestamp",  # for submit
    "confidence": 1.0
}
```

#### LogObservation

```python
{
    "command_output": "string",
    "stderr": "string",
    "exit_code": 0,
    "steps_remaining": 15,
    "total_steps": 15,
    "answer_submitted": false,
    "task_difficulty": "easy" | "medium" | "hard",
    "done": false,
    "reward": 0.0,
    "metadata": {}
}
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=log_anomaly_env --cov-report=html
```

### Project Structure

```
log_anomaly_env/
├── __init__.py           # Package exports
├── models.py             # Data models
├── client.py             # Client implementation
├── log_utils.py          # Log parsing and injection
├── grader.py             # Grading system
├── inference.py          # Baseline inference script
├── server/
│   ├── __init__.py
│   ├── log_anomaly_environment.py  # Main environment
│   ├── app.py            # FastAPI server
│   └── Dockerfile        # Container image
├── tests/
│   └── test_environment.py
├── openenv.yaml          # OpenEnv config
├── pyproject.toml        # Project config
└── README.md
```

## Deployment

### HuggingFace Spaces

```bash
# Install OpenEnv CLI
pip install openenv-core

# Login to HuggingFace
huggingface-cli login

# Push to Spaces
openenv push --repo-id your-username/log-anomaly-env
```

### Docker

```bash
# Build image
docker build -t log-anomaly-env:latest -f server/Dockerfile .

# Run container
docker run -p 8000:8000 log-anomaly-env:latest

# With GPU support (for baseline)
docker run --gpus all -p 8000:8000 log-anomaly-env:latest
```

## Acknowledgments

- Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework
- Inspired by real-world DevOps/SRE workflows
- Part of OpenEnv AI Hackathon
