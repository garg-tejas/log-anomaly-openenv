# Log Anomaly Investigation Environment

A real-world OpenEnv environment for training AI agents to investigate log anomalies using bash command exploration.

## Overview

This environment simulates a realistic log investigation scenario where an agent must identify anomalies in system logs using only read-only bash commands. It provides a sandboxed environment for training and evaluating AI agents on multi-turn investigation tasks.

### Key Features

- **Realistic Task**: Simulates actual DevOps/SRE log investigation workflows
- **Sandboxed Execution**: Safe bash command execution with security controls
- **Multiple Difficulty Levels**: Easy, Medium, and Hard tasks
- **Multi-axis Grading**: Component identification, type classification, window precision, efficiency
- **Baseline Inference**: ReAct + GPT-4o baseline included

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv

### From Source

```bash
# Clone the repository
git clone https://github.com/openenv/log-anomaly-env.git
cd log-anomaly-env

# Install dependencies
pip install -e ".[all]"

# Or with uv
uv sync
```

### From PyPI (when available)

```bash
pip install log-anomaly-env
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
    "bash_command": {"command": "grep ERROR log.txt | head -20"}
  }'

# Submit answer
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "submit",
    "answer": {
      "anomaly_type": "error_spike",
      "component": "service_a",
      "start_time": "2024-01-15T10:00:00",
      "end_time": "2024-01-15T10:15:00"
    }
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
| Hard   | Low (0.25)     | All types        | ~2000 lines |

## Baseline Inference

Run the ReAct + GPT-4o baseline:

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Run baseline
python baseline_inference.py --difficulty all --episodes 5

# Or programmatically
from baseline_inference import run_baseline_inference

results = run_baseline_inference(
    environment=env,
    difficulty="all",
    model="gpt-4o",
    num_episodes=3,
)
```

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

#### InvestigationAction

```python
{
    "action_type": "bash" | "submit",
    "bash_command": {"command": "string"},
    "answer": {
        "anomaly_type": "error_spike" | "memory_leak" | "service_dropout",
        "component": "string",
        "start_time": "ISO timestamp",
        "end_time": "ISO timestamp"
    }
}
```

#### InvestigationObservation

```python
{
    "command_output": "string",
    "command_history": [{"command": "...", "output": "..."}],
    "steps_remaining": 15,
    "total_steps": 15,
    "answer_submitted": false,
    "task_difficulty": "easy" | "medium" | "hard",
    "episode_reward": 0.0
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
├── baseline_inference.py # Baseline script
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
- Part of PyTorch OpenEnv AI Hackathon
