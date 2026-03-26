"""
FastAPI Application for Log Anomaly Investigation Environment.

This module creates an HTTP server that exposes the LogAnomalyEnvironment
over REST endpoints following the OpenEnv specification.
"""

import os
import sys
from typing import Any, Dict, Optional, List

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars only

# Add parent directory to path for imports when running directly
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.log_anomaly_environment import LogAnomalyEnvironment
from models import (
    InvestigationAction,
    InvestigationObservation,
    InvestigationState,
    EpisodeResult,
    AnomalyType,
    DifficultyLevel,
    SubmitAnswer,
    BashCommand,
)

# Create FastAPI app
app = FastAPI(
    title="Log Anomaly Investigation Environment",
    description="""
    A real-world OpenEnv environment for training AI agents to investigate
    log anomalies using bash command exploration.

    ## Features
    - Realistic log investigation scenarios
    - Multiple difficulty levels (easy, medium, hard)
    - Sandboxed bash command execution
    - Automated grading with multi-axis scoring
    - Baseline inference support

    ## Quick Start
    1. POST /reset - Start a new investigation episode
    2. POST /step - Execute bash commands or submit answers
    3. GET /tasks - List available tasks
    4. GET /grade/{episode_id} - Get grading results
    5. POST /grader - Grade the current episode
    """,
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
environment = LogAnomalyEnvironment()


# Request/Response models for endpoints
class ResetRequest(BaseModel):
    """Request body for reset endpoint."""

    difficulty: str = "easy"
    seed: Optional[int] = None
    task_id: Optional[str] = None
    episode_id: Optional[str] = None
    mode: str = "eval"  # "training" or "eval"
    data_source: str = "synthetic"  # "synthetic" or "loghub"
    log_source: Optional[str] = None  # "HDFS", "BGL", "OpenStack", "Apache"


class StepRequest(BaseModel):
    """Request body for step endpoint."""

    episode_id: Optional[str] = None  # Episode ID for thread safety
    action_type: str
    bash_command: Optional[BashCommand] = None
    answer: Optional[SubmitAnswer] = None


class BaselineRequest(BaseModel):
    """Request body for baseline endpoint."""

    difficulty: str = "all"
    model: str = "Qwen/Qwen3.5-2B"
    num_episodes: int = 5


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "log-anomaly-env"}


@app.post("/reset")
async def reset_environment(request: ResetRequest) -> Dict[str, Any]:
    """
    Reset the environment and start a new episode.

    Args:
        request: Reset parameters including difficulty, seed, task_id

    Returns:
        Initial observation and metadata
    """
    try:
        observation = environment.reset(
            seed=request.seed,
            difficulty=request.difficulty,
            episode_id=request.episode_id,
            task_id=request.task_id,
            mode=request.mode,
            data_source=request.data_source,
            log_source=request.log_source,
        )

        # Build response with mode-appropriate metadata
        # In EVAL mode: NO ground truth hints (prevents cheating)
        # In TRAINING mode: General difficulty hints only
        response_metadata = {
            "log_file": "log.txt",
            "difficulty": request.difficulty,
            "mode": request.mode,
            "data_source": request.data_source,
        }

        # Add log_source if specified
        if request.log_source:
            response_metadata["log_source"] = request.log_source

        return {
            "observation": observation.model_dump(),
            "episode_id": environment.state.episode_id,
            "metadata": response_metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step_environment(request: StepRequest) -> Dict[str, Any]:
    """
    Execute an action in the environment.

    Args:
        request: Action to execute (bash command or submit answer)

    Returns:
        Observation and reward
    """
    try:
        # Use provided episode_id or fall back to current
        episode_id = request.episode_id or environment.state.episode_id

        if not episode_id or episode_id not in environment.episodes:
            raise HTTPException(
                status_code=400, detail=f"Invalid episode_id. Call /reset first. Got: {episode_id}"
            )

        # Get the specific episode
        episode = environment.episodes[episode_id]

        # Build InvestigationAction from request
        action = InvestigationAction(
            action_type=request.action_type,
            bash_command=request.bash_command,
            answer=request.answer,
        )

        # Execute on the specific episode
        observation = episode.step(action)
        environment.state.step_count = episode.step_count

        return {
            "observation": observation.model_dump(),
            "episode_id": episode_id,
            "reward": observation.episode_reward,
            "done": observation.answer_submitted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """
    Get current environment state.

    Returns:
        Current state information
    """
    try:
        state = environment.state
        return {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "log_file_path": state.log_file_path,
            "task_id": state.task_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """
    List all available tasks.

    Returns:
        Task definitions and schemas
    """
    try:
        tasks = environment.list_tasks()
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grade/{episode_id}")
async def grade_episode(episode_id: str) -> Dict[str, Any]:
    """
    Grade a completed episode.

    Args:
        episode_id: Episode to grade

    Returns:
        Grading results with scores
    """
    try:
        result = environment.grade(episode_id)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/grader")
async def grader_endpoint() -> Dict[str, Any]:
    """
    Grade the current episode.

    Returns:
        Grading results with scores for the current episode
    """
    try:
        if environment.episode is None:
            raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")

        episode_id = environment.state.episode_id
        result = environment.grade(episode_id)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline")
async def run_baseline(request: BaselineRequest) -> Dict[str, Any]:
    """
    Run baseline inference on the environment.

    Args:
        request: Baseline parameters (difficulty, model, num_episodes)

    Returns:
        Baseline results
    """
    try:
        from baseline_inference import run_baseline_inference

        results = run_baseline_inference(
            environment=environment,
            difficulty=request.difficulty,
            model=request.model,
            num_episodes=request.num_episodes,
        )
        return results
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Baseline inference not available. Please install openai package.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Log Anomaly Investigation Environment",
        "version": "1.0.0",
        "description": "A real-world OpenEnv environment for training AI agents to investigate log anomalies",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "grade": "GET /grade/{episode_id}",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
        },
        "anomaly_types": [t.value for t in AnomalyType],
        "difficulty_levels": [d.value for d in DifficultyLevel],
    }


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
