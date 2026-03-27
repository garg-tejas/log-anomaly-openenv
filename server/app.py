"""
FastAPI Application for Log Anomaly Investigation Environment.

This module creates an HTTP server that exposes the LogAnomalyEnvironment
using the OpenEnv spec with WebSocket support.
"""

import os
import sys

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add parent directory to path for imports when running directly
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Try to use OpenEnv's create_app, fallback to custom implementation
try:
    from openenv.core.env_server import create_app
    from server.log_anomaly_environment import LogAnomalyEnvironment
    from models import LogAction, LogObservation

    # Create the app using OpenEnv's create_app (adds WebSocket /ws endpoint)
    app = create_app(LogAnomalyEnvironment, LogAction, LogObservation, env_name="log_anomaly_env")

except ImportError:
    # Fallback: Use custom FastAPI implementation for local development
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Any, Dict, Optional

    from server.log_anomaly_environment import LogAnomalyEnvironment
    from models import (
        LogAction,
        LogObservation,
        InvestigationAction,
        BashCommand,
        SubmitAnswer,
        AnomalyType,
        DifficultyLevel,
    )

    app = FastAPI(
        title="Log Anomaly Investigation Environment",
        description="A real-world OpenEnv environment for training AI agents to investigate log anomalies.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize environment
    environment = LogAnomalyEnvironment()

    class ResetRequest(BaseModel):
        difficulty: str = "easy"
        seed: Optional[int] = None
        task_id: Optional[str] = None
        episode_id: Optional[str] = None
        mode: str = "eval"
        data_source: str = "synthetic"
        log_source: Optional[str] = None

    class StepRequest(BaseModel):
        episode_id: Optional[str] = None
        action_type: str
        command: Optional[str] = None  # For bash actions
        anomaly_type: Optional[str] = None  # For submit actions
        component: Optional[str] = None
        start_time: Optional[str] = None
        end_time: Optional[str] = None
        confidence: float = 1.0

    class BaselineRequest(BaseModel):
        difficulty: str = "all"
        model: str = "Qwen/Qwen3.5-2B"
        num_episodes: int = 5

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "log-anomaly-env"}

    @app.post("/reset")
    async def reset_environment(request: ResetRequest) -> Dict[str, Any]:
        try:
            observation = environment.reset(
                seed=request.seed,
                episode_id=request.episode_id,
                difficulty=request.difficulty,
                task_id=request.task_id,
                mode=request.mode,
                data_source=request.data_source,
                log_source=request.log_source,
            )
            return {
                "observation": observation.model_dump(),
                "episode_id": environment.state.episode_id,
                "reward": observation.reward,
                "done": observation.done,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/step")
    async def step_environment(request: StepRequest) -> Dict[str, Any]:
        try:
            episode_id = request.episode_id or environment.state.episode_id
            if not episode_id or episode_id not in environment.episodes:
                raise HTTPException(
                    status_code=400, detail="Invalid episode_id. Call /reset first."
                )

            episode = environment.episodes[episode_id]

            # Build LogAction from request
            action = LogAction(
                action_type=request.action_type,
                command=request.command,
                anomaly_type=request.anomaly_type,
                component=request.component,
                start_time=request.start_time,
                end_time=request.end_time,
                confidence=request.confidence,
            )

            observation = episode.step(action)
            environment._state.step_count = episode.step_count

            return {
                "observation": observation.model_dump(),
                "episode_id": episode_id,
                "reward": observation.reward,
                "done": observation.done,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/state")
    async def get_state() -> Dict[str, Any]:
        try:
            state = environment.state
            return state.model_dump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/tasks")
    async def list_tasks() -> Dict[str, Any]:
        try:
            tasks = environment.list_tasks()
            return {"tasks": tasks}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/grade/{episode_id}")
    async def grade_episode(episode_id: str) -> Dict[str, Any]:
        try:
            result = environment.grade(episode_id)
            return result.model_dump()
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/grader")
    async def grader_endpoint() -> Dict[str, Any]:
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
