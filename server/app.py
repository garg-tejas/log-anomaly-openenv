"""
FastAPI Application for Log Anomaly Investigation Environment.

This module creates an HTTP server that exposes the LogAnomalyEnvironment
using the OpenEnv spec with WebSocket support.

For GRPO training with TRL, this server supports concurrent sessions
to handle parallel generation batches.
"""

import os
import sys

from dotenv import load_dotenv
from fastapi.responses import RedirectResponse

load_dotenv()
# Default to web UI enabled unless explicitly overridden by environment.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "1")

# Add parent directory to path for imports when running directly
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from openenv.core.env_server import create_app
from server.log_anomaly_environment import LogAnomalyEnvironment
from server.custom_web_ui import build_log_anomaly_tab
from models import LogAction, LogObservation


# =============================================================================
# Server Configuration
# =============================================================================

# Enable concurrent sessions for GRPO training
# TRL's GRPOTrainer opens N WebSocket connections (one per generation)
# This should be >= gradient_accumulation_steps * per_device_batch_size
SUPPORTS_CONCURRENT_SESSIONS: bool = True
MAX_CONCURRENT_ENVS: int = int(os.getenv("LOG_ANOMALY_MAX_CONCURRENT_ENVS", "64"))
WEB_INTERFACE_ENABLED: bool = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in (
    "true",
    "1",
    "yes",
)


# =============================================================================
# Environment Factory
# =============================================================================


def create_environment() -> LogAnomalyEnvironment:
    """
    Factory function to create environment instances.

    This is called by OpenEnv for each new session/connection.
    """
    return LogAnomalyEnvironment()


# =============================================================================
# Application Setup
# =============================================================================

# Create the app using OpenEnv's create_app (adds WebSocket /ws endpoint)
app = create_app(
    create_environment,
    LogAction,
    LogObservation,
    env_name="log_anomaly_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
    gradio_builder=build_log_anomaly_tab,
)


@app.get("/", include_in_schema=False, response_model=None)
def root():
    """Serve UI from root when web interface is enabled."""
    if WEB_INTERFACE_ENABLED:
        return RedirectResponse(url="/web")
    return {
        "message": "Log Anomaly Environment API",
        "docs": "/docs",
        "web": "Set ENABLE_WEB_INTERFACE=1 and open /web",
    }


def main() -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
