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

from openenv.core.env_server import create_app
from server.log_anomaly_environment import LogAnomalyEnvironment
from models import LogAction, LogObservation


# Create the app using OpenEnv's create_app (adds WebSocket /ws endpoint)
# create_app expects a factory function, not a class
def create_environment():
    return LogAnomalyEnvironment()


app = create_app(create_environment, LogAction, LogObservation, env_name="log_anomaly_env")


def main():
    """Run the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
