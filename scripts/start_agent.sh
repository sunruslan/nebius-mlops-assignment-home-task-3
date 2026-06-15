#!/usr/bin/env bash
# Start the agent server with concurrent workers for load testing.
# Workers default to 48; override with AGENT_WORKERS=32 ./scripts/start_agent.sh

set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${AGENT_WORKERS:-48}"
exec .venv/bin/uvicorn agent.server:app --host 0.0.0.0 --port 8001 --workers "$WORKERS"
