#!/bin/bash
# Helper script to start vLLM server for baseline inference
# 
# Usage:
#   ./start-vllm.sh                          # Start with default Qwen3.5-4B
#   ./start-vllm.sh Qwen/Qwen3.5-2B          # Start with different model
#   ./start-vllm.sh --help                   # Show help

set -e

# Default configuration
MODEL="${1:-Qwen/Qwen3.5-4B}"
PORT="${VLLM_PORT:-8080}"
MAX_MODEL_LEN="${VLLM_MAX_LEN:-4096}"
GPU_MEMORY="${VLLM_GPU_MEM:-0.8}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [MODEL] [OPTIONS]"
    echo ""
    echo "Start vLLM server for log anomaly baseline inference"
    echo ""
    echo "Arguments:"
    echo "  MODEL                Model name from HuggingFace (default: Qwen/Qwen3.5-4B)"
    echo ""
    echo "Environment Variables:"
    echo "  VLLM_PORT           Port to run vLLM on (default: 8080)"
    echo "  VLLM_MAX_LEN        Max model length (default: 4096)"
    echo "  VLLM_GPU_MEM        GPU memory utilization 0.0-1.0 (default: 0.8)"
    echo "  HF_TOKEN            HuggingFace token for model access"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start with Qwen3.5-4B"
    echo "  $0 Qwen/Qwen3.5-2B                    # Start with Qwen3.5-2B"
    echo "  VLLM_PORT=8090 $0                     # Use different port"
    echo ""
    exit 0
fi

echo -e "${GREEN}Starting vLLM server...${NC}"
echo -e "${YELLOW}Model:${NC} $MODEL"
echo -e "${YELLOW}Port:${NC} $PORT"
echo -e "${YELLOW}Max Length:${NC} $MAX_MODEL_LEN"
echo -e "${YELLOW}GPU Memory:${NC} $GPU_MEMORY"
echo ""

# Check if HF_TOKEN is set
if [[ -z "$HF_TOKEN" ]]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set. Some models may require authentication.${NC}"
    echo -e "${YELLOW}Set it with: export HF_TOKEN=your_token${NC}"
    echo ""
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU may not be available.${NC}"
    exit 1
fi

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM not installed.${NC}"
    echo -e "${YELLOW}Install with: pip install vllm${NC}"
    exit 1
fi

# Start vLLM server
echo -e "${GREEN}Launching vLLM server (this may take 1-2 minutes to load model)...${NC}"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    2>&1 | tee vllm_server.log &

VLLM_PID=$!
echo -e "${GREEN}vLLM server started with PID: $VLLM_PID${NC}"
echo -e "${YELLOW}Logs: vllm_server.log${NC}"
echo ""
echo -e "${GREEN}Waiting for server to be ready...${NC}"

# Wait for server to be ready (up to 3 minutes)
MAX_WAIT=180
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ vLLM server is ready!${NC}"
        echo ""
        echo -e "${GREEN}Server Info:${NC}"
        echo -e "  Health:  http://localhost:$PORT/health"
        echo -e "  Models:  http://localhost:$PORT/v1/models"
        echo -e "  API:     http://localhost:$PORT/v1"
        echo ""
        echo -e "${GREEN}Run baseline with:${NC}"
        echo -e "  export OPENAI_BASE_URL=http://localhost:$PORT/v1"
        echo -e "  export OPENAI_API_KEY=dummy"
        echo -e "  python baseline_inference.py --difficulty all --episodes 10"
        echo ""
        echo -e "${YELLOW}To stop the server:${NC} kill $VLLM_PID"
        exit 0
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo -ne "  Waiting... ${WAITED}s / ${MAX_WAIT}s\r"
done

echo -e "\n${RED}Error: Server did not start within ${MAX_WAIT}s${NC}"
echo -e "${YELLOW}Check logs: tail -f vllm_server.log${NC}"
kill $VLLM_PID 2>/dev/null || true
exit 1
