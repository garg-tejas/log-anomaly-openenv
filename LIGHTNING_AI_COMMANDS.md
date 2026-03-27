# Lightning AI L4 GPU - Exact Command Sequence

This document provides the **exact order of commands** to run baseline inference on Lightning AI with an L4 GPU.

**📋 For debugging model response parsing issues, see [DEBUG_INSTRUCTIONS.md](DEBUG_INSTRUCTIONS.md)**

---

## Prerequisites

- Lightning AI Studio with L4 GPU (24GB VRAM)
- Terminal access to the Lightning AI instance

---

## Step-by-Step Commands

### 1. Clone Repository

```bash
git clone https://github.com/garg-tejas/log-anomaly-openenv.git
cd log-anomaly-openenv
```

### 2. Install Dependencies

```bash
# Install all dependencies including vLLM
pip install -e ".[all]"
```

**Expected time**: ~2-3 minutes

---

## Option A: Using Docker (Recommended for Isolated Environment)

### 3a. Build Docker Image

```bash
docker build -t log-anomaly-env .
```

**Expected time**: ~5-10 minutes

### 4a. Run Docker Container (Environment Server)

```bash
# Terminal 1: Start environment server
docker run -p 8000:8000 log-anomaly-env
```

**Expected output**: 
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5a. Start vLLM Server (Separate Terminal)

```bash
# Terminal 2: Start vLLM server
vllm serve Qwen/Qwen3.5-4B \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype auto
```

**Expected time to load**: ~1-2 minutes  
**Expected output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### 6a. Run Baseline Inference (Separate Terminal)

```bash
# Terminal 3: Run baseline inference
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python baseline_inference.py \
  --difficulty all \
  --episodes 10 \
  --model Qwen/Qwen3.5-4B \
  --output baseline_before_enhancement.json
```

**Expected time**: ~30-60 minutes (30 episodes total: 10 easy, 10 medium, 10 hard)

---

## Option B: Without Docker (Direct Python)

### 3b. Start Environment Server

```bash
# Terminal 1: Start environment server
python -m server.app
```

**Expected output**:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4b. Start vLLM Server

```bash
# Terminal 2: Start vLLM server
vllm serve Qwen/Qwen3.5-4B \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype auto
```

**Expected time to load**: ~1-2 minutes

### 5b. Run Baseline Inference

```bash
# Terminal 3: Run baseline inference
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python baseline_inference.py \
  --difficulty all \
  --episodes 10 \
  --model Qwen/Qwen3.5-4B \
  --output baseline_before_enhancement.json
```

**Expected time**: ~30-60 minutes

---

## Option C: Using Helper Scripts (Easiest)

### 3c. Start vLLM with Helper Script

```bash
# Terminal 1: Start vLLM server
./start-vllm.sh Qwen/Qwen3.5-4B
```

**Note**: This script automatically sets optimal parameters for L4 GPU.

### 4c. Start Environment Server

```bash
# Terminal 2: Start environment server
python -m server.app
```

### 5c. Run Baseline Inference

```bash
# Terminal 3: Run baseline inference
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python baseline_inference.py \
  --difficulty all \
  --episodes 10 \
  --model Qwen/Qwen3.5-4B \
  --output baseline_before_enhancement.json
```

---

## Simplified Single-Terminal Option (Background Processes)

If you only have one terminal available:

```bash
# 1. Clone and setup
git clone https://github.com/garg-tejas/log-anomaly-openenv.git
cd log-anomaly-openenv
pip install -e ".[all]"

# 2. Start environment server in background
nohup python -m server.app > env_server.log 2>&1 &

# 3. Wait 5 seconds for server to start
sleep 5

# 4. Start vLLM in background
nohup vllm serve Qwen/Qwen3.5-4B \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype auto > vllm.log 2>&1 &

# 5. Wait for vLLM to load model (~2 minutes)
echo "Waiting for vLLM to load model (this takes ~2 minutes)..."
sleep 120

# 6. Check if vLLM is ready
tail -f vllm.log
# Look for: "Uvicorn running on http://0.0.0.0:8080"
# Press Ctrl+C when you see it

# 7. Run baseline inference
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python baseline_inference.py \
  --difficulty all \
  --episodes 10 \
  --model Qwen/Qwen3.5-4B \
  --output baseline_before_enhancement.json
```

---

## Expected Results to Share Back

After the baseline completes, you'll find `baseline_before_enhancement.json`. Please share:

1. **Overall mean reward** (compare vs 44% baseline)
2. **Breakdown by difficulty**:
   - Easy: ? %
   - Medium: ? %
   - Hard: ? %
3. **Component scores**: Mean component/type/window/efficiency
4. **Runtime duration**: How long did 30 episodes take?
5. **Any errors encountered**: Especially command failures or OOM issues

### Quick Summary Command

```bash
# Get summary from the output JSON
python -c "
import json
with open('baseline_before_enhancement.json') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f\"Overall Mean Reward: {summary.get('mean_reward', 0):.2%}\")
    print(f\"By Difficulty:\")
    for diff, stats in summary.get('by_difficulty', {}).items():
        print(f\"  {diff}: {stats.get('mean_reward', 0):.2%}\")
    print(f\"Mean Component: {summary.get('mean_component', 0):.4f}\")
    print(f\"Mean Type: {summary.get('mean_type', 0):.4f}\")
    print(f\"Mean Window: {summary.get('mean_window', 0):.4f}\")
    print(f\"Mean Efficiency: {summary.get('mean_efficiency', 0):.4f}\")
"
```

---

## Troubleshooting

### vLLM OOM (Out of Memory)
```bash
# Reduce GPU memory utilization
vllm serve Qwen/Qwen3.5-4B \
  --port 8080 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096 \
  --dtype auto
```

### vLLM Not Responding
```bash
# Check vLLM logs
tail -f vllm.log

# Check if vLLM is listening
curl http://localhost:8080/v1/models
```

### Environment Server Not Found
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start it
python -m server.app
```

### Baseline Hangs or Errors
```bash
# Run with verbose logging
python baseline_inference.py \
  --difficulty all \
  --episodes 10 \
  --model Qwen/Qwen3.5-4B \
  --output baseline_before_enhancement.json \
  --verbose
```

---

## GPU Monitoring (Optional)

```bash
# Monitor GPU usage while running
watch -n 1 nvidia-smi
```

**Expected GPU usage**:
- vLLM loading: ~3-4 GB VRAM
- During inference: ~4-6 GB VRAM peak

---

## After Baseline Completes

1. Stop all background processes:
```bash
# Stop vLLM
pkill -f "vllm serve"

# Stop environment server
pkill -f "python -m server.app"
```

2. Share the results summary (see "Expected Results to Share Back" above)

3. We'll implement Phase 2 reward enhancements based on your results
