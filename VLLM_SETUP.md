# vLLM Setup Guide for Baseline Inference

This guide shows how to run vLLM locally for baseline inference testing.

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support
- HuggingFace account token (for model access)

## Installation

```bash
pip install vllm
```

## Quick Start

### Step 1: Start vLLM Server

**Option A: Using the helper script (recommended)**
```bash
chmod +x start-vllm.sh
./start-vllm.sh Qwen/Qwen3.5-4B
```

**Option B: Manual command**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8080
```

Wait for the message: `"Application startup complete"`

### Step 2: Run Baseline (in another terminal)

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python baseline_inference.py \
    --difficulty all \
    --episodes 10 \
    --model Qwen/Qwen3.5-4B \
    --output baseline_results.json
```

**Expected runtime**: 30-60 minutes (30 episodes total)

## Model Options

We support models in the 3.5B-7B parameter range:

| Model | Params | VRAM (4-bit) | Recommendation |
|-------|--------|--------------|----------------|
| **Qwen/Qwen3.5-4B** | 5B | ~3GB | **Recommended** - Best balance |
| Qwen/Qwen3.5-2B | 2B | ~1.5GB | Faster, lower quality |
| Qwen/Qwen3.5-7B | 7B | ~4GB | Higher quality, slower |

To use a different model:
```bash
./start-vllm.sh Qwen/Qwen3.5-2B
```

## Configuration

The helper script accepts environment variables:

```bash
# Change port
VLLM_PORT=8090 ./start-vllm.sh

# Adjust GPU memory utilization (0.0-1.0)
VLLM_GPU_MEM=0.7 ./start-vllm.sh

# Adjust max sequence length
VLLM_MAX_LEN=2048 ./start-vllm.sh
```

## Monitoring

### Check server health:
```bash
curl http://localhost:8080/health
```

### Check available models:
```bash
curl http://localhost:8080/v1/models
```

### Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### CUDA out of memory
**Solution**: Use a smaller model or reduce GPU memory:
```bash
./start-vllm.sh Qwen/Qwen3.5-2B
# OR
VLLM_GPU_MEM=0.7 ./start-vllm.sh
```

### Port already in use
**Solution**: Use a different port:
```bash
VLLM_PORT=8090 ./start-vllm.sh
export OPENAI_BASE_URL=http://localhost:8090/v1
```

### Model download fails
**Solution**: Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
./start-vllm.sh
```

## Performance Tips

1. **GPU Memory Utilization**: Default 0.8 (80%)
   - Lower (0.6-0.7) if running other GPU processes
   - Can raise to 0.9 for maximum throughput

2. **Max Model Length**: Default 4096 tokens
   - Baseline needs ~1-2k tokens per episode
   - Can lower to 2048 to save memory

3. **Quantization**: vLLM automatically uses optimized kernels
   - No manual configuration needed

## Expected Output

When baseline completes, you should see:

```json
{
  "status": "success",
  "model": "Qwen/Qwen3.5-4B",
  "total_episodes": 30,
  "baseline_scores": {
    "overall": {
      "mean_reward": 0.4567,
      "mean_component": 0.7333,
      "mean_type": 0.4333,
      "mean_window": 0.2567,
      "mean_efficiency": 0.4100
    },
    "by_difficulty": {
      "easy": { "mean_reward": 0.72, ... },
      "medium": { "mean_reward": 0.25, ... },
      "hard": { "mean_reward": 0.40, ... }
    }
  }
}
```

Results are saved to the output JSON file for later analysis.

## Stopping the Server

Find the vLLM process:
```bash
ps aux | grep vllm
```

Kill it:
```bash
kill <PID>
```

Or if using the helper script, it prints the PID:
```bash
kill <PID_from_startup_message>
```
