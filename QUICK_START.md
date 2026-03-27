# Quick Start: Run Baseline on Lightning AI

Follow these steps to run the baseline inference with vLLM.

## 1. Setup on Lightning AI

```bash
# Clone your repo (if not already)
git clone <your-repo-url>
cd log-anomaly

# Install dependencies
pip install -r requirements.txt
pip install vllm

# Set HuggingFace token
export HF_TOKEN="your_token_here"
```

## 2. Start vLLM Server

**Terminal 1:**
```bash
chmod +x start-vllm.sh
./start-vllm.sh Qwen/Qwen3.5-4B
```

Wait for: `"✓ vLLM server is ready!"`

## 3. Run Baseline

**Terminal 2:**
```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

python baseline_inference.py \
    --difficulty all \
    --episodes 10 \
    --model Qwen/Qwen3.5-4B \
    --output baseline_before_enhancement.json
```

## 4. Results

Expected runtime: **30-60 minutes** (30 episodes total)

Results will be saved to `baseline_before_enhancement.json`

Share back:
- Overall mean reward
- Breakdown by difficulty
- Any errors encountered

## Troubleshooting

See [VLLM_SETUP.md](VLLM_SETUP.md) for detailed troubleshooting.

Quick fixes:
- **Out of memory**: Use `Qwen/Qwen3.5-2B` instead
- **Port in use**: `VLLM_PORT=8090 ./start-vllm.sh`
- **GPU not found**: Check `nvidia-smi`
