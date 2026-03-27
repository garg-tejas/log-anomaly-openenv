# Debug Script Execution Instructions

This guide explains how to run the debug script to capture raw model responses and identify parsing failures.

## Purpose

The `debug_model_responses.py` script captures:
- **Raw model outputs** before parsing
- **Parsing success/failure** for each step
- **Fallback triggers** when default commands are used
- **Command history patterns** (repetition detection)

This helps us understand **why 67% of episodes use fallback** (20 out of 30 hit the 15-step limit).

---

## Quick Start on Lightning AI

### Step 1: Start Environment Server

In **Terminal 1**:

```bash
cd log-anomaly
git pull  # Get latest code with debug script
python -m server.app
```

Expected output:
```
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start vLLM Server

In **Terminal 2**:

```bash
cd log-anomaly
vllm serve Qwen/Qwen3.5-4B \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype auto
```

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Step 3: Run Debug Script

In **Terminal 3**:

```bash
cd log-anomaly
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=dummy

# Run debug experiments (7 episodes total, ~7-10 minutes)
python debug_model_responses.py --difficulty easy --episodes 3 --output debug_easy.json
python debug_model_responses.py --difficulty medium --episodes 2 --output debug_medium.json
python debug_model_responses.py --difficulty hard --episodes 2 --output debug_hard.json
```

---

## Understanding the Output

### Console Report

The script prints a human-readable report:

```
================================================================================
MODEL RESPONSE DEBUG REPORT
================================================================================

📊 SUMMARY:
  Total steps executed: 42
  Successful parses: 15
  Fallback parses: 27
  Fallback rate: 64.29%

🚨 COMMON FAILURE REASONS:
  - No 'Command:' or 'Submit:' prefix found: 18 times
  - Missing timestamps in submission: 5 times
  - Command matches default fallback pattern: 4 times

❌ PARSING FAILURES (First 5):

  --- Failure #1 ---
  Episode: abc123
  Difficulty: easy
  Step: 3
  Reason: No 'Command:' or 'Submit:' prefix found
  Steps remaining: 12

  Raw model output:
  Let me check for error patterns in the log file. I'll search for ERROR entries
  and count them by component to identify which service is having issues.

  grep ERROR log.txt | awk '{print $3}' | sort | uniq -c | sort -rn
  ... [truncated]

✅ SUCCESSFUL PARSES (First 3):

  --- Success #1 ---
  Episode: def456
  Difficulty: easy
  Step: 1
  Action type: bash

  Raw model output:
  Command: wc -l log.txt && head -10 log.txt

  Extracted:
  wc -l log.txt && head -10 log.txt
```

### Key Patterns to Look For

1. **Missing Prefix Problem**
   - Model outputs: `"Let me check the logs: grep ERROR log.txt"`
   - Expected format: `"Command: grep ERROR log.txt"`
   - **Fix**: Add explicit FORMAT section in prompt

2. **Never Submitting**
   - Model always says "I need more information"
   - Never generates `Submit:` prefix
   - **Fix**: Add step budget awareness, few-shot submit examples

3. **Command Repetition**
   - Same command executed multiple times (e.g., 7 times)
   - Happens when output is unparseable → fallback → repeat
   - **Fix**: Show command history with duplicate warnings

4. **Malformed JSON**
   - Model generates: `Submit: {anomaly_type: error_spike, ...}` (missing quotes)
   - Parser fails, uses default submission
   - **Fix**: Use structured output (Outlines library)

---

## JSON Output Structure

Each debug JSON file contains:

```json
{
  "metadata": {
    "model": "Qwen/Qwen3.5-4B",
    "difficulty": "easy",
    "episodes": 3,
    "generated_at": "2026-03-27T10:30:00"
  },
  "analysis": {
    "summary": {
      "total_steps": 42,
      "fallback_steps": 27,
      "fallback_rate": 64.29
    },
    "parsing_failures": [...],
    "successful_parses": [...]
  },
  "raw_debug_logs": [
    [
      {
        "episode_id": "abc123",
        "difficulty": "easy",
        "started_at": "2026-03-27T10:25:00"
      },
      {
        "step": 1,
        "raw_model_output": "Command: wc -l log.txt",
        "prompt_context": {
          "steps_remaining": 15,
          "command_history_length": 0
        },
        "parsing": {
          "action_type": "bash",
          "used_fallback": false,
          "extracted_command": "wc -l log.txt"
        }
      },
      ...
    ]
  ]
}
```

---

## After Running: Next Steps

### 1. Review the Console Reports

Look for the **top failure reasons** in each difficulty level:

```bash
# Quick analysis
echo "=== EASY ==="
grep "COMMON FAILURE REASONS" -A 5 debug_easy.json

echo "=== MEDIUM ==="
grep "COMMON FAILURE REASONS" -A 5 debug_medium.json

echo "=== HARD ==="
grep "COMMON FAILURE REASONS" -A 5 debug_hard.json
```

### 2. Identify Root Causes

Common patterns we're looking for:

| Pattern | Root Cause | Fix Priority |
|---------|-----------|--------------|
| No 'Command:' prefix | Model adds explanations before command | High - Fix prompting |
| Missing timestamps | Model generates partial JSON | High - Structured output |
| Command repetition | Parser fails → same fallback command | High - Command history |
| Never submits | Model too cautious, no budget awareness | Medium - Add step warnings |
| Wrong format JSON | Model forgets quotes or structure | High - Outlines schema |

### 3. Implement Fixes Based on Findings

**If fallback rate > 50%**: Implement structured output (Outlines) first
**If repetition > 30%**: Add command history display  
**If submissions empty**: Enhance prompting with submission examples

---

## Expected Timeline

- **Environment server**: Instant startup
- **vLLM server**: ~2-3 minutes to load model
- **Debug runs**: ~7-10 minutes total
  - Easy (3 episodes): ~3 minutes
  - Medium (2 episodes): ~2-3 minutes
  - Hard (2 episodes): ~2-3 minutes

Total time: **~12-15 minutes** from start to analysis.

---

## Troubleshooting

### vLLM won't start

```bash
# Check GPU availability
nvidia-smi

# If OOM, reduce memory usage
vllm serve Qwen/Qwen3.5-4B --gpu-memory-utilization 0.8 --max-model-len 4096
```

### Environment server connection refused

```bash
# Check if port 8000 is already in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill existing process if needed
kill -9 <PID>
```

### Debug script fails with import errors

```bash
# Make sure you're in the log-anomaly directory
cd log-anomaly
pwd

# Verify environment setup
pip list | grep openenv
python -c "from baseline_inference import ReactAgent; print('OK')"
```

---

## Files Generated

After completion, you'll have:

```
log-anomaly/
├── debug_easy.json       (~50-100 KB)
├── debug_medium.json     (~30-60 KB)
├── debug_hard.json       (~30-60 KB)
└── vllm_server.log       (auto-generated, in .gitignore)
```

**Important**: The debug JSON files are in `.gitignore` (pattern: `baseline_*.json` doesn't match, but we may want to add `debug_*.json`). 

If you want to share results, copy them locally:

```bash
# On Lightning AI
cat debug_easy.json | head -100  # Preview

# Or download via Lightning AI file browser
# Right-click → Download
```

---

## What Happens Next

1. **Analyze the output** to identify top failure patterns
2. **Implement targeted fixes**:
   - Structured output with Outlines (biggest impact expected)
   - Enhanced prompting with format clarity
   - Command history display with duplicate warnings
3. **Re-run baseline** to measure improvement (expect fallback rate: 67% → 15-20%)
4. **Iterate** until parsing is stable
5. **Implement reward enhancements** once parsing is reliable
6. **Train with GRPO** to improve investigation strategy

---

## Questions?

If you encounter issues or unexpected patterns:

1. Check the raw model outputs in the JSON files
2. Look for patterns in successful vs failed parses
3. Consider whether the model needs:
   - Clearer format instructions
   - More examples (few-shot)
   - Structured output constraints
   - Better feedback (command history, step budget)

The goal is to **understand the model's natural output format** so we can either:
- **Adapt the parser** to match what the model generates, OR
- **Constrain the model** to generate parseable output (preferred for reliability)
