#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "transformers>=4.45.0",
#     "datasets>=3.0.0",
#     "accelerate>=1.0.0",
#     "vllm>=0.6.0",
#     "torch>=2.0.0",
#     "pydantic>=2.0.0",
# ]
# ///
"""
GRPO Training Script for Log Anomaly Investigation Environment.

This script trains a language model to investigate log files and identify
anomalies using Group Relative Policy Optimization (GRPO).

Usage:
    # Single GPU with colocate mode (recommended for getting started)
    python train_grpo.py --model Qwen/Qwen3-4B --vllm-mode colocate

    # Multi-GPU with vLLM server
    # Terminal 1: Start vLLM server
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-4B --port 8000
    # Terminal 2: Run training
    CUDA_VISIBLE_DEVICES=1 python train_grpo.py --vllm-mode server --vllm-server-url http://localhost:8000

    # With curriculum learning
    python train_grpo.py --curriculum --model Qwen/Qwen3-4B

    # Quick test with small dataset
    python train_grpo.py --model Qwen/Qwen3-0.6B --num-samples 20 --num-generations 2

Requirements:
    pip install -e ".[training]"
    # or
    pip install trl transformers datasets accelerate vllm
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a log anomaly investigation agent with GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model to train (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/grpo_log_anomaly",
        help="Directory to save checkpoints and logs",
    )

    # Dataset configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="Total number of training samples (split across difficulties)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "all", "curriculum"],
        default="all",
        help="Difficulty level(s) to train on",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning (progressive difficulty)",
    )

    # Training configuration
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of completions per prompt for GRPO",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=4096,
        help="Maximum completion length (tokens)",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device training batch size",
    )

    # vLLM configuration
    parser.add_argument(
        "--vllm-mode",
        type=str,
        choices=["colocate", "server"],
        default="colocate",
        help="vLLM mode: colocate (same process) or server (separate)",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL (only used with --vllm-mode server)",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM (slower, but works without GPU)",
    )

    # Logging
    parser.add_argument(
        "--log-completions",
        action="store_true",
        default=True,
        help="Log completions during training",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (optional)",
    )

    return parser.parse_args()


def create_dataset(num_samples: int, difficulty: str):
    """Create the training dataset."""
    from datasets import Dataset
    from training_prompts import create_training_dataset_dict, get_diverse_prompts

    if difficulty == "curriculum":
        # For curriculum learning, we only need prompts - difficulty is auto-selected
        # Use easy prompts as starting point, env will adjust
        prompts = get_diverse_prompts("easy", num_samples)
        return Dataset.from_dict(
            {
                "prompt": prompts,
                "difficulty": ["auto"] * num_samples,  # Will be selected by curriculum
            }
        )

    if difficulty == "all":
        # Equal split across difficulties
        per_diff = num_samples // 3
        data = create_training_dataset_dict(
            num_easy=per_diff,
            num_medium=per_diff,
            num_hard=num_samples - 2 * per_diff,  # Handle remainder
        )
    else:
        # Single difficulty
        prompts = get_diverse_prompts(difficulty, num_samples)
        data = {
            "prompt": prompts,
            "difficulty": [difficulty] * num_samples,
        }

    return Dataset.from_dict(data)


def create_environment_factory(use_curriculum: bool = False):
    """Create the appropriate environment factory."""
    if use_curriculum:
        from training_client import CurriculumLogAnomalyEnv

        return CurriculumLogAnomalyEnv
    else:
        from training_client import LogAnomalyTrainingEnv

        return LogAnomalyTrainingEnv


def create_reward_function():
    """Create the reward function for GRPO."""
    from training_client import LogAnomalyTrainingEnv

    def reward_func(environments: List[LogAnomalyTrainingEnv], **kwargs) -> List[float]:
        """Extract rewards from environment instances."""
        rewards = []
        for env in environments:
            reward = env.reward
            # Log curriculum stats if available
            if hasattr(env, "curriculum_stats"):
                stats = env.curriculum_stats
                logger.debug(
                    f"Curriculum: ep={stats['episode_count']}, "
                    f"diff={stats['current_difficulty']}, "
                    f"success_rate={stats['success_rate']:.2f}"
                )
            rewards.append(reward)
        return rewards

    return reward_func


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("GRPO Training for Log Anomaly Investigation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Difficulty: {args.difficulty}")
    logger.info(f"Curriculum: {args.curriculum}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"vLLM mode: {args.vllm_mode if not args.no_vllm else 'disabled'}")
    logger.info("=" * 60)

    # Import TRL components
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        logger.error(
            "TRL not installed. Install with: pip install -e '.[training]' "
            "or pip install trl transformers datasets accelerate vllm"
        )
        raise SystemExit(1) from e

    # Create dataset
    logger.info("Creating training dataset...")
    dataset = create_dataset(args.num_samples, args.difficulty)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Create environment factory
    use_curriculum = args.curriculum or args.difficulty == "curriculum"
    env_factory = create_environment_factory(use_curriculum)
    logger.info(f"Environment factory: {env_factory.__name__}")

    # Create reward function
    reward_func = create_reward_function()

    # Configure training
    config_kwargs = {
        "output_dir": args.output_dir,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "log_completions": args.log_completions,
        "logging_steps": 1,
        "save_steps": 100,
        "save_total_limit": 3,
        # Disable thinking mode for cleaner tool calls
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # vLLM configuration
    if not args.no_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "server":
            config_kwargs["vllm_server_url"] = args.vllm_server_url

    # W&B configuration
    if args.wandb_project:
        config_kwargs["report_to"] = ["wandb"]
        os.environ["WANDB_PROJECT"] = args.wandb_project
    else:
        config_kwargs["report_to"] = []

    config = GRPOConfig(**config_kwargs)

    # Create trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=config,
        environment_factory=env_factory,
    )

    # Train!
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training complete!")

        # Save final model
        final_path = os.path.join(args.output_dir, "final")
        trainer.save_model(final_path)
        logger.info(f"Final model saved to: {final_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        interrupt_path = os.path.join(args.output_dir, "interrupted")
        trainer.save_model(interrupt_path)
        logger.info(f"Interrupted model saved to: {interrupt_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
