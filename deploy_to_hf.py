#!/usr/bin/env python3
"""
Deploy Log Anomaly Environment to HuggingFace Spaces.

Usage:
    # First time setup
    uv run python deploy_to_hf.py --create

    # Update existing Space
    uv run python deploy_to_hf.py

    # With custom repo name
    uv run python deploy_to_hf.py --repo-id your-username/log-anomaly-env
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace Spaces")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SPACE_ID", "garg-tejas/log-anomaly-env"),
        help="HuggingFace Space repo ID (default: garg-tejas/log-anomaly-env)",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create new Space (first time only)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Space private",
    )
    args = parser.parse_args()

    # Get token from environment
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return 1

    api = HfApi(token=token)
    repo_id = args.repo_id

    # Step 1: Create Space if requested
    if args.create:
        print(f"Creating Space: {repo_id}")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                private=args.private,
            )
            print(f"Space created: https://huggingface.co/spaces/{repo_id}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Space already exists: {repo_id}")
            else:
                raise

    # Step 2: Upload all files
    print(f"\nUploading files to: {repo_id}")

    # Files/folders to exclude from upload
    ignore_patterns = [
        ".git",
        ".git/*",
        "__pycache__",
        "__pycache__/*",
        "*.pyc",
        ".env",
        ".venv",
        ".venv/*",
        "venv",
        "venv/*",
        "outputs/*",
        "*.log",
        ".mypy_cache",
        ".mypy_cache/*",
        ".pytest_cache",
        ".pytest_cache/*",
        ".ruff_cache",
        ".ruff_cache/*",
        "deploy_to_hf.py",  # Don't upload this script
    ]

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=".",
        ignore_patterns=ignore_patterns,
    )
    print("Upload complete!")

    # Step 3: Check runtime status
    print("\nChecking Space status...")
    try:
        runtime = api.get_space_runtime(repo_id=repo_id)
        print(f"  Stage: {runtime.stage}")
        print(f"  Hardware: {runtime.hardware}")
        if runtime.stage == "RUNNING":
            print(f"\nSpace is live at: https://huggingface.co/spaces/{repo_id}")
            print(f"API endpoint: https://{repo_id.replace('/', '-')}.hf.space")
        elif runtime.stage == "BUILDING":
            print("\nSpace is building... Check status at:")
            print(f"  https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Could not get runtime status: {e}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
