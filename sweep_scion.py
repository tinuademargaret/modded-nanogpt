"""
Wandb sweep for tuning Scion optimizer hyperparameters.

Usage:
    # Create the sweep and start an agent (single GPU)
    python sweep_scion.py

    # Or, create the sweep first, then run agents separately
    python sweep_scion.py --create-only
    wandb agent <sweep_id>

    # Resume a previous sweep
    python sweep_scion.py --sweep-id <entity/project/sweep_id>

    # Customize GPU count
    python sweep_scion.py --nproc 8
"""

import argparse
import os
import subprocess
import sys

import wandb

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
SWEEP_CONFIG = {
    "method": "bayes",  # bayesian optimization
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        # Primary sweep targets: lr and momentum
        "scion_lr": {
            # Log-uniform over [2^-15, 2^-8] — centered around current 2^-12
            "distribution": "log_uniform_values",
            "min": 2**-15,
            "max": 2**-8,
        },
        "scion_momentum": {
            # Uniform over [0.01, 0.5] — current is 0.1
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.5,
        },
        # Optional: per-group scale factors (fixed by default, uncomment to sweep)
        # "spectral_scale": {
        #     "distribution": "log_uniform_values",
        #     "min": 10,
        #     "max": 200,
        # },
        # "lm_head_scale": {
        #     "distribution": "log_uniform_values",
        #     "min": 500,
        #     "max": 10000,
        # },
        # "embed_scale": {
        #     "distribution": "log_uniform_values",
        #     "min": 500,
        #     "max": 10000,
        # },
        # "scalar_scale": {
        #     "distribution": "log_uniform_values",
        #     "min": 1,
        #     "max": 200,
        # },
    },
}

# Defaults for parameters not being swept (used when the key is absent from wandb config)
DEFAULTS = {
    "spectral_scale": 50,
    "lm_head_scale": 3000,
    "embed_scale": 3000,
    "scalar_scale": 50,
}

# ---------------------------------------------------------------------------
# Training launcher
# ---------------------------------------------------------------------------


def train(nproc: int = 1):
    """Called by wandb agent for each trial."""
    run = wandb.init()
    config = wandb.config

    # Build environment with Scion hyperparameters
    env = os.environ.copy()
    env["WANDB_SWEEP"] = "1"
    env["SCION_LR"] = str(config.get("scion_lr", 2**-12))
    env["SCION_MOMENTUM"] = str(config.get("scion_momentum", 0.1))
    env["SCION_SPECTRAL_SCALE"] = str(
        config.get("spectral_scale", DEFAULTS["spectral_scale"])
    )
    env["SCION_LM_HEAD_SCALE"] = str(
        config.get("lm_head_scale", DEFAULTS["lm_head_scale"])
    )
    env["SCION_EMBED_SCALE"] = str(config.get("embed_scale", DEFAULTS["embed_scale"]))
    env["SCION_SCALAR_SCALE"] = str(
        config.get("scalar_scale", DEFAULTS["scalar_scale"])
    )

    # Inherit WANDB_RUN_ID so the training process logs to the same run
    env["WANDB_RUN_ID"] = run.id
    env["WANDB_RESUME"] = "allow"

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "train_gpt.py",
    ]

    print(f"[sweep] Launching: {' '.join(cmd)}")
    print(f"[sweep] lr={env['SCION_LR']} momentum={env['SCION_MOMENTUM']}")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        print(f"[sweep] Training failed with return code {result.returncode}")
        wandb.finish(exit_code=1)
    else:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Wandb sweep for Scion optimizer")
    parser.add_argument(
        "--project",
        type=str,
        default="scion-nanogpt-sweep",
        help="Wandb project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Wandb entity (team or username). Defaults to your default entity.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Resume an existing sweep by ID (entity/project/sweep_id)",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create the sweep, don't start an agent",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of trials to run (default: 30)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of GPUs per trial (default: 1)",
    )
    args = parser.parse_args()

    # Create or resume sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"[sweep] Resuming sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            sweep=SWEEP_CONFIG,
            project=args.project,
            entity=args.entity,
        )
        print(f"[sweep] Created sweep: {sweep_id}")

    if args.create_only:
        print(f"[sweep] Sweep created. Run agents with:")
        print(f"  wandb agent {sweep_id}")
        return

    # Run agent
    wandb.agent(
        sweep_id,
        function=lambda: train(nproc=args.nproc),
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
