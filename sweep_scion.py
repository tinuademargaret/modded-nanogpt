"""
Wandb sweep for tuning Scion optimizer hyperparameters.

Two-stage tuning strategy:
  Stage 1 ("scales"):  Fix lr & momentum at defaults, sweep all 5 per-group scale factors.
                        This finds the right constraint geometry (relative radii).
  Stage 2 ("dynamics"): Fix scales at the best values from Stage 1, sweep lr & momentum.
                        This finds the right optimization speed for that geometry.

Usage:
    # Stage 1: tune scales (use best scales in Stage 2 via --defaults)
    python sweep_scion.py --stage scales

    # Stage 2: tune lr & momentum with best scales from Stage 1
    python sweep_scion.py --stage dynamics \
        --defaults spectral_scale=60 lm_head_scale=4000 embed_scale=2500 \
                   col_norm_scale=15 rms_norm_scale=8

    # Or sweep everything at once (joint, less efficient but simpler)
    python sweep_scion.py --stage all

    # Resume a previous sweep
    python sweep_scion.py --stage scales --sweep-id <entity/project/sweep_id>

    # Customize GPU count
    python sweep_scion.py --stage scales --nproc 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import wandb

# ---------------------------------------------------------------------------
# Per-stage sweep configurations
# ---------------------------------------------------------------------------

SCALES_SWEEP = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,
        "eta": 3,
        "s": 2,
    },
    "parameters": {
        "spectral_scale": {
            "distribution": "log_uniform_values",
            "min": 10,
            "max": 200,
        },
        "lm_head_scale": {
            "distribution": "log_uniform_values",
            "min": 500,
            "max": 10000,
        },
        "embed_scale": {
            "distribution": "log_uniform_values",
            "min": 500,
            "max": 10000,
        },
        "col_norm_scale": {
            "distribution": "log_uniform_values",
            "min": 1,
            "max": 200,
        },
        "rms_norm_scale": {
            "distribution": "log_uniform_values",
            "min": 1,
            "max": 200,
        },
    },
}

DYNAMICS_SWEEP = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,
        "eta": 3,
        "s": 2,
    },
    "parameters": {
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
    },
}

ALL_SWEEP = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,
        "eta": 3,
        "s": 2,
    },
    "parameters": {
        "scion_lr": {
            "distribution": "log_uniform_values",
            "min": 2**-15,
            "max": 2**-8,
        },
        "scion_momentum": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.5,
        },
        "spectral_scale": {
            "distribution": "log_uniform_values",
            "min": 10,
            "max": 200,
        },
        "lm_head_scale": {
            "distribution": "log_uniform_values",
            "min": 500,
            "max": 10000,
        },
        "embed_scale": {
            "distribution": "log_uniform_values",
            "min": 500,
            "max": 10000,
        },
        "col_norm_scale": {
            "distribution": "log_uniform_values",
            "min": 1,
            "max": 200,
        },
        "rms_norm_scale": {
            "distribution": "log_uniform_values",
            "min": 1,
            "max": 200,
        },
    },
}

STAGE_CONFIGS = {
    "scales": SCALES_SWEEP,
    "dynamics": DYNAMICS_SWEEP,
    "all": ALL_SWEEP,
}

# Defaults for parameters not being swept (used when the key is absent from wandb config)
DEFAULTS = {
    "scion_lr": 2**-12,
    "scion_momentum": 0.1,
    "spectral_scale": 50,
    "lm_head_scale": 3000,
    "embed_scale": 3000,
    "col_norm_scale": 10,
    "rms_norm_scale": 10,
}

# Maps sweep param names -> environment variable names consumed by train_gpt.py
PARAM_TO_ENV = {
    "scion_lr": "SCION_LR",
    "scion_momentum": "SCION_MOMENTUM",
    "spectral_scale": "SCION_SPECTRAL_SCALE",
    "lm_head_scale": "SCION_LM_HEAD_SCALE",
    "embed_scale": "SCION_EMBED_SCALE",
    "col_norm_scale": "SCION_COL_NORM_SCALE",
    "rms_norm_scale": "SCION_RMS_NORM_SCALE",
}

# ---------------------------------------------------------------------------
# Training launcher
# ---------------------------------------------------------------------------


def train(nproc: int = 1, defaults: dict = None):
    """Called by wandb agent for each trial.

    wandb is only initialized once here in the parent process.
    The child (train_gpt.py) writes metrics to a JSON-lines file
    via SWEEP_METRICS_FILE, which the parent reads back and logs to wandb.
    """
    merged_defaults = {**DEFAULTS, **(defaults or {})}

    run = wandb.init()
    config = wandb.config

    # Build environment with Scion hyperparameters
    env = os.environ.copy()
    env["WANDB_SWEEP"] = "1"

    for param_name, env_var in PARAM_TO_ENV.items():
        value = config.get(param_name, merged_defaults[param_name])
        env[env_var] = str(value)

    # Tell the child to write metrics to a temp file instead of using wandb
    metrics_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", prefix="sweep_metrics_", delete=False
    )
    metrics_path = metrics_file.name
    metrics_file.close()
    env["SWEEP_METRICS_FILE"] = metrics_path

    # Prevent the child from initializing wandb itself
    env.pop("WANDB_RESUME", None)
    env.pop("WANDB_RUN_ID", None)

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "train_gpt.py",
    ]

    summary_parts = [f"{k}={env[v]}" for k, v in PARAM_TO_ENV.items()]
    print(f"[sweep] Launching: {' '.join(cmd)}")
    print(f"[sweep] {' '.join(summary_parts)}")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Read metrics written by the child and log them to the wandb run
    try:
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    step = entry.pop("step", None)
                    wandb.log(entry, step=step)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[sweep] Warning: could not read metrics file: {e}")
    finally:
        try:
            os.unlink(metrics_path)
        except OSError:
            pass

    if result.returncode != 0:
        print(f"[sweep] Training failed with return code {result.returncode}")

    wandb.finish(exit_code=0 if result.returncode == 0 else 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_defaults(items: list[str] | None) -> dict:
    """Parse key=value pairs from --defaults into a dict."""
    if not items:
        return {}
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid default format '{item}', expected key=value")
        key, value = item.split("=", 1)
        if key not in DEFAULTS:
            raise ValueError(
                f"Unknown parameter '{key}'. Valid: {list(DEFAULTS.keys())}"
            )
        result[key] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser(description="Wandb sweep for Scion optimizer")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["scales", "dynamics", "all"],
        default="scales",
        help="Which stage to sweep: 'scales' (5 scale factors), "
        "'dynamics' (lr + momentum), or 'all' (everything jointly)",
    )
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
        help="Wandb entity (team or username)",
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
    parser.add_argument(
        "--defaults",
        nargs="*",
        metavar="KEY=VALUE",
        help="Override default values for non-swept parameters, e.g. "
        "--defaults spectral_scale=60 rms_norm_scale=8",
    )
    args = parser.parse_args()

    user_defaults = parse_defaults(args.defaults)
    sweep_config = STAGE_CONFIGS[args.stage]

    # Create or resume sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        parts = sweep_id.split("/")
        if len(parts) == 3:
            args.entity, args.project = parts[0], parts[1]
        print(f"[sweep] Resuming sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project,
            entity=args.entity,
        )
        print(f"[sweep] Created sweep: {sweep_id}")

    print(f"[sweep] Stage: {args.stage}")
    if user_defaults:
        print(f"[sweep] Overridden defaults: {user_defaults}")

    if args.create_only:
        print(f"[sweep] Sweep created. Run agents with:")
        print(f"  wandb agent {sweep_id}")
        return

    # Run agent
    wandb.agent(
        sweep_id,
        function=lambda: train(nproc=args.nproc, defaults=user_defaults),
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
