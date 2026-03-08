#!/usr/bin/env python3
"""Plot Reward and Steps vs Games from TensorBoard event files."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_run_folder(logdir: str, run_name: str | None) -> str:
    """Find and return the path to the run folder to use."""
    if run_name:
        run_path = os.path.join(logdir, run_name)
        if not os.path.isdir(run_path):
            available = sorted(
                d for d in os.listdir(logdir)
                if os.path.isdir(os.path.join(logdir, d))
            )
            print(f"Error: run '{run_name}' not found in {logdir}/")
            print("Available runs:")
            for d in available:
                print(f"  {d}")
            raise SystemExit(1)
        return run_path

    # Find the most recently modified run subfolder
    subdirs = [
        os.path.join(logdir, d) for d in os.listdir(logdir)
        if os.path.isdir(os.path.join(logdir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No run folders found in {logdir}/")
    return max(subdirs, key=os.path.getmtime)


def load_scalars(run_dir: str, tag: str) -> tuple[list[int], list[float]]:
    """Load scalar values from event files in a single run directory."""
    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")
    event_file_dir = os.path.dirname(max(event_files, key=os.path.getmtime))
    ea = EventAccumulator(event_file_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    games = list(range(1, len(events) + 1))
    values = [e.value for e in events]
    return games, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="runs", help="TensorBoard log directory")
    parser.add_argument("--run", default=None, help="Run folder name (default: most recent)")
    parser.add_argument("--output", default="Ch6/training_plots.png", help="Output image path")
    args = parser.parse_args()

    run_dir = find_run_folder(args.logdir, args.run)
    print(f"Using run: {os.path.basename(run_dir)}")

    games_r, rewards = load_scalars(run_dir, "reward")
    games_s, steps = load_scalars(run_dir, "steps")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(games_r, rewards, color="black", linewidth=0.8)
    ax1.set_xlabel("Games")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward vs Games")
    ax1.grid(True)

    ax2.plot(games_s, steps, color="black", linewidth=0.8)
    ax2.set_xlabel("Games")
    ax2.set_ylabel("Steps")
    ax2.set_title("Steps vs Games")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
