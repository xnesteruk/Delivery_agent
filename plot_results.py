"""Plot repeated-run results for a single experiment batch."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def results(folder: Path) -> None:
    """Save an on-time letter count and simulation-time plot."""
    folder = Path(folder)

    with (folder / "runs.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No run rows found in {folder / 'runs.csv'}")
    with (folder / "settings.json").open(encoding="utf-8") as file:
        settings = json.load(file)

    width, height = settings["config"]["map_width"], settings["config"]["map_height"]
    if width == height:
        title=f"Map {width} × {height}: repeated simulation results"

    total_letters = settings["generation"]["letter_count"]
    run_numbers = [int(row["run"]) for row in rows]
    on_time = [int(row["on_time_letters"]) for row in rows]
    simulation_minutes = [int(row["simulation_minutes"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    axes[0].bar(run_numbers, on_time, color="#377eb8")
    axes[0].set(ylabel="Letters delivered on time", ylim=(0, total_letters + 1), title=title)
    axes[1].plot(run_numbers, simulation_minutes, "o-", color="#7570b3")
    axes[1].set(xlabel="Run", ylabel="Simulation time (min)")
    for axis in axes:
        axis.set_xticks(run_numbers)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.savefig(folder / "results.png", dpi=150)
    plt.close(fig)
    print("Graph saved to:", folder / "results.png")


plot_results = results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    results(parser.parse_args().folder)
