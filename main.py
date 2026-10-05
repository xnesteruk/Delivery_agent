import argparse
import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from agents.delivery_agent import DeliveryAgent
from experiments.scenario_generator import ScenarioGenerator
from experiments.seeds import make_run_seeds
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.evaluation import DeliveryEvaluator
from simulation.result_exporter import ResultExporter
from simulation.runner import Simulation

def create_experiment_folder(results_path: Path) -> Path:
    results_path.mkdir(parents=True, exist_ok=True)

    numbers = [
        int(path.name.removeprefix("experiment_"))
        for path in results_path.iterdir()
        if path.is_dir()
           and path.name.startswith("experiment_")
           and path.name.removeprefix("experiment_").isdigit()
    ]

    number = max(numbers, default=0) + 1

    while True:
        experiment_path = results_path / f"experiment_{number:03d}"

        try:
            experiment_path.mkdir(exist_ok=False)
            return experiment_path
        except FileExistsError:
            number += 1

def run_experiment(config_path: Path, results_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as file:
        settings = json.load(file)
    # Accept both the original flat config and the new experiment presets.
    config = SimulationConfig(**settings.get("config", settings))
    post_settings = {"count": settings.get("pickup_posts", {}).get("count", 3)}
    master_seed = config.seed
    run_count = settings.get("run_count", 20)
    max_steps = settings.get("max_steps", 2000)
    closure_memory_minutes = settings.get("closure_memory_minutes", 12)
    if type(run_count) is not int or run_count < 1:
        raise ValueError("Run count must be a positive integer.")
    generation_settings = settings.get("generation", {
        "letter_count": 10, "closure_count": 4,
        "arrival_window_minutes": 60,
        "min_closure_minutes": 6, "max_closure_minutes": 24,
    })
    blocked_cells = {tuple(cell) for cell in settings.get("blocked_cells", [])}

    generator = ScenarioGenerator(
        config=config,
        blocked_cells=blocked_cells,
    )

    run_seeds = make_run_seeds(master_seed, run_count)

    batch_path = create_experiment_folder(results_path)

    experiment_settings = {
        "preset": config_path.name,
        "master_seed": master_seed,
        "run_count": run_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_seeds": run_seeds,
        "max_steps": max_steps,
        "closure_memory_minutes": closure_memory_minutes,
        "python_version": platform.python_version(),
        "config": {
            "map_width": config.map_width,
            "map_height": config.map_height,
            "carrying_capacity": config.carrying_capacity,
            "delivery_allowance_minutes": (
                config.delivery_allowance_minutes
            ),
            "seed": config.seed,
        },
        "fixed_rules": {
            "move_minutes": config.move_minutes,
            "cell_distance_meters": config.cell_distance_meters,
            "deadline_starts_at": "pickup",
            "post_capacity": "unlimited",
            "on_time_rate_denominator": "all_scenario_letters",
        },
        "generation": generation_settings,
        "pickup_posts": post_settings,
        "blocked_cells": sorted(blocked_cells),
    }

    with (batch_path / "settings.json").open(
            "w", encoding="utf-8"
    ) as file:
        json.dump(
            experiment_settings,
            file,
            ensure_ascii=False,
            indent=2,
        )

    rows = []

    for run_number, run_seed in enumerate(run_seeds, start=1):
        scenario = generator.generate(
            seed=run_seed,
            post_count=post_settings["count"],
            **generation_settings,
        )

        letters = scenario.create_letters()

        environment = Environment(
            config=config,
            letters=letters,
            blocked_cells=set(scenario.blocked_cells),
            scheduled_closures=list(scenario.closures),
            pickup_posts=scenario.pickup_posts,
            arrivals_seed=scenario.arrivals_seed,
        )

        agent = DeliveryAgent(
            map_width=config.map_width,
            map_height=config.map_height,
            blocked_cells=set(scenario.blocked_cells),
            move_minutes=config.move_minutes,
            closure_memory_minutes=closure_memory_minutes,
        )

        simulation = Simulation(
            environment=environment,
            agent=agent,
            max_steps=max_steps,
        )

        result = simulation.run()

        metrics = DeliveryEvaluator.evaluate(
            letters=letters,
            current_time=result.simulation_minutes,
        )

        run_path = batch_path / f"run_{run_number:03d}"
        run_path.mkdir()

        # Preserve the exact generated scenario.
        with (run_path / "scenario.json").open(
                "w", encoding="utf-8"
        ) as file:
            json.dump(
                asdict(scenario),
                file,
                ensure_ascii=False,
                indent=2,
            )

        ResultExporter.save_history(
            result,
            run_path / "history.csv",
            )
        ResultExporter.save_letters(
            letters,
            run_path / "letters.csv",
            )

        row = {
            "run": run_number,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "simulation_minutes": result.simulation_minutes,
            "execution_seconds": result.execution_seconds,
        }
        # Four disjoint outcomes account for every scheduled letter.
        row.update({
            "on_time_letters": metrics.on_time_letters,
            "awaiting_delivery_letters": metrics.undelivered_letters,
            "not_yet_appeared_letters": metrics.future_letters,
            "on_time_rate": metrics.on_time_rate,
            "total_lateness_minutes": metrics.total_lateness_minutes,
        })
        rows.append(row)

        print(
            f"Run {run_number:02}/{run_count} | "
            f"Delivered: {metrics.delivered_letters}/"
            f"{metrics.total_letters} | "
            f"On time: {metrics.on_time_rate:.1%} | "
            f"Finished: {result.finished}"
        )

    with (batch_path / "runs.csv").open(
            "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "on_time_rate",
        "awaiting_delivery_letters",
        "not_yet_appeared_letters",
        "on_time_letters",
        "total_lateness_minutes",
        "steps",
        "simulation_minutes",
        "execution_seconds",
    ]

    aggregate_rows = []

    for name in metric_names:
        values = [row[name] for row in rows]

        aggregate_rows.append({
            "metric": name,
            "mean": mean(values),
            "stddev": stdev(values) if len(values) > 1 else 0.0,
        })

    with (batch_path / "aggregate.csv").open(
            "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "mean", "stddev"],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    finished_runs = sum(row["stop_reason"] == "all_delivered" for row in rows)

    print(f"\nCompleted runs: {finished_runs}/{run_count}")
    print(
        "Mean on-time rate:",
        f"{mean(row['on_time_rate'] for row in rows):.1%}",
    )
    print("Results saved to:", batch_path)
    return batch_path


def main():
    project_path = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run a delivery experiment.")
    parser.add_argument("--config", type=Path,
                        default=project_path / "configs" / "default.json")
    args = parser.parse_args()
    batch_path = run_experiment(args.config, project_path / "results")

    from plot_results import results
    results(batch_path)


if __name__ == "__main__":
    main()
