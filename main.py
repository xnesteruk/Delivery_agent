from pathlib import Path

from agents.delivery_agent import DeliveryAgent
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.models import Letter, Position


def run_scenario(
    name: str,
    config: SimulationConfig,
    letters: list[Letter],
    blocked_cells: set[tuple[int, int]],
    max_steps: int = 100,
) -> Environment:
    environment = Environment(
        config=config,
        letters=letters,
        blocked_cells=blocked_cells,
    )

    agent = DeliveryAgent(
        map_width=config.map_width,
        map_height=config.map_height,
        blocked_cells=blocked_cells,
        move_minutes=config.move_minutes,
    )

    print(f"\n{name}")

    for step in range(max_steps):
        if environment.is_finished():
            break

        observation = environment.get_observation()
        action = agent.choose_action(observation)
        result = environment.step(action)

        print(
            f"Step: {step + 1:2} | "
            f"Time: {result.current_time:2} | "
            f"Position: {result.courier_position} | "
            f"{result.last_action_result}"
        )

    print("Finished:", environment.is_finished())
    print("Total time:", environment.current_time)

    for letter in letters:
        print(
            f"Letter {letter.letter_id}: "
            f"deadline={letter.deadline}, "
            f"delivered={letter.delivery_time}, "
            f"lateness={letter.lateness_minutes}"
        )

    if not environment.is_finished():
        print("Stopped: action limit reached.")

    return environment


def main():
    config_path = (
        Path(__file__).resolve().parent / "configs" / "default.json"
    )
    config = SimulationConfig.from_json(config_path)

    if config.map_width != 5 or config.map_height != 5:
        raise ValueError("These demo scenarios require a 5x5 map.")

    if config.carrying_capacity < 3:
        raise ValueError("These demos require capacity of at least 3.")

    # 1. Deliver all three letters.
    letters = [
        Letter(
            letter_id=letter_id,
            destination=destination,
            available_time=0,
            delivery_allowance_minutes=config.delivery_allowance_minutes,
        )
        for letter_id, destination in [
            (1, Position(3, 2)),
            (2, Position(3, 3)),
            (3, Position(2, 3)),
        ]
    ]

    environment = run_scenario(
        name="Scenario 1: three deliveries",
        config=config,
        letters=letters,
        blocked_cells=set(),
    )

    assert environment.is_finished()
    assert all(letter.is_delivered for letter in letters)
    assert (
        environment.get_observation().remaining_capacity
        == config.carrying_capacity
    )

    # 2. Find a detour around a building.
    letters = [
        Letter(
            letter_id=1,
            destination=Position(4, 2),
            available_time=0,
            delivery_allowance_minutes=config.delivery_allowance_minutes,
        )
    ]

    environment = run_scenario(
        name="Scenario 2: detour",
        config=config,
        letters=letters,
        blocked_cells={(3, 2)},
    )

    assert environment.is_finished()
    assert environment.current_time == 4 * config.move_minutes

    # 3. Do not collect a letter with an unreachable address.
    letters = [
        Letter(
            letter_id=1,
            destination=Position(4, 2),
            available_time=0,
            delivery_allowance_minutes=config.delivery_allowance_minutes,
        )
    ]

    environment = run_scenario(
        name="Scenario 3: unreachable address",
        config=config,
        letters=letters,
        blocked_cells={(3, 2), (4, 1), (4, 3)},
        max_steps=5,
    )

    assert not environment.is_finished()
    assert not letters[0].is_picked_up
    assert environment.current_time == 5

    # 4. The farther letter has a much tighter deadline.
    regular_letter = Letter(
        letter_id=1,
        destination=Position(1, 2),
        available_time=0,
        delivery_allowance_minutes=120,
    )

    urgent_letter = Letter(
        letter_id=2,
        destination=Position(4, 2),
        available_time=0,
        delivery_allowance_minutes=15,
    )

    environment = run_scenario(
        name="Scenario 4: urgent delivery first",
        config=config,
        letters=[regular_letter, urgent_letter],
        blocked_cells=set(),
    )

    assert environment.is_finished()
    assert urgent_letter.delivery_time == 12
    assert regular_letter.delivery_time == 30
    assert urgent_letter.lateness_minutes == 0
    assert regular_letter.lateness_minutes == 0

    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()