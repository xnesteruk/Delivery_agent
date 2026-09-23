from agents.delivery_agent import DeliveryAgent
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.models import Letter, Position


def run_case(first_deadline: int, should_return: bool):
    config = SimulationConfig(carrying_capacity=2)

    regular = Letter(
        letter_id=1,
        destination=Position(4, 4),
        available_time=0,
        delivery_allowance_minutes=first_deadline,
    )

    urgent = Letter(
        letter_id=2,
        destination=Position(1, 2),
        available_time=6,
        delivery_allowance_minutes=18,
    )

    environment = Environment(config, [regular, urgent])

    agent = DeliveryAgent(
        map_width=config.map_width,
        map_height=config.map_height,
        move_minutes=config.move_minutes,
    )

    for _ in range(100):
        if environment.is_finished():
            break

        observation = environment.get_observation()
        action = agent.choose_action(observation)
        environment.step(action)

    assert environment.is_finished()

    if should_return:
        assert urgent.pickup_time == 12
        assert urgent.delivery_time == 18
        assert regular.delivery_time == 48
        assert urgent.lateness_minutes == 0
        assert regular.lateness_minutes == 0
    else:
        assert regular.delivery_time == 24
        assert regular.lateness_minutes == 0
        assert urgent.pickup_time == 48
        assert urgent.delivery_time == 54
        assert urgent.lateness_minutes == 30

    print(
        f"First deadline: {first_deadline} | "
        f"First delivery: {regular.delivery_time} | "
        f"New delivery: {urgent.delivery_time}"
    )


def main():
    # Returning saves the new letter without delaying the old one
    # beyond its deadline.
    run_case(first_deadline=120, should_return=True)

    # Returning would make the old letter late, so continue.
    run_case(first_deadline=30, should_return=False)

    print("Urgent return checks passed.")


if __name__ == "__main__":
    main()