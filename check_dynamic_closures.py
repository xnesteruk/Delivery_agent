from agents.delivery_agent import DeliveryAgent
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.models import Letter, Position
from simulation.road_closures import RoadClosure


def main():
    config = SimulationConfig()

    letter = Letter(
        letter_id=1,
        destination=Position(3, 2),
        available_time=0,
        delivery_allowance_minutes=120,
    )

    environment = Environment(
        config=config,
        letters=[letter],
        scheduled_closures=[
            RoadClosure(
                cell=(3, 2),
                start_minute=0,
                end_minute=3,
            )
        ],
    )

    agent = DeliveryAgent(
        map_width=config.map_width,
        map_height=config.map_height,
        move_minutes=config.move_minutes,
    )

    for step in range(20):
        if environment.is_finished():
            break

        observation = environment.get_observation()

        if observation.current_time < 3:
            assert (3, 2) in observation.vision.visible_closures
        else:
            assert (3, 2) not in observation.vision.visible_closures

        action = agent.choose_action(observation)
        result = environment.step(action)

        print(
            f"Step: {step + 1} | "
            f"Time: {result.current_time} | "
            f"{result.last_action_result}"
        )

    assert environment.is_finished()
    assert letter.pickup_time == 3
    assert letter.delivery_time == 9
    assert (3, 2) not in agent.known_closures

    print("Dynamic closure check passed.")


if __name__ == "__main__":
    main()