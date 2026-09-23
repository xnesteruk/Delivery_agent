from dataclasses import replace

from agents.delivery_agent import DeliveryAgent
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.models import ActionType, Letter, Position
from simulation.sensors import VisionObservation


def main():
    config = SimulationConfig()

    environment = Environment(
        config=config,
        letters=[
            Letter(
                letter_id=1,
                destination=Position(4, 2),
                available_time=0,
                delivery_allowance_minutes=120,
            )
        ],
        closures={(3, 2), (0, 0)},
    )

    agent = DeliveryAgent(
        map_width=config.map_width,
        map_height=config.map_height,
        move_minutes=config.move_minutes,
    )

    # The closure ahead is visible; the distant one is hidden.
    observation = environment.get_observation()

    assert observation.vision.visible_closures == frozenset({(3, 2)})

    pickup = agent.choose_action(observation)

    assert pickup.action_type == ActionType.PICK_UP
    assert agent.known_closures == frozenset({(3, 2)})

    environment.step(pickup)

    # Deliver by going around the observed closure.
    for _ in range(20):
        if environment.is_finished():
            break

        observation = environment.get_observation()
        action = agent.choose_action(observation)
        result = environment.step(action)

        print(
            f"Time: {result.current_time:2} | "
            f"Position: {result.courier_position} | "
            f"{result.last_action_result}"
        )

        assert result.courier_position != Position(3, 2)
        assert (0, 0) not in agent.known_closures

    assert environment.is_finished()
    assert environment.current_time == 24
    assert (3, 2) in agent.known_closures

    # Separately test memory updates using controlled observations.
    memory_agent = DeliveryAgent(
        map_width=config.map_width,
        map_height=config.map_height,
        move_minutes=config.move_minutes,
    )

    blocked_view = replace(
        observation,
        vision=VisionObservation(
            visible_cells=frozenset({(3, 2)}),
            visible_closures=frozenset({(3, 2)}),
        ),
    )

    memory_agent.choose_action(blocked_view)
    assert (3, 2) in memory_agent.known_closures

    hidden_view = replace(
        blocked_view,
        vision=VisionObservation(
            visible_cells=frozenset({(2, 2)}),
            visible_closures=frozenset(),
        ),
    )

    memory_agent.choose_action(hidden_view)
    assert (3, 2) in memory_agent.known_closures

    clear_view = replace(
        blocked_view,
        vision=VisionObservation(
            visible_cells=frozenset({(3, 2)}),
            visible_closures=frozenset(),
        ),
    )

    memory_agent.choose_action(clear_view)
    assert (3, 2) not in memory_agent.known_closures

    print("All sensor and memory checks passed.")


if __name__ == "__main__":
    main()