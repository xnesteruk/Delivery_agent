from dataclasses import replace

from agents.delivery_agent import DeliveryAgent
from simulation.models import ActionType, Letter, LetterInfo, Position
from simulation.observation import AgentObservation
from simulation.sensors import Direction, VisionObservation


def main():
    agent = DeliveryAgent(
        map_width=5,
        map_height=5,
        closure_memory_minutes=12,
    )

    letter = Letter(
        letter_id=1,
        destination=Position(3, 2),
        available_time=0,
        delivery_allowance_minutes=120,
    )
    letter.mark_picked_up(0)

    observation = AgentObservation(
        current_time=0,
        courier_position=Position(2, 2),
        depot_position=Position(2, 2),
        remaining_capacity=9,
        letters=(LetterInfo(letter),),
        direction=Direction.EAST,
        vision=VisionObservation(
            visible_cells=frozenset({(2, 2), (3, 2)}),
            visible_closures=frozenset({(3, 2)}),
        ),
    )

    # The destination is visibly closed.
    action = agent.choose_action(observation)
    assert action.action_type == ActionType.WAIT
    assert (3, 2) in agent.known_closures

    # A controlled observation looking away from the closure.
    hidden_observation = replace(
        observation,
        current_time=11,
        direction=Direction.WEST,
        vision=VisionObservation(
            visible_cells=frozenset({
                (2, 2), (1, 2), (0, 2),
                (2, 1), (2, 0), (2, 3), (2, 4),
            }),
            visible_closures=frozenset(),
        ),
    )

    action = agent.choose_action(hidden_observation)
    assert action.action_type == ActionType.WAIT
    assert (3, 2) in agent.known_closures

    # After 12 minutes, the agent is willing to retry the route.
    expired_observation = replace(
        hidden_observation,
        current_time=12,
    )

    action = agent.choose_action(expired_observation)
    assert (3, 2) not in agent.known_closures
    assert action.action_type == ActionType.MOVE
    assert action.destination == Position(3, 2)

    # Seeing the barrier again immediately restores the restriction.
    visible_again = replace(observation, current_time=13)

    action = agent.choose_action(visible_again)
    assert action.action_type == ActionType.WAIT
    assert (3, 2) in agent.known_closures

    print("Memory expiry checks passed.")


if __name__ == "__main__":
    main()