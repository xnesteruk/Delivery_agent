from config import SimulationConfig
from environment import Environment
from models import Action, ActionType, Letter, Position


def main():
    config = SimulationConfig()

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

    environment = Environment(config, letters)

    actions = [
        Action(ActionType.PICK_UP, letter_id=1),
        Action(ActionType.PICK_UP, letter_id=2),
        Action(ActionType.PICK_UP, letter_id=3),

        Action(ActionType.MOVE, destination=Position(3, 2)),
        Action(ActionType.DELIVER, letter_id=1),

        Action(ActionType.MOVE, destination=Position(3, 3)),
        Action(ActionType.DELIVER, letter_id=2),

        Action(ActionType.MOVE, destination=Position(2, 3)),
        Action(ActionType.DELIVER, letter_id=3),
    ]

    for action in actions:
        observation = environment.step(action)

        print(
            f"Time: {observation.current_time:2} | "
            f"Free capacity: {observation.remaining_capacity:2} | "
            f"{observation.last_action_result}"
        )
        print("Finished:", environment.is_finished())

    assert environment.is_finished()
    assert observation.current_time == 18
    assert observation.remaining_capacity == config.carrying_capacity


if __name__ == "__main__":
    main()