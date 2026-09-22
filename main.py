from config import SimulationConfig
from environment import Environment
from models import Action, ActionType, Position, Letter

def main():
    config = SimulationConfig()

    letter = Letter(
        letter_id=1,
        destination=Position(3, 2),
        available_time=0,
        delivery_allowance_minutes=config.delivery_allowance_minutes,
    )

    environment = Environment(config, letters=[letter])

    observation = environment.step(
        Action(ActionType.PICK_UP, letter_id=1)
    )

    print("Pickup:", observation.last_action_result)
    print("Free capacity:", observation.remaining_capacity)


    observation = environment.get_observation()
    print("Before:", observation.courier_position)

    action = Action(
        action_type=ActionType.MOVE,
        destination=Position(3, 2),
    )
    observation = environment.step(action)

    print("After:", observation.courier_position)
    print("Time:", observation.current_time)
    print("Result:", observation.last_action_result)

    observation = environment.step(Action(ActionType.WAIT))

    print("Wait:", observation.last_action_result)
    print("Time after waiting:", observation.current_time)

    observation = environment.step(
        Action(ActionType.DELIVER, letter_id=1)
    )

    print("Delivery:", observation.last_action_result)
    print("Free capacity:", observation.remaining_capacity)
    print("Finished:", environment.is_finished())

if __name__ == "__main__":
    main()