from agents.delivery_agent import DeliveryAgent
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.models import Letter, Position


def main():
    config = SimulationConfig(carrying_capacity=1)

    first_letter = Letter(
        letter_id=1,
        destination=Position(4, 4),
        available_time=0,
        delivery_allowance_minutes=120,
    )

    incoming_letter = Letter(
        letter_id=2,
        destination=Position(1, 2),
        available_time=12,
        delivery_allowance_minutes=60,
    )

    environment = Environment(
        config=config,
        letters=[first_letter, incoming_letter],
    )

    agent = DeliveryAgent(
        map_width=config.map_width,
        map_height=config.map_height,
        move_minutes=config.move_minutes,
    )

    notification_received = False

    for step in range(100):
        if environment.is_finished():
            break

        observation = environment.get_observation()

        available_ids = {
            letter.letter_id
            for letter in observation.letters
        }

        if observation.current_time < 12:
            assert 2 not in available_ids
        else:
            assert 2 in available_ids

            if not notification_received:
                assert observation.current_time == 12
                assert (
                    observation.courier_position
                    != observation.depot_position
                )
                assert observation.remaining_capacity == 0

                print("Notification: letter 2 is now available.")
                notification_received = True

        action = agent.choose_action(observation)
        result = environment.step(action)

        print(
            f"Step: {step + 1:2} | "
            f"Time: {result.current_time:2} | "
            f"Free capacity: {result.remaining_capacity} | "
            f"{result.last_action_result}"
        )

        assert 0 <= result.remaining_capacity <= config.carrying_capacity

        if first_letter.is_delivered and not incoming_letter.is_delivered:
            assert not environment.is_finished()

    assert notification_received
    assert environment.is_finished()

    assert first_letter.pickup_time == 0
    assert first_letter.delivery_time == 24

    assert incoming_letter.pickup_time == 48
    assert incoming_letter.delivery_time == 54
    assert incoming_letter.lateness_minutes == 0

    print("\nIncoming letter checks passed.")


if __name__ == "__main__":
    main()