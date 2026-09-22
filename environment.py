from config import SimulationConfig
from models import Position, Letter, Courier, LetterObservation, Observation
from models import Action, ActionType

class Environment:
    def __init__(
        self,
        config: SimulationConfig,
        letters: list[Letter],
    ):
        self._config = config
        self._current_time = 0
        self._last_action_result = None

        depot_x, depot_y = config.depot_position
        self._depot_position = Position(depot_x, depot_y)

        self._courier = Courier(
            position=self._depot_position,
            capacity=config.carrying_capacity,
        )

        self._letters = {}

        for letter in letters:
            if letter.letter_id in self._letters:
                raise ValueError("Letter IDs must be unique.")

            if not self._is_inside_map(letter.destination):
                raise ValueError(
                    f"Letter {letter.letter_id} has a destination "
                    "outside the map."
                )

            if letter.is_picked_up:
                raise ValueError("Simulation requires uncollected letters.")

            self._letters[letter.letter_id] = letter

    def _move(self, destination: Position) -> None:
        if not self._is_inside_map(destination):
            self._last_action_result = (
                "Movement failed: destination is outside the map."
            )
            return

        current = self._courier.position
        distance = (
                abs(destination.x - current.x)
                + abs(destination.y - current.y)
        )

        if distance != 1:
            self._last_action_result = (
                "Movement failed: choose an adjacent cell."
            )
            return

        self._courier.position = destination
        self._current_time += self._config.move_minutes
        self._last_action_result = "Moved successfully."

    @property
    def current_time(self) -> int:
        return self._current_time

    def _is_inside_map(self, position: Position) -> bool:
        return (
            0 <= position.x < self._config.map_width
            and 0 <= position.y < self._config.map_height
        )

    def get_observation(self) -> Observation:
        visible_letters = []

        for letter in self._letters.values():
            if letter.available_time <= self._current_time:
                visible_letters.append(LetterObservation(letter))

        return Observation(
            current_time=self._current_time,
            courier_position=self._courier.position,
            depot_position=self._depot_position,
            remaining_capacity=self._courier.remaining_capacity,
            letters=tuple(visible_letters),
            last_action_result=self._last_action_result,
        )

    def is_finished(self) -> bool:
        return all(
            letter.is_delivered
            for letter in self._letters.values()
        )

    def step(self, action: Action) -> Observation:
        if action.action_type == ActionType.MOVE:
            self._move(action.destination)
        elif action.action_type == ActionType.PICK_UP:
            self._pick_up(action.letter_id)
        elif action.action_type == ActionType.DELIVER:
            self._deliver(action.letter_id)
        elif action.action_type == ActionType.WAIT:
            self._wait()
        else:
            raise NotImplementedError(
                f"Action {action.action_type.value} is not implemented yet."
            )

        return self.get_observation()

    def _pick_up(self, letter_id: int) -> None:
        if self._courier.position != self._depot_position:
            self._last_action_result = (
                "Pickup failed: courier must be at the depot."
            )
            return

        letter = self._letters.get(letter_id)

        if letter is None:
            self._last_action_result = "Pickup failed: unknown letter ID."
            return

        if letter.available_time > self._current_time:
            self._last_action_result = "Pickup failed: letter is not available yet."
            return

        if letter.is_picked_up:
            self._last_action_result = "Pickup failed: letter already picked up."
            return

        if self._courier.remaining_capacity == 0:
            self._last_action_result = "Pickup failed: courier is at full capacity."
            return

        self._courier.pick_up(letter, self._current_time)
        self._last_action_result = f"Picked up letter {letter_id}."

    def _deliver(self, letter_id: int) -> None:
        carried_letter = None

        for letter in self._courier.letters:
            if letter.letter_id == letter_id:
                carried_letter = letter
                break

        if carried_letter is None:
            self._last_action_result = (
                "Delivery failed: courier does not carry this letter."
            )
            return

        if self._courier.position != carried_letter.destination:
            self._last_action_result = (
                "Delivery failed: courier is not at the delivery address."
            )
            return

        self._courier.deliver(letter_id, self._current_time)
        self._last_action_result = f"Delivered letter {letter_id}."

    def _wait(self) -> None:
        self._current_time += 1
        self._last_action_result = "Waited for 1 minute."