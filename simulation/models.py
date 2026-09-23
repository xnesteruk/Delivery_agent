from enum import Enum

class Position:
    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def __eq__(self, other):
        if not isinstance(other, Position):
            return NotImplemented

        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"




class Letter:
    def __init__(
        self,
        letter_id: int,
        destination: Position,
        available_time: int,
        delivery_allowance_minutes: int,
    ):
        if available_time < 0:
            raise ValueError("Available time cannot be negative.")

        if delivery_allowance_minutes <= 0:
            raise ValueError("Delivery allowance must be positive.")

        self._letter_id = letter_id
        self._destination = destination
        self._available_time = available_time

        self._deadline = available_time + delivery_allowance_minutes
        self._pickup_time = None
        self._delivery_time = None

    @property
    def letter_id(self) -> int:
        return self._letter_id

    @property
    def destination(self) -> Position:
        return self._destination

    @property
    def available_time(self) -> int:
        return self._available_time

    @property
    def deadline(self) -> int:
        return self._deadline

    @property
    def pickup_time(self) -> int | None:
        return self._pickup_time

    @property
    def delivery_time(self) -> int | None:
        return self._delivery_time

    @property
    def is_picked_up(self) -> bool:
        return self._pickup_time is not None

    @property
    def is_delivered(self) -> bool:
        return self._delivery_time is not None

    @property
    def lateness_minutes(self) -> int | None:
        if not self.is_delivered:
            return None

        return max(0, self.delivery_time - self.deadline)

    def mark_picked_up(self, current_time: int) -> None:
        if self.is_picked_up:
            raise ValueError("Letter has already been picked up.")

        if current_time < self.available_time:
            raise ValueError("Letter is not available yet.")

        self._pickup_time = current_time

    def mark_delivered(self, current_time: int) -> None:
        if not self.is_picked_up:
            raise ValueError("Letter must be picked up before delivery.")

        if self.is_delivered:
            raise ValueError("Letter has already been delivered.")

        if current_time < self.pickup_time:
            raise ValueError("Delivery cannot happen before pickup.")

        self._delivery_time = current_time


class Courier:
    def __init__(self, position: Position, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")

        self.position = position
        self._capacity = capacity
        self._letters = []

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, value: Position):
        if not isinstance(value, Position):
            raise TypeError("Position must be a Position object.")

        self._position = value

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def letters(self) -> tuple[Letter, ...]:
        return tuple(self._letters)

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - len(self._letters)

    def pick_up(self, letter: Letter, current_time: int) -> None:
        if self.remaining_capacity == 0:
            raise ValueError("Courier is at full capacity.")

        if any(
            carried.letter_id == letter.letter_id
            for carried in self._letters
        ):
            raise ValueError("Courier already carries this letter ID.")

        letter.mark_picked_up(current_time)
        self._letters.append(letter)

    def deliver(self, letter_id: int, current_time: int) -> None:
        for letter in self._letters:
            if letter.letter_id == letter_id:
                if self.position != letter.destination:
                    raise ValueError(
                        "Courier is not at the delivery address."
                    )

                letter.mark_delivered(current_time)
                self._letters.remove(letter)
                return

        raise ValueError("Courier does not carry this letter.")


class ActionType(Enum):
    MOVE = "move"
    PICK_UP = "pick_up"
    DELIVER = "deliver"
    WAIT = "wait"


class Action:
    def __init__(
        self,
        action_type: ActionType,
        destination: Position | None = None,
        letter_id: int | None = None,
    ):
        if not isinstance(action_type, ActionType):
            raise TypeError("Action type must be an ActionType.")

        if action_type == ActionType.MOVE:
            if not isinstance(destination, Position):
                raise ValueError("Movement requires a destination.")

            if letter_id is not None:
                raise ValueError("Movement does not use a letter ID.")

        elif action_type in (ActionType.PICK_UP, ActionType.DELIVER):
            if type(letter_id) is not int:
                raise ValueError("Pickup and delivery require a letter ID.")

            if destination is not None:
                raise ValueError("Pickup and delivery do not use a destination.")

        elif destination is not None or letter_id is not None:
            raise ValueError("Waiting does not use a destination or letter ID.")

        self._action_type = action_type
        self._destination = destination
        self._letter_id = letter_id

    @property
    def action_type(self) -> ActionType:
        return self._action_type

    @property
    def destination(self) -> Position | None:
        return self._destination

    @property
    def letter_id(self) -> int | None:
        return self._letter_id


class LetterObservation:
    def __init__(self, letter: Letter):
        self._letter_id = letter.letter_id

        # Copy the coordinates instead of sharing the original position.
        self._destination = Position(
            letter.destination.x,
            letter.destination.y,
        )

        self._available_time = letter.available_time
        self._deadline = letter.deadline
        self._is_picked_up = letter.is_picked_up
        self._is_delivered = letter.is_delivered

    @property
    def letter_id(self) -> int:
        return self._letter_id

    @property
    def destination(self) -> Position:
        return self._destination

    @property
    def available_time(self) -> int:
        return self._available_time

    @property
    def deadline(self) -> int:
        return self._deadline

    @property
    def is_picked_up(self) -> bool:
        return self._is_picked_up

    @property
    def is_delivered(self) -> bool:
        return self._is_delivered


class Observation:
    def __init__(
        self,
        current_time: int,
        courier_position: Position,
        depot_position: Position,
        remaining_capacity: int,
        letters: tuple[LetterObservation, ...],
        last_action_result: str | None = None,
    ):
        self._current_time = current_time

        self._courier_position = Position(
            courier_position.x,
            courier_position.y,
        )

        self._depot_position = Position(
            depot_position.x,
            depot_position.y,
        )

        self._remaining_capacity = remaining_capacity
        self._letters = tuple(letters)
        self._last_action_result = last_action_result

    @property
    def current_time(self) -> int:
        return self._current_time

    @property
    def courier_position(self) -> Position:
        return self._courier_position

    @property
    def depot_position(self) -> Position:
        return self._depot_position

    @property
    def remaining_capacity(self) -> int:
        return self._remaining_capacity

    @property
    def letters(self) -> tuple[LetterObservation, ...]:
        return self._letters

    @property
    def last_action_result(self) -> str | None:
        return self._last_action_result