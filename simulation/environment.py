from random import Random

from simulation.config import SimulationConfig
from simulation.models import (
    Action,
    ActionType,
    Courier,
    Letter,
    LetterInfo,
    Position,
)
from simulation.observation import AgentObservation
from simulation.road_closures import ClosureSchedule, RoadClosure
from simulation.sensors import Direction, VisionSensor


class Environment:
    def __init__(
        self,
        config: SimulationConfig,
        letters: list[Letter],
        blocked_cells: set[tuple[int, int]] | None = None,
        closures: set[tuple[int, int]] | None = None,
        initial_direction: Direction = Direction.EAST,
        scheduled_closures: list[RoadClosure] | None = None,
        pickup_posts: tuple[tuple[int, int], ...] | None = None,
        arrivals_seed: int = 0,
    ):
        self._config = config
        self._current_time = 0
        self._last_action_result = None

        self._direction = initial_direction
        self._vision_sensor = VisionSensor(side_range=2)

        self._blocked_cells = set(
            blocked_cells if blocked_cells is not None else ()
        )

        # These closures remain active for the entire run.
        self._fixed_closures = set(
            closures if closures is not None else ()
        )

        self._closure_schedule = ClosureSchedule(
            scheduled_closures if scheduled_closures is not None else []
        )

        all_closure_cells = (
            self._fixed_closures
            | self._closure_schedule.affected_cells
        )

        for x, y in self._blocked_cells | all_closure_cells:
            if not self._is_inside_map(Position(x, y)):
                raise ValueError("Obstacle is outside the map.")

        if self._blocked_cells & all_closure_cells:
            raise ValueError("A closure cannot be inside a building.")

        depot_x, depot_y = config.depot_position
        self._depot_position = Position(depot_x, depot_y)

        # Keep the depot accessible throughout the simulation.
        if (depot_x, depot_y) in self._blocked_cells | all_closure_cells:
            raise ValueError("Depot cannot be blocked.")

        self._courier = Courier(
            position=self._depot_position,
            capacity=config.carrying_capacity,
        )

        cells = tuple(pickup_posts) if pickup_posts is not None else (config.depot_position,)
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("Pickup posts must be nonempty and unique.")
        for cell in cells:
            if not self._is_inside_map(Position(*cell)) or cell in self._blocked_cells | all_closure_cells:
                raise ValueError("Pickup post must be an unblocked map cell.")
        self._pickup_posts = tuple(Position(*cell) for cell in cells)
        self._arrivals_random = Random(arrivals_seed)
        self._letters = {}

        for letter in letters:
            if letter.letter_id in self._letters:
                raise ValueError("Letter IDs must be unique.")

            if not self._is_inside_map(letter.destination):
                raise ValueError("Letter destination is outside the map.")

            destination = (
                letter.destination.x,
                letter.destination.y,
            )

            if destination in self._blocked_cells:
                raise ValueError(
                    "Letter destination cannot be inside a building."
                )

            if letter.is_picked_up:
                raise ValueError("Simulation requires uncollected letters.")

            letter.defer_appearance()
            self._letters[letter.letter_id] = letter
        self._pending = sorted(letters, key=lambda letter: (letter.scheduled_time, letter.letter_id))
        self._admit_letters()

    def _admit_letters(self) -> None:
        while self._pending and self._pending[0].scheduled_time <= self._current_time:
            post = self._arrivals_random.choice(self._pickup_posts)
            self._pending.pop(0).appear(self._current_time, post)

    def _advance_time(self, minutes: int) -> None:
        # Admit at the actual minute, including arrivals during a move.
        for _ in range(minutes):
            self._current_time += 1
            self._admit_letters()

    @property
    def current_time(self) -> int:
        return self._current_time

    def _is_inside_map(self, position: Position) -> bool:
        return (
            0 <= position.x < self._config.map_width
            and 0 <= position.y < self._config.map_height
        )

    def _active_closures(self) -> set[tuple[int, int]]:
        return (
            self._fixed_closures
            | self._closure_schedule.active_cells(self._current_time)
        )

    def get_observation(self) -> AgentObservation:
        position = self._courier.position

        vision = self._vision_sensor.observe(
            position=(position.x, position.y),
            direction=self._direction,
            map_width=self._config.map_width,
            map_height=self._config.map_height,
            buildings=self._blocked_cells,
            closures=self._active_closures(),
        )

        available_letters = tuple(
            LetterInfo(letter)
            for letter in self._letters.values()
            if letter.available_time is not None and letter.available_time <= self._current_time
        )

        return AgentObservation(
            current_time=self._current_time,
            courier_position=Position(position.x, position.y),
            depot_position=Position(
                self._depot_position.x,
                self._depot_position.y,
            ),
            remaining_capacity=self._courier.remaining_capacity,
            letters=available_letters,
            direction=self._direction,
            vision=vision,
            last_action_result=self._last_action_result,
            pickup_posts=tuple(Position(p.x, p.y) for p in self._pickup_posts),
        )

    def is_finished(self) -> bool:
        return all(
            letter.is_delivered
            for letter in self._letters.values()
        )

    def step(self, action: Action) -> AgentObservation:
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
                f"Action {action.action_type} is not implemented."
            )

        return self.get_observation()

    def _move(self, destination: Position) -> None:
        if not self._is_inside_map(destination):
            self._last_action_result = (
                "Movement failed: destination is outside the map."
            )
            return

        current = self._courier.position
        dx = destination.x - current.x
        dy = destination.y - current.y

        if abs(dx) + abs(dy) != 1:
            self._last_action_result = (
                "Movement failed: choose an adjacent cell."
            )
            return

        self._direction = Direction((dx, dy))
        cell = (destination.x, destination.y)

        if cell in self._blocked_cells:
            self._last_action_result = (
                "Movement failed: cell is blocked."
            )
            return

        if cell in self._active_closures():
            self._advance_time(1)
            self._last_action_result = (
                "Movement failed: road is temporarily closed."
            )
            return

        self._courier.position = destination
        self._advance_time(self._config.move_minutes)
        self._last_action_result = "Moved successfully."

    def _pick_up(self, letter_id: int) -> None:
        letter = self._letters.get(letter_id)

        if letter is None:
            self._last_action_result = (
                "Pickup failed: unknown letter ID."
            )
            return

        if letter.available_time is None or letter.available_time > self._current_time:
            self._last_action_result = (
                "Pickup failed: letter is not available yet."
            )
            return

        if self._courier.position != letter.pickup_position:
            self._last_action_result = "Pickup failed: courier must be at the letter's post."
            return

        if letter.is_picked_up:
            self._last_action_result = (
                "Pickup failed: letter already picked up."
            )
            return

        if self._courier.remaining_capacity == 0:
            self._last_action_result = (
                "Pickup failed: courier is at full capacity."
            )
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
        self._advance_time(1)
        self._last_action_result = "Waited for 1 minute."
