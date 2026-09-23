from collections import deque

from simulation.models import Action, ActionType, LetterInfo, Position
from simulation.observation import AgentObservation


class DeliveryAgent:
    def __init__(
        self,
        map_width: int,
        map_height: int,
        blocked_cells: set[tuple[int, int]] | None = None,
        move_minutes: int = 6,
        closure_memory_minutes: int = 12,
    ):
        if move_minutes <= 0:
            raise ValueError("Movement time must be positive.")

        if closure_memory_minutes <= 0:
            raise ValueError("Closure memory duration must be positive.")

        self._map_width = map_width
        self._map_height = map_height
        self._move_minutes = move_minutes
        self._closure_memory_minutes = closure_memory_minutes

        self._blocked_cells = frozenset(
            blocked_cells if blocked_cells is not None else ()
        )
        self._known_closures: dict[tuple[int, int], int] = {}

        # Current delivery destination, retained between decisions.
        self._target_letter_id: int | None = None

        # A special plan: return, collect, and deliver an urgent letter.
        self._priority_letter_id: int | None = None

    @property
    def known_closures(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._known_closures)

    @property
    def target_letter_id(self) -> int | None:
        if self._priority_letter_id is not None:
            return self._priority_letter_id

        return self._target_letter_id

    def choose_action(self, observation: AgentObservation) -> Action:
        self._update_memory(observation)
        position = observation.courier_position

        carried = [
            letter
            for letter in observation.letters
            if letter.is_picked_up and not letter.is_delivered
        ]
        waiting = [
            letter
            for letter in observation.letters
            if not letter.is_picked_up
        ]

        # Deliver any carried letter at the current address.
        for letter in carried:
            if letter.destination == position:
                if letter.letter_id == self._target_letter_id:
                    self._target_letter_id = None

                if letter.letter_id == self._priority_letter_id:
                    self._priority_letter_id = None

                return Action(
                    ActionType.DELIVER,
                    letter_id=letter.letter_id,
                )

        priority_action = self._follow_priority_plan(observation)

        if priority_action is not None:
            return priority_action

        if (
            position == observation.depot_position
            and observation.remaining_capacity > 0
        ):
            choices = self._delivery_choices(
                position,
                waiting,
                observation.current_time,
            )

            if choices:
                _, letter, _ = min(
                    choices,
                    key=lambda item: item[0],
                )

                return Action(
                    ActionType.PICK_UP,
                    letter_id=letter.letter_id,
                )

        current_choice = self._get_current_delivery(
            position,
            carried,
            observation.current_time,
        )

        if current_choice is not None:
            current_letter, path = current_choice

            # Returning for an urgent letter remains an explicit
            # reason to interrupt the current delivery.
            if (
                position != observation.depot_position
                and observation.remaining_capacity > 0
            ):
                urgent = self._find_return_candidate(
                    observation,
                    current_letter,
                    carried,
                    waiting,
                )

                if urgent is not None:
                    self._priority_letter_id = urgent.letter_id
                    action = self._follow_priority_plan(observation)

                    if action is not None:
                        self._target_letter_id = None
                        return action

            return Action(
                ActionType.MOVE,
                destination=path[0],
            )

        if not carried and position != observation.depot_position:
            path = self._find_path(
                position,
                observation.depot_position,
            )

            if path:
                return Action(
                    ActionType.MOVE,
                    destination=path[0],
                )

        return Action(ActionType.WAIT)

    def _get_current_delivery(
        self,
        position: Position,
        carried: list[LetterInfo],
        current_time: int,
    ) -> tuple[LetterInfo, list[Position]] | None:
        # Keep the selected letter while a route remains available.
        for letter in carried:
            if letter.letter_id == self._target_letter_id:
                path = self._find_path(position, letter.destination)

                if path is not None:
                    return letter, path

                break

        # The previous target is gone or currently unreachable.
        self._target_letter_id = None

        choices = self._delivery_choices(
            position,
            carried,
            current_time,
        )

        if not choices:
            return None

        _, letter, path = min(
            choices,
            key=lambda item: item[0],
        )

        self._target_letter_id = letter.letter_id
        return letter, path

    def _follow_priority_plan(
        self,
        observation: AgentObservation,
    ) -> Action | None:
        if self._priority_letter_id is None:
            return None

        letter = next(
            (
                item
                for item in observation.letters
                if item.letter_id == self._priority_letter_id
            ),
            None,
        )

        if letter is None or letter.is_delivered:
            self._priority_letter_id = None
            return None

        position = observation.courier_position

        if letter.is_picked_up:
            target = letter.destination
        else:
            if observation.remaining_capacity == 0:
                self._priority_letter_id = None
                return None

            target = observation.depot_position

            if position == target:
                return Action(
                    ActionType.PICK_UP,
                    letter_id=letter.letter_id,
                )

        path = self._find_path(position, target)

        if path:
            return Action(
                ActionType.MOVE,
                destination=path[0],
            )

        self._priority_letter_id = None
        return None

    def _delivery_choices(
        self,
        position: Position,
        letters: list[LetterInfo],
        current_time: int,
    ) -> list:
        choices = []

        for letter in letters:
            path = self._find_path(position, letter.destination)

            if path is None:
                continue

            travel = len(path) * self._move_minutes
            slack = letter.deadline - current_time - travel

            if slack >= 0:
                priority = (0, slack, travel, letter.letter_id)
            else:
                priority = (
                    1,
                    travel,
                    letter.deadline,
                    letter.letter_id,
                )

            choices.append((priority, letter, path))

        return choices

    def _find_return_candidate(
        self,
        observation: AgentObservation,
        current_letter: LetterInfo,
        carried: list[LetterInfo],
        waiting: list[LetterInfo],
    ) -> LetterInfo | None:
        position = observation.courier_position
        depot = observation.depot_position
        now = observation.current_time

        to_depot = self._travel_time(position, depot)
        to_current = self._travel_time(
            position,
            current_letter.destination,
        )
        current_to_depot = self._travel_time(
            current_letter.destination,
            depot,
        )

        if (
            to_depot is None
            or to_current is None
            or current_to_depot is None
        ):
            return None

        candidates = []

        for letter in waiting:
            depot_to_letter = self._travel_time(
                depot,
                letter.destination,
            )

            if depot_to_letter is None:
                continue

            arrival_if_return = now + to_depot + depot_to_letter
            arrival_if_continue = (
                now
                + to_current
                + current_to_depot
                + depot_to_letter
            )

            if not (
                arrival_if_return <= letter.deadline
                < arrival_if_continue
            ):
                continue

            if not self._can_finish_on_time(
                letter.destination,
                arrival_if_return,
                carried,
            ):
                continue

            slack = letter.deadline - arrival_if_return
            candidates.append((slack, letter.letter_id, letter))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[:2])[2]

    def _can_finish_on_time(
        self,
        position: Position,
        current_time: int,
        letters: list[LetterInfo],
    ) -> bool:
        remaining = list(letters)

        while remaining:
            choices = self._delivery_choices(
                position,
                remaining,
                current_time,
            )

            if not choices:
                return False

            _, letter, path = min(
                choices,
                key=lambda item: item[0],
            )

            current_time += len(path) * self._move_minutes

            if current_time > letter.deadline:
                return False

            position = letter.destination
            remaining.remove(letter)

        return True

    def _travel_time(
        self,
        start: Position,
        goal: Position,
    ) -> int | None:
        path = self._find_path(start, goal)

        if path is None:
            return None

        return len(path) * self._move_minutes

    def _update_memory(self, observation: AgentObservation) -> None:
        now = observation.current_time
        vision = observation.vision

        self._known_closures = {
            cell: last_seen
            for cell, last_seen in self._known_closures.items()
            if now - last_seen < self._closure_memory_minutes
        }

        for cell in vision.visible_cells:
            if cell in vision.visible_closures:
                self._known_closures[cell] = now
            else:
                self._known_closures.pop(cell, None)

    def _find_path(
        self,
        start: Position,
        goal: Position,
    ) -> list[Position] | None:
        start_cell = (start.x, start.y)
        goal_cell = (goal.x, goal.y)

        if not self._is_static_walkable(start_cell):
            return None

        if start_cell == goal_cell:
            return []

        if not self._is_walkable(goal_cell):
            return None

        frontier = deque([start_cell])
        came_from = {start_cell: None}

        while frontier:
            current = frontier.popleft()

            if current == goal_cell:
                return self._restore_path(came_from, goal_cell)

            x, y = current

            for neighbour in (
                (x + 1, y),
                (x, y + 1),
                (x - 1, y),
                (x, y - 1),
            ):
                if neighbour in came_from:
                    continue

                if not self._is_walkable(neighbour):
                    continue

                came_from[neighbour] = current
                frontier.append(neighbour)

        return None

    def _is_static_walkable(self, cell: tuple[int, int]) -> bool:
        x, y = cell

        return (
            0 <= x < self._map_width
            and 0 <= y < self._map_height
            and cell not in self._blocked_cells
        )

    def _is_walkable(self, cell: tuple[int, int]) -> bool:
        return (
            self._is_static_walkable(cell)
            and cell not in self._known_closures
        )

    def _restore_path(
        self,
        came_from: dict[
            tuple[int, int],
            tuple[int, int] | None,
        ],
        goal: tuple[int, int],
    ) -> list[Position]:
        path = []
        current = goal

        while came_from[current] is not None:
            path.append(Position(*current))
            current = came_from[current]

        path.reverse()
        return path