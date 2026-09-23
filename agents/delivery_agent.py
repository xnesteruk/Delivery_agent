from collections import deque

from simulation.models import Action, ActionType, Position
from simulation.observation import AgentObservation


class DeliveryAgent:
    def __init__(
        self,
        map_width: int,
        map_height: int,
        blocked_cells: set[tuple[int, int]] | None = None,
        move_minutes: int = 6,
    ):
        if move_minutes <= 0:
            raise ValueError("Movement time must be positive.")

        self._map_width = map_width
        self._map_height = map_height
        self._move_minutes = move_minutes

        self._blocked_cells = frozenset(
            blocked_cells if blocked_cells is not None else ()
        )

        self._known_closures: set[tuple[int, int]] = set()

    @property
    def known_closures(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._known_closures)

    def choose_action(self, observation: AgentObservation) -> Action:
        self._update_memory(observation)

        position = observation.courier_position

        carried_letters = [
            letter
            for letter in observation.letters
            if letter.is_picked_up and not letter.is_delivered
        ]

        for letter in carried_letters:
            if letter.destination == position:
                return Action(
                    ActionType.DELIVER,
                    letter_id=letter.letter_id,
                )

        if (
            position == observation.depot_position
            and observation.remaining_capacity > 0
        ):
            for letter in observation.letters:
                if letter.is_picked_up:
                    continue

                path = self._find_path(position, letter.destination)

                if path is not None:
                    return Action(
                        ActionType.PICK_UP,
                        letter_id=letter.letter_id,
                    )

        best_path = None
        best_priority = None

        for letter in carried_letters:
            path = self._find_path(position, letter.destination)

            if path is None:
                continue

            travel_minutes = len(path) * self._move_minutes

            slack = (
                letter.deadline
                - observation.current_time
                - travel_minutes
            )

            if slack >= 0:
                priority = (
                    0,
                    slack,
                    travel_minutes,
                    letter.letter_id,
                )
            else:
                priority = (
                    1,
                    travel_minutes,
                    letter.deadline,
                    letter.letter_id,
                )

            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_path = path

        if best_path:
            return Action(
                ActionType.MOVE,
                destination=best_path[0],
            )

        if not carried_letters and position != observation.depot_position:
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

    def _update_memory(self, observation: AgentObservation) -> None:
        vision = observation.vision

        # Replace old information only for cells currently visible.
        self._known_closures.difference_update(vision.visible_cells)
        self._known_closures.update(vision.visible_closures)

    def _find_path(
        self,
        start: Position,
        goal: Position,
    ) -> list[Position] | None:
        start_cell = (start.x, start.y)
        goal_cell = (goal.x, goal.y)

        if not self._is_walkable(start_cell):
            return None

        if not self._is_walkable(goal_cell):
            return None

        frontier = deque([start_cell])
        came_from = {start_cell: None}

        while frontier:
            current = frontier.popleft()

            if current == goal_cell:
                return self._restore_path(came_from, goal_cell)

            x, y = current

            neighbours = (
                (x + 1, y),
                (x, y + 1),
                (x - 1, y),
                (x, y - 1),
            )

            for neighbour in neighbours:
                if neighbour in came_from:
                    continue

                if not self._is_walkable(neighbour):
                    continue

                came_from[neighbour] = current
                frontier.append(neighbour)

        return None

    def _is_walkable(self, cell: tuple[int, int]) -> bool:
        x, y = cell

        return (
            0 <= x < self._map_width
            and 0 <= y < self._map_height
            and cell not in self._blocked_cells
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