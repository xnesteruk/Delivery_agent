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

        # Commit to a depot visit once the joint forecast favors returning.
        self._returning_to_depot = False

    @property
    def known_closures(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._known_closures)

    @property
    def target_letter_id(self) -> int | None:
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

                return Action(
                    ActionType.DELIVER,
                    letter_id=letter.letter_id,
                )

        return_action = self._follow_return_plan(observation)

        if return_action is not None:
            return return_action

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

            # Compare complete forecasts for all currently known letters.
            if (
                position != observation.depot_position
                and observation.remaining_capacity > 0
            ):
                should_return = self._should_return_to_depot(
                    observation,
                    current_letter,
                    carried,
                    waiting,
                )

                if should_return:
                    self._returning_to_depot = True
                    action = self._follow_return_plan(observation)

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
                path = self._find_path(position, letter.destination, carried)

                if path is not None:
                    return letter, path

                break

        # The previous target is gone or currently unreachable.
        self._target_letter_id = None

        choices = self._delivery_choices(
            position,
            carried,
            current_time,
            carried=carried,
        )

        if not choices:
            return None

        _, letter, path = min(
            choices,
            key=lambda item: item[0],
        )

        self._target_letter_id = letter.letter_id
        return letter, path

    def _follow_return_plan(
        self,
        observation: AgentObservation,
    ) -> Action | None:
        if not self._returning_to_depot:
            return None

        if observation.courier_position == observation.depot_position:
            # The usual pickup rule fills the available capacity before leaving.
            self._returning_to_depot = False
            return None

        path = self._find_path(
            observation.courier_position, observation.depot_position,
            carried=[
                letter for letter in observation.letters
                if letter.is_picked_up and not letter.is_delivered
            ],
        )
        if path:
            return Action(ActionType.MOVE, destination=path[0])

        # A newly observed closure can invalidate the committed route.
        self._returning_to_depot = False
        return None

    def _delivery_choices(
        self,
        position: Position,
        letters: list[LetterInfo],
        current_time: int,
        carried: list[LetterInfo] | None = None,
    ) -> list:
        choices = []

        for letter in letters:
            path = self._find_path(position, letter.destination, carried)

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

    def _should_return_to_depot(
        self,
        observation: AgentObservation,
        current_letter: LetterInfo,
        carried: list[LetterInfo],
        waiting: list[LetterInfo],
    ) -> bool:
        if not waiting or observation.remaining_capacity <= 0:
            return False

        continue_score = self._forecast_delivery_score(
            observation, current_letter, carried, waiting, return_now=False,
        )
        return_score = self._forecast_delivery_score(
            observation, current_letter, carried, waiting, return_now=True,
        )
        # Unknown routes are not evidence that a detour is better.
        if continue_score is None or return_score is None:
            return False

        # Prefer fewer late letters, then fewer total late minutes.
        # Keep the current route on a tie to avoid unnecessary interruptions.
        return return_score < continue_score

    def _forecast_delivery_score(
        self,
        observation: AgentObservation,
        current_letter: LetterInfo,
        carried: list[LetterInfo],
        waiting: list[LetterInfo],
        return_now: bool,
    ) -> tuple[int, int] | None:
        """Roll out the normal policy without changing the agent or letters.

        Only observed letters and currently known closures are used. Future
        arrivals and changes to closures are unknown. Pickup and delivery take
        zero minutes, as in the simulation; movement consumes time.
        """
        position = observation.courier_position
        depot = observation.depot_position
        now = observation.current_time
        capacity = observation.remaining_capacity + len(carried)
        bag = list(carried)
        depot_letters = list(waiting)
        target_id = None if return_now else current_letter.letter_id
        returning = return_now
        late_count = 0
        total_lateness = 0

        while bag or depot_letters:
            # Match actual deliveries encountered on the way to any target.
            for letter in list(bag):
                if letter.destination == position:
                    lateness = max(0, now - letter.deadline)
                    late_count += int(lateness > 0)
                    total_lateness += lateness
                    bag.remove(letter)
                    if target_id == letter.letter_id:
                        target_id = None

            if position == depot:
                returning = False
                while len(bag) < capacity and depot_letters:
                    choices = self._delivery_choices(position, depot_letters, now)
                    if not choices:
                        return None
                    _, letter, _ = min(choices, key=lambda item: item[0])
                    bag.append(letter)
                    depot_letters.remove(letter)
                # A newly picked-up letter may be addressed to the depot.
                if any(letter.destination == position for letter in bag):
                    continue

            if not bag and not depot_letters:
                break

            if returning or not bag:
                path = self._find_path(position, depot, bag)
            else:
                target = next(
                    (letter for letter in bag if letter.letter_id == target_id),
                    None,
                )
                path = (
                    self._find_path(position, target.destination, bag)
                    if target is not None else None
                )
                if path is None:
                    choices = self._delivery_choices(position, bag, now, carried=bag)
                    if not choices:
                        return None
                    _, target, path = min(choices, key=lambda item: item[0])
                    target_id = target.letter_id

            if not path:
                return None
            position = path[0]
            now += self._move_minutes

        return late_count, total_lateness

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
        carried: list[LetterInfo] | None = None,
    ) -> list[Position] | None:
        """Find a shortest route; break ties by letters delivered on the way.

        BFS processes cells by distance. For equal-distance alternatives,
        retain the predecessor yielding the most carried-letter deliveries.
        A shortest path never revisits a cell, so each letter is counted once.
        """
        start_cell = (start.x, start.y)
        goal_cell = (goal.x, goal.y)

        if not self._is_static_walkable(start_cell):
            return None
        if start_cell == goal_cell:
            return []
        if not self._is_walkable(goal_cell):
            return None

        deliveries_at: dict[tuple[int, int], int] = {}
        for letter in carried or ():
            cell = (letter.destination.x, letter.destination.y)
            deliveries_at[cell] = deliveries_at.get(cell, 0) + 1

        frontier = deque([start_cell])
        came_from = {start_cell: None}
        distances = {start_cell: 0}
        delivery_counts = {start_cell: 0}

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
                if not self._is_walkable(neighbour):
                    continue

                distance = distances[current] + 1
                count = delivery_counts[current] + deliveries_at.get(neighbour, 0)
                if neighbour not in distances:
                    distances[neighbour] = distance
                    delivery_counts[neighbour] = count
                    came_from[neighbour] = current
                    frontier.append(neighbour)
                elif (
                    distances[neighbour] == distance
                    and count > delivery_counts[neighbour]
                ):
                    # All predecessors are processed before this cell is popped.
                    delivery_counts[neighbour] = count
                    came_from[neighbour] = current

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
