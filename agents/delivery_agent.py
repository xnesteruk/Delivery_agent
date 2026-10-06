from collections import deque

from simulation.models import Action, ActionType, LetterInfo, Position
from simulation.observation import AgentObservation
from agents.delivery_strategy import DeliveryStrategy
from agents.lookahead_strategy import LookaheadStrategy


class DeliveryAgent:
    def __init__(
        self,
        map_width: int,
        map_height: int,
        blocked_cells: set[tuple[int, int]] | None = None,
        move_minutes: int = 6,
        closure_memory_minutes: int = 12,
        delivery_strategy: DeliveryStrategy | None = None,
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

        # current delivery destination
        self._target_letter_id: int | None = None

        # go to a pickup post when its visit improves the result
        self._pickup_target = None
        self._delivery_strategy = (delivery_strategy if delivery_strategy is not None else LookaheadStrategy())
        self._path_cache = {}

    @property
    def known_closures(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._known_closures)

    @property
    def target_letter_id(self) -> int | None:
        return self._target_letter_id

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

    @staticmethod
    def _post(letter: LetterInfo, observation: AgentObservation) -> Position:
        return letter.pickup_position or observation.depot_position

    def _pickup_choices(self, position, waiting, observation, carried):
        choices = []
        seen = set()
        for letter in sorted(waiting, key=lambda item: (item.available_time, item.letter_id)):
            post = self._post(letter, observation)
            cell = (post.x, post.y)
            if cell in seen:
                continue
            seen.add(cell)
            path = self._find_path(position, post, carried)
            if path is not None:
                # Waiting letters have no deadline until pickup.
                priority = (letter.available_time,
                            len(path), cell)
                choices.append((priority, post, path))
        return choices

    def _get_current_delivery(
            self,
            position: Position,
            carried: list[LetterInfo],
            current_time: int,
    ) -> tuple[LetterInfo, list[Position]] | None:
        #loop through and keep first selected lette, and find path to it\\\
        for letter in carried:
            if letter.letter_id == self._target_letter_id:
                path = self._find_path(position, letter.destination, carried)
                if path is not None:
                    return letter, path
                break

        #   previous target is already has path
        self._target_letter_id = None

        choice = self._delivery_strategy.select_delivery(
            position, carried, current_time, self._move_minutes, self._find_path,
        )
        if choice is None:
            return None

        letter, path = choice
        self._target_letter_id = letter.letter_id
        return letter, path

    def _forecast_delivery_score(
            self, observation, current_letter, carried, waiting, pickup_target,
    ) -> tuple[int, int] | None:
        """Compare routes using observed letters only; no future arrivals."""
        position = observation.courier_position
        now = observation.current_time
        capacity = observation.remaining_capacity + len(carried)
        bag, pending = list(carried), list(waiting)
        target_id = current_letter.letter_id if pickup_target is None else None
        late_count = total_lateness = 0
        while bag or pending:
            for letter in list(bag):
                if letter.destination == position:
                    lateness = max(0, now - letter.deadline)
                    late_count += int(lateness > 0)
                    total_lateness += lateness
                    bag.remove(letter)
                    if target_id == letter.letter_id:
                        target_id = None
            for letter in sorted(pending, key=lambda item: (item.available_time, item.letter_id)):
                if len(bag) < capacity and self._post(letter, observation) == position:
                    bag.append(letter.after_pickup(now))
                    pending.remove(letter)
            if any(letter.destination == position for letter in bag):
                continue
            if position == pickup_target:
                pickup_target = None
            if not bag and not pending:
                break
            if pickup_target is not None:
                path = self._find_path(position, pickup_target, bag)
            elif bag:
                target = next((letter for letter in bag if letter.letter_id == target_id), None)
                path = self._find_path(position, target.destination, bag) if target else None
                if path is None:
                    choice = self._delivery_strategy.select_delivery(
                        position, bag, now, self._move_minutes, self._find_path,
                    )
                    if choice is None:
                        return None
                    target, path = choice
                    target_id = target.letter_id
            else:
                posts = self._pickup_choices(position, pending, observation, bag)
                if not posts:
                    return None
                _, pickup_target, path = min(posts, key=lambda item: item[0])
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

    def _find_path(
            self,
            start: Position,
            goal: Position,
            carried: list[LetterInfo] | None = None) -> list[Position] | None:
        """Find the shortest route; break ties by letters delivered on the way.

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

        cache_key = (start_cell, goal_cell, tuple(sorted(deliveries_at.items())))
        if cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            return None if cached_path is None else list(cached_path)

        frontier = deque([start_cell])
        came_from = {start_cell: None}
        distances = {start_cell: 0}
        delivery_counts = {start_cell: 0}

        while frontier:
            current = frontier.popleft()
            if current == goal_cell:
                path = self._restore_path(came_from, goal_cell)
                self._path_cache[cache_key] = tuple(path)
                return path

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

        self._path_cache[cache_key] = None
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

    def choose_action(self, observation: AgentObservation) -> Action:
        self._update_memory(observation)
        # forecasting evaluates many alternative routes
        # reuse paths only within this decision so new observations can never leave a stale route in the cache
        self._path_cache.clear()
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

        # deliver any carried letter at the destin. address
        for letter in carried:
            if letter.destination == position:
                if letter.letter_id == self._target_letter_id:
                    self._target_letter_id = None

                return Action( ActionType.DELIVER, letter_id=letter.letter_id)

        if observation.remaining_capacity > 0:
            local = [letter for letter in waiting
                     if self._post(letter, observation) == position]
            if local:
                letter = min(local, key=lambda item: (item.available_time, item.letter_id))
                return Action( ActionType.PICK_UP, letter_id=letter.letter_id)

        planned_stop = self._delivery_strategy.select_stop(
            observation, self._move_minutes, self._find_path, self.known_closures,
        )
        if planned_stop is not None:
            self._target_letter_id = next(
                (letter.letter_id for letter in carried
                 if letter.destination == planned_stop), None,
            )
            route = self._find_path(position, planned_stop, carried)
            if route:
                return Action( ActionType.MOVE, destination=route[0])

        if self._pickup_target == position or observation.remaining_capacity == 0:
            self._pickup_target = None
        if self._pickup_target is not None:
            if any(self._post(letter, observation) == self._pickup_target for letter in waiting):
                path = self._find_path(position, self._pickup_target, carried)
                if path:
                    return Action(ActionType.MOVE, destination=path[0])
            self._pickup_target = None

        current_choice = self._get_current_delivery(position, carried, observation.current_time)
        posts = self._pickup_choices(position, waiting, observation, carried)
        if current_choice is not None:
            current_letter, path = current_choice
            if observation.remaining_capacity > 0 and posts:
                baseline = self._forecast_delivery_score(
                    observation, current_letter, carried, waiting, None)
                candidates = []
                for _, post, _ in posts:
                    score = self._forecast_delivery_score(
                        observation, current_letter, carried, waiting, post)
                    if score is not None and baseline is not None and score < baseline:
                        candidates.append((score, post))
                if candidates:
                    _, self._pickup_target = min(candidates, key=lambda item: item[0])
                    self._target_letter_id = None
                    route = self._find_path(position, self._pickup_target, carried)
                    if route:
                        return Action(ActionType.MOVE, destination=route[0])
            return Action(ActionType.MOVE, destination=path[0])

        if posts and observation.remaining_capacity > 0:
            _, self._pickup_target, path = min(posts, key=lambda item: item[0])
            if path:
                return Action(ActionType.MOVE, destination=path[0])
        return Action(ActionType.WAIT)
