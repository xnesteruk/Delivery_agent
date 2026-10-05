"""Bounded pickup and delivery planning using observed letters only."""
from dataclasses import dataclass

from agents.delivery_strategy import DeadlineAwareStrategy
from simulation.models import Position


@dataclass(frozen=True)
class _RouteState:
    position: int
    collected: int
    delivered: int
    time: int
    late_count: int
    lateness: int
    first_stop: int
    deadlines: tuple[int | None, ...]

    @property
    def score(self):
        return self.late_count, self.lateness, self.time, self.first_stop


class LookaheadStrategy(DeadlineAwareStrategy):
    """Keep a bounded set of alternative pickup and delivery sequences.

    A route is preferred when it delivers more observed letters on time,
    then causes less total lateness, then finishes sooner. Partial routes
    are ranked using an optimistic bound on unavoidable late deliveries.
    Pruning makes this a heuristic, not an optimality guarantee.

    Inherited delivery-only methods provide the fallback when no complete
    route can be found. Future letters and hidden closures are never queried.
    """

    def __init__(self, beam_width: int = 64):
        if type(beam_width) is not int or beam_width < 1:
            raise ValueError("Beam width must be a positive integer.")
        self._beam_width = beam_width
        self._signature = None
        self._goal = None

    def select_stop(self, observation, move_minutes, find_path, known_closures):
        carried = [letter for letter in observation.letters
                   if letter.is_picked_up and not letter.is_delivered]
        waiting = [letter for letter in observation.letters
                   if not letter.is_picked_up]
        signature = (
            tuple((letter.letter_id, letter.is_picked_up, letter.deadline,
                   letter.delivery_allowance_minutes,
                   letter.destination.x, letter.destination.y,
                   (letter.pickup_position or observation.depot_position).x,
                   (letter.pickup_position or observation.depot_position).y)
                  for letter in observation.letters),
            tuple(sorted(known_closures)),
            observation.remaining_capacity,
        )
        # Walking alone does not trigger a new search. New observations,
        # pickups, deliveries and closure-memory changes invalidate the plan.
        if (signature == self._signature and self._goal is not None
                and self._goal != observation.courier_position
                and find_path(observation.courier_position, self._goal, carried)):
            return self._goal
        self._signature = signature
        self._goal = self._plan(
            observation, carried, waiting, move_minutes, find_path,
        )
        return self._goal

    def _plan(self, observation, carried, waiting, move_minutes, find_path):
        letters = sorted(carried + waiting,
                         key=lambda letter: (letter.deadline if letter.deadline is not None else float("inf"),
                                             letter.available_time, letter.letter_id))
        # An isolated address must not prevent work on reachable letters.
        letters = [letter for letter in letters
                   if find_path(observation.courier_position,
                                letter.destination, None) is not None
                   and (letter.is_picked_up or find_path(
                       observation.courier_position,
                       letter.pickup_position or observation.depot_position,
                       None) is not None)]
        if not letters:
            return None

        cells = [(observation.courier_position.x,
                  observation.courier_position.y)]
        for letter in letters:
            post = letter.pickup_position or observation.depot_position
            for position in (letter.destination, post):
                cell = (position.x, position.y)
                if cell not in cells:
                    cells.append(cell)

        destinations, posts, deadlines, allowances = [], [], [], []
        delivery_masks = [0] * len(cells)
        pickup_masks = [0] * len(cells)
        collected = 0
        for index, letter in enumerate(letters):
            post = letter.pickup_position or observation.depot_position
            destination = cells.index((letter.destination.x, letter.destination.y))
            pickup = cells.index((post.x, post.y))
            destinations.append(destination)
            posts.append(pickup)
            deadlines.append(letter.deadline)
            allowances.append(letter.delivery_allowance_minutes)
            delivery_masks[destination] |= 1 << index
            pickup_masks[pickup] |= 1 << index
            if letter.is_picked_up:
                collected |= 1 << index

        distances = []
        for start in cells:
            row = []
            for goal in cells:
                path = find_path(Position(*start), Position(*goal), None)
                row.append(None if path is None else len(path) * move_minutes)
            distances.append(row)

        # Carried letters excluded for an isolated address still occupy space.
        excluded_carried = len(carried) - collected.bit_count()
        capacity = observation.remaining_capacity + len(carried) - excluded_carried
        full_mask = (1 << len(letters)) - 1
        beam = [_RouteState(0, collected, 0, observation.current_time, 0, 0, -1,
                            tuple(deadlines))]
        best = None

        def rank(state):
            unavoidable = 0
            for index, deadline in enumerate(state.deadlines):
                bit = 1 << index
                if state.delivered & bit:
                    continue
                if state.collected & bit:
                    travel = distances[state.position][destinations[index]]
                else:
                    to_address = distances[posts[index]][destinations[index]]
                    # time starts at pickup
                    unavoidable += to_address is None or to_address > allowances[index]
                    continue
                unavoidable += travel is None or state.time + travel > deadline
            return (state.late_count + unavoidable, state.time,
                    -state.delivered.bit_count(), state.lateness, state.first_stop)

        # Each stop collects or delivers at least one letter: at most 2N stops.
        for _ in range(2 * len(letters) + 1):
            expanded = {}
            for state in beam:
                bag = state.collected & ~state.delivered
                for goal in range(len(cells)):
                    if goal == state.position:
                        continue
                    can_deliver = delivery_masks[goal] & bag
                    can_collect = (bag.bit_count() < capacity
                                   and pickup_masks[goal] & ~state.collected)
                    travel = distances[state.position][goal]
                    if not (can_deliver or can_collect) or travel is None:
                        continue
                    candidate = self._visit(
                        state, goal, travel, delivery_masks[goal],
                        pickup_masks[goal], deadlines, capacity,
                        allowances,
                    )
                    if candidate.delivered == full_mask:
                        if best is None or candidate.score < best.score:
                            best = candidate
                        continue
                    if best is not None and candidate.late_count > best.late_count:
                        continue
                    key = (goal, candidate.collected, candidate.delivered, candidate.deadlines)
                    previous = expanded.get(key)
                    if previous is None or candidate.score < previous.score:
                        expanded[key] = candidate
            if not expanded:
                break
            beam = sorted(expanded.values(), key=rank)[:self._beam_width]

        return Position(*cells[best.first_stop]) if best is not None else None

    @staticmethod
    def _visit(state, goal, travel, delivery_mask, pickup_mask, deadlines, capacity, allowances):
        now = state.time + travel
        deadlines = list(state.deadlines)
        collected, delivered = state.collected, state.delivered
        late_count, lateness = state.late_count, state.lateness

        # Delivery frees bag space before pickup at this stop.
        newly_delivered = delivery_mask & collected & ~delivered
        for index, deadline in enumerate(deadlines):
            if newly_delivered & (1 << index):
                delay = max(0, now - deadline)
                late_count += delay > 0
                lateness += delay
        delivered |= newly_delivered
        slots = capacity - (collected & ~delivered).bit_count()
        # At a post, the agent collects waiting letters by appearance and ID.
        # Waiting entries were sorted in that order by _plan.
        for index in range(len(deadlines)):
            bit = 1 << index
            if slots and pickup_mask & bit and not collected & bit:
                collected |= bit
                deadlines[index] = now + allowances[index]
                slots -= 1

        # Pickup and destination can be the same address.
        newly_delivered = delivery_mask & collected & ~delivered
        for index, deadline in enumerate(deadlines):
            if newly_delivered & (1 << index):
                delay = max(0, now - deadline)
                late_count += delay > 0
                lateness += delay
        delivered |= newly_delivered
        first_stop = goal if state.first_stop < 0 else state.first_stop
        return _RouteState(goal, collected, delivered, now,
                           late_count, lateness, first_stop, tuple(deadlines))
