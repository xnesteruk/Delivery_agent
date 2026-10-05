from collections import deque
from dataclasses import dataclass
from random import Random

from experiments.seeds import derive_seed
from simulation.config import SimulationConfig
from simulation.models import Letter, Position
from simulation.road_closures import RoadClosure


@dataclass(frozen=True)
class LetterSpec:
    letter_id: int
    destination: tuple[int, int]
    available_time: int
    delivery_allowance_minutes: int

    def create_letter(self) -> Letter:
        return Letter(
            letter_id=self.letter_id,
            destination=Position(*self.destination),
            available_time=self.available_time,
            delivery_allowance_minutes=self.delivery_allowance_minutes,
        )


@dataclass(frozen=True)
class Scenario:
    seed: int
    pickup_posts: tuple[tuple[int, int], ...]
    arrivals_seed: int
    letters_seed: int
    closures_seed: int
    letters: tuple[LetterSpec, ...]
    closures: tuple[RoadClosure, ...]
    blocked_cells: tuple[tuple[int, int], ...]

    def create_letters(self) -> list[Letter]:
        return [
            specification.create_letter()
            for specification in self.letters
        ]


class ScenarioGenerator:
    def __init__(
            self,
            config: SimulationConfig,
            blocked_cells: set[tuple[int, int]]
    ):
        self._config = config
        self._blocked_cells = frozenset(
            blocked_cells if blocked_cells is not None else ()
        )

        for x, y in self._blocked_cells:
            if not (
                    0 <= x < config.map_width
                    and 0 <= y < config.map_height
            ):
                raise ValueError("Building is outside the map.")

        if config.depot_position in self._blocked_cells:
            raise ValueError("Depot cannot be inside a building.")

        reachable = self._find_reachable_cells()

        # Sorting makes random choices independent of set iteration order.
        self._destinations = sorted(
            reachable - {config.depot_position}
        )

        if not self._destinations:
            raise ValueError("No reachable delivery addresses.")

    def generate(
            self,
            seed: int,
            closure_window_minutes: int,
            letter_count: int = 10,
            closure_count: int = 4,
            arrival_window_minutes: int = 120,
            min_closure_minutes: int = 6,
            max_closure_minutes: int = 24,
            building_count: int = 0,
            post_count: int = 1,
    ) -> Scenario:
        if letter_count <= 0:
            raise ValueError("Letter count must be positive.")

        if closure_count < 0:
            raise ValueError("Closure count cannot be negative.")

        if arrival_window_minutes < 0:
            raise ValueError("Arrival window cannot be negative.")

        if letter_count - 1 > arrival_window_minutes:
            raise ValueError(
                "Arrival window is too short for unique letter arrival times."
            )

        if not 0 < min_closure_minutes <= max_closure_minutes:
            raise ValueError("Invalid closure duration range.")
        if closure_window_minutes < 0:
            raise ValueError("Closure window cannot be negative.")

        if type(building_count) is not int or building_count < 0:
            raise ValueError("Building count must be a non-negative integer.")
        if building_count:
            blocked = self._generate_buildings(seed, building_count)
            # A separate generator keeps this object's state unchanged.
            return ScenarioGenerator(self._config, blocked).generate(
                post_count=post_count, seed=seed, letter_count=letter_count, closure_count=closure_count,
                arrival_window_minutes=arrival_window_minutes,
                min_closure_minutes=min_closure_minutes,
                max_closure_minutes=max_closure_minutes,
                closure_window_minutes=closure_window_minutes,
            )

        reachable = sorted(self._find_reachable_cells())
        if type(post_count) is not int or not 1 <= post_count <= len(reachable):
            raise ValueError("Post count must fit reachable street cells.")
        posts = tuple(sorted(Random(derive_seed(seed, "posts")).sample(reachable, post_count)))
        closure_cells = [cell for cell in self._destinations if cell not in posts]
        if closure_count and not closure_cells:
            raise ValueError("No non-post cells available for closures.")
        letters_seed = derive_seed(seed, "letters")
        closures_seed = derive_seed(seed, "closures")

        letters_random = Random(letters_seed)
        closures_random = Random(closures_seed)

        # Keep the first letter available at minute 0, assign each later a unique minute in a run
        arrival_times = [0] + letters_random.sample(
            range(1, arrival_window_minutes + 1), letter_count - 1
        )

        letters = []

        for letter_id in range(1, letter_count + 1):
            destination = letters_random.choice(self._destinations)

            available_time = arrival_times[letter_id - 1]

            letters.append(
                LetterSpec(
                    letter_id=letter_id,
                    destination=destination,
                    available_time=available_time,
                    delivery_allowance_minutes=(
                        self._config.delivery_allowance_minutes
                    ),
                )
            )

        closures = []

        for _ in range(closure_count):
            cell = closures_random.choice(closure_cells)

            start = closures_random.randint(
                0, closure_window_minutes
            )
            duration = closures_random.randint(
                min_closure_minutes,
                max_closure_minutes,
            )

            closures.append(
                RoadClosure(
                    cell=cell,
                    start_minute=start,
                    end_minute=start + duration,
                )
            )

        return Scenario(
            seed=seed,
            pickup_posts=posts,
            arrivals_seed=derive_seed(seed, "post_assignments"),
            letters_seed=letters_seed,
            closures_seed=closures_seed,
            letters=tuple(letters),
            closures=tuple(closures),
            blocked_cells=tuple(sorted(self._blocked_cells)),
        )

    def _generate_buildings(self, seed: int, count: int) -> set[tuple[int, int]]:
        """Place buildings while keeping all remaining street cells connected."""
        blocked = set(self._blocked_cells)
        candidates = [
            (x, y) for x in range(self._config.map_width)
            for y in range(self._config.map_height)
            if (x, y) != self._config.depot_position and (x, y) not in blocked
        ]
        if count >= len(candidates):
            raise ValueError("Leave at least one delivery address besides the depot.")
        Random(derive_seed(seed, "buildings")).shuffle(candidates)
        total_cells = self._config.map_width * self._config.map_height
        added = 0
        for cell in candidates:
            trial = blocked | {cell}
            if len(self._find_reachable_cells(trial)) == total_cells - len(trial):
                blocked = trial
                added += 1
                if added == count:
                    return blocked
        raise ValueError("Cannot place this many buildings without disconnecting streets.")

    def _find_reachable_cells(
            self, blocked_cells: set[tuple[int, int]] | None = None,
    ) -> set[tuple[int, int]]:
        blocked = self._blocked_cells if blocked_cells is None else blocked_cells
        depot = self._config.depot_position
        reachable = {depot}
        frontier = deque([depot])

        while frontier:
            x, y = frontier.popleft()

            for neighbour in (
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
            ):
                nx, ny = neighbour

                if not (
                        0 <= nx < self._config.map_width
                        and 0 <= ny < self._config.map_height
                ):
                    continue

                if neighbour in blocked:
                    continue

                if neighbour in reachable:
                    continue

                reachable.add(neighbour)
                frontier.append(neighbour)

        return reachable
