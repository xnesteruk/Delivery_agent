"""Pluggable policies for selecting and forecasting letter deliveries."""
from abc import ABC, abstractmethod
from collections.abc import Callable

from simulation.models import LetterInfo, Position

PathFinder = Callable[[Position, Position, list[LetterInfo] | None], list[Position] | None]


class DeliveryStrategy(ABC):
    """Interface for an agent's delivery-order policy."""

    def select_stop(self, observation, move_minutes, find_path, known_closures):
        """Optionally plan a pickup or delivery stop; None uses delivery routing."""
        return None

    @abstractmethod
    def select_delivery(
        self,
        position: Position,
        carried: list[LetterInfo],
        current_time: int,
        move_minutes: int,
        find_path: PathFinder,
    ) -> tuple[LetterInfo, list[Position]] | None:
        raise NotImplementedError

    @abstractmethod
    def score_carried_route(
        self,
        position: Position,
        current_time: int,
        carried: list[LetterInfo],
        first: LetterInfo,
        move_minutes: int,
        find_path: PathFinder,
    ) -> tuple[int, int, int] | None:
        raise NotImplementedError


class DeadlineAwareStrategy(DeliveryStrategy):
    """Choose a first delivery, then continue by earliest feasible slack."""

    @staticmethod
    def _delivery_choices(position, letters, current_time, move_minutes, find_path, carried=None):
        choices = []
        for letter in letters:
            path = find_path(position, letter.destination, carried)
            if path is None:
                continue
            travel = len(path) * move_minutes
            slack = letter.deadline - current_time - travel
            if slack >= 0:
                priority = (0, slack, travel, letter.letter_id)
            else:
                priority = (1, travel, letter.deadline, letter.letter_id)
            choices.append((priority, letter, path))
        return choices

    def select_delivery(self, position, carried, current_time, move_minutes, find_path):
        choices = self._delivery_choices(position, carried, current_time, move_minutes, find_path, carried=carried)
        if not choices:
            return None
        candidates = []
        for priority, letter, path in choices:
            score = self.score_carried_route(position, current_time, carried, letter, move_minutes, find_path)
            if score is not None:
                candidates.append((score, priority, letter, path))
        if not candidates:
            _, letter, path = min(choices, key=lambda item: item[0])
            return letter, path
        _, _, letter, path = min(candidates, key=lambda item: item[:2])
        return letter, path

    def score_carried_route(self, position, current_time, carried, first, move_minutes, find_path):
        remaining = list(carried)
        target = first
        late_count = total_lateness = 0
        while remaining:
            path = find_path(position, target.destination, remaining)
            if path is None:
                return None
            for index, cell in enumerate([position, *path]):
                if index:
                    current_time += move_minutes
                for letter in list(remaining):
                    if letter.destination == cell:
                        lateness = max(0, current_time - letter.deadline)
                        late_count += int(lateness > 0)
                        total_lateness += lateness
                        remaining.remove(letter)
            position = target.destination
            if remaining:
                choices = self._delivery_choices(position, remaining, current_time, move_minutes, find_path, carried=remaining)
                if not choices:
                    return None
                _, target, _ = min(choices, key=lambda item: item[0])
        return late_count, total_lateness, current_time
