from dataclasses import dataclass


@dataclass(frozen=True)
class RoadClosure:
    cell: tuple[int, int]
    start_minute: int
    end_minute: int

    def __post_init__(self):
        if self.start_minute < 0:
            raise ValueError("Start time cannot be negative.")

        if self.end_minute <= self.start_minute:
            raise ValueError("End time must be after start time.")

    def is_active(self, current_time: int) -> bool:
        return self.start_minute <= current_time < self.end_minute


class ClosureSchedule:
    def __init__(self, closures: list[RoadClosure]):
        self._closures = tuple(closures)

    def active_cells(self, current_time: int) -> set[tuple[int, int]]:
        return {
            closure.cell
            for closure in self._closures
            if closure.is_active(current_time)
        }

    @property
    def affected_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            closure.cell
            for closure in self._closures
        )