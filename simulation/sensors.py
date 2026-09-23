from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)


@dataclass(frozen=True)
class VisionObservation:
    visible_cells: frozenset[tuple[int, int]]
    visible_closures: frozenset[tuple[int, int]]


class VisionSensor:
    def __init__(self, side_range: int = 2):
        if side_range < 0:
            raise ValueError("Side range cannot be negative.")

        self._side_range = side_range

    def observe(
        self,
        position: tuple[int, int],
        direction: Direction,
        map_width: int,
        map_height: int,
        buildings: set[tuple[int, int]] | frozenset[tuple[int, int]],
        closures: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> VisionObservation:
        x, y = position

        if not (0 <= x < map_width and 0 <= y < map_height):
            raise ValueError("Sensor position is outside the map.")

        if position in buildings:
            raise ValueError("Sensor cannot be inside a building.")

        forward_x, forward_y = direction.value

        left = (forward_y, -forward_x)
        right = (-forward_y, forward_x)

        visible = {position}

        visible.update(
            self._trace_ray(
                position,
                (forward_x, forward_y),
                max(map_width, map_height),
                map_width,
                map_height,
                buildings,
            )
        )

        for side in (left, right):
            visible.update(
                self._trace_ray(
                    position,
                    side,
                    self._side_range,
                    map_width,
                    map_height,
                    buildings,
                )
            )

        return VisionObservation(
            visible_cells=frozenset(visible),
            visible_closures=frozenset(
                cell for cell in visible if cell in closures
            ),
        )

    def _trace_ray(
        self,
        start: tuple[int, int],
        direction: tuple[int, int],
        distance: int,
        map_width: int,
        map_height: int,
        buildings: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        visible = set()
        x, y = start
        dx, dy = direction

        for _ in range(distance):
            x += dx
            y += dy

            if not (0 <= x < map_width and 0 <= y < map_height):
                break

            cell = (x, y)
            visible.add(cell)

            if cell in buildings:
                break

        return visible