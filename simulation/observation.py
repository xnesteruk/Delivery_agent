from dataclasses import dataclass

from simulation.models import LetterInfo, Position
from simulation.sensors import Direction, VisionObservation


@dataclass(frozen=True)
class AgentObservation:
    current_time: int
    courier_position: Position
    depot_position: Position
    remaining_capacity: int
    letters: tuple[LetterInfo, ...]
    direction: Direction
    vision: VisionObservation
    last_action_result: str | None = None
    pickup_posts: tuple[Position, ...] = ()
