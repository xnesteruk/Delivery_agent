from dataclasses import dataclass
from time import perf_counter

from agents.delivery_agent import DeliveryAgent
from simulation.environment import Environment


@dataclass(frozen=True)
class StepRecord:
    step: int
    time_before: int
    time_after: int
    action: str
    position: tuple[int, int]
    remaining_capacity: int
    message: str | None


@dataclass(frozen=True)
class SimulationResult:
    finished: bool
    stop_reason: str
    steps: int
    simulation_minutes: int
    execution_seconds: float
    history: tuple[StepRecord, ...]


class Simulation:
    def __init__(
        self,
        environment: Environment,
        agent: DeliveryAgent,
        max_steps: int = 1000,
    ):
        if max_steps <= 0:
            raise ValueError("Step limit must be positive.")

        self._environment = environment
        self._agent = agent
        self._max_steps = max_steps
        self._has_run = False

    def run(self) -> SimulationResult:
        if self._has_run:
            raise RuntimeError(
                "Create a new simulation, environment and agent "
                "for another experiment."
            )

        self._has_run = True
        history = []
        started_at = perf_counter()

        for step in range(1, self._max_steps + 1):
            if self._environment.is_finished():
                break

            observation = self._environment.get_observation()
            action = self._agent.choose_action(observation)
            result = self._environment.step(action)

            history.append(
                StepRecord(
                    step=step,
                    time_before=observation.current_time,
                    time_after=result.current_time,
                    action=action.action_type.value,
                    position=(
                        result.courier_position.x,
                        result.courier_position.y,
                    ),
                    remaining_capacity=result.remaining_capacity,
                    message=result.last_action_result,
                )
            )

        execution_seconds = perf_counter() - started_at
        finished = self._environment.is_finished()

        return SimulationResult(
            finished=finished,
            stop_reason=(
                "all_delivered" if finished else "step_limit"
            ),
            steps=len(history),
            simulation_minutes=self._environment.current_time,
            execution_seconds=execution_seconds,
            history=tuple(history),
        )