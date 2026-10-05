from dataclasses import dataclass

from simulation.models import Letter


@dataclass(frozen=True)
class DeliveryMetrics:
    total_letters: int
    available_letters: int
    future_letters: int
    delivered_letters: int
    on_time_letters: int
    late_letters: int
    undelivered_letters: int
    overdue_undelivered_letters: int
    on_time_rate: float
    total_lateness_minutes: int
    average_lateness_minutes: float


class DeliveryEvaluator:
    @staticmethod
    def evaluate(
        letters: list[Letter],
        current_time: int,
    ) -> DeliveryMetrics:
        available = [
            letter
            for letter in letters
            if letter.available_time is not None and letter.available_time <= current_time
        ]

        delivered = [
            letter
            for letter in available
            if letter.is_delivered
        ]

        undelivered = [
            letter
            for letter in available
            if not letter.is_delivered
        ]

        on_time = [
            letter
            for letter in delivered
            if letter.delivery_time <= letter.deadline
        ]

        late = [
            letter
            for letter in delivered
            if letter.delivery_time > letter.deadline
        ]

        total_lateness = sum(
            letter.lateness_minutes
            for letter in late
        )

        return DeliveryMetrics(
            total_letters=len(letters),
            available_letters=len(available),
            future_letters=len(letters) - len(available),
            delivered_letters=len(delivered),
            on_time_letters=len(on_time),
            late_letters=len(late),
            undelivered_letters=len(undelivered),
            overdue_undelivered_letters=sum(
                letter.deadline is not None and current_time > letter.deadline
                for letter in undelivered
            ),
            on_time_rate=(
                len(on_time) / len(letters)
                if letters else 0.0
            ),
            total_lateness_minutes=total_lateness,
            average_lateness_minutes=(
                total_lateness / len(late)
                if late else 0.0
            ),
        )
