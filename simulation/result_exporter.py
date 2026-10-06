import csv
from pathlib import Path

from simulation.runner import SimulationResult


class ResultExporter:
    @staticmethod
    def save_letters(letters, path: str | Path) -> None:
        """Export all Letter objects"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "letter_id", "post_x", "post_y", "destination_x", "destination_y",
            "available_time", "pickup_time", "delivery_allowance_minutes",
            "deadline", "delivery_time", "lateness_minutes", "status",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for letter in letters:
                allowance = letter.delivery_allowance_minutes
                if letter.available_time is None:
                    status = "Очікує появи"
                elif not letter.is_picked_up:
                    status = "Очікує отримання"
                elif not letter.is_delivered:
                    status = "У кур’єра"
                elif letter.lateness_minutes > 0:
                    status = f"Запізнення на {letter.lateness_minutes} хв"
                else:
                    status = "Вчасно"

                writer.writerow({
                    "letter_id": letter.letter_id,
                    "post_x": letter.pickup_position.x if letter.pickup_position else None,
                    "post_y": letter.pickup_position.y if letter.pickup_position else None,
                    "destination_x": letter.destination.x,
                    "destination_y": letter.destination.y,
                    "available_time": letter.scheduled_time,
                    "pickup_time": letter.pickup_time,
                    "delivery_time": letter.delivery_time,
                    "deadline": letter.deadline,
                    "delivery_allowance_minutes": allowance,
                    "lateness_minutes": letter.lateness_minutes,
                    "status": status,
                })

    @staticmethod
    def save_history(
        result: SimulationResult,
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "step",
            "time_before",
            "time_after",
            "action",
            "x",
            "y",
            "remaining_capacity",
            "message",
        ]

        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for record in result.history:
                writer.writerow({
                    "step": record.step,
                    "time_before": record.time_before,
                    "time_after": record.time_after,
                    "action": record.action,
                    "x": record.position[0],
                    "y": record.position[1],
                    "remaining_capacity": record.remaining_capacity,
                    "message": record.message,
                })
