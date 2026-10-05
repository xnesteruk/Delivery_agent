import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import run_experiment
from simulation.evaluation import DeliveryEvaluator
from simulation.models import Letter, Position
from simulation.result_exporter import ResultExporter



class ResultOutputTests(unittest.TestCase):
    def test_csv_has_simple_deadline_math_and_distinct_pending_statuses(self):
        waiting = Letter(1, Position(4, 2), 0, 120)
        carried = Letter(2, Position(3, 2), 0, 120)
        late = Letter(3, Position(0, 2), 0, 120)
        future = Letter(4, Position(1, 2), 500, 120)
        for letter in (waiting, carried, late):
            letter.defer_appearance()
            letter.appear(0, Position(2, 2))
        future.defer_appearance()
        carried.mark_picked_up(200)
        late.mark_picked_up(72)
        late.mark_delivered(205)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "letters.csv"
            ResultExporter.save_letters([waiting, carried, late, future], path)
            with path.open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        self.assertNotIn("elapsed_delivery_minutes", rows[0])
        self.assertNotIn("elapsed_vs_limit", rows[0])
        self.assertNotIn("scheduled_time", rows[0])
        self.assertEqual(rows[0]["deadline"], "")
        self.assertEqual(rows[1]["deadline"], "320")
        self.assertEqual(rows[2]["deadline"], "192")
        self.assertEqual(rows[2]["lateness_minutes"], "13")
        self.assertEqual(rows[2]["destination_x"], "0")
        self.assertEqual(rows[2]["post_x"], "2")
        self.assertEqual(rows[3]["available_time"], "500")
        self.assertEqual([r["status"] for r in rows],
                         ["Очікує отримання", "У кур’єра", "Запізнення на 13 хв", "Очікує появи"])

    def test_future_letters_do_not_inflate_success_rate(self):
        letters = [Letter(i, Position(1, 0), 500 if i else 0, 120) for i in range(10)]
        letters[0].mark_picked_up(0)
        letters[0].mark_delivered(6)
        metrics = DeliveryEvaluator.evaluate(letters, 6)
        self.assertEqual(metrics.on_time_rate, 0.1)
        self.assertEqual(metrics.total_letters, 10)
        self.assertEqual(metrics.future_letters, 9)

    def test_exported_batch_reconciles_without_summary_csv(self):
        config = {
            "run_count": 2, "max_steps": 300, "pickup_posts": {"count": 3},
            "config": {"map_width": 5, "map_height": 5, "seed": 31},
            "generation": {"letter_count": 6, "closure_count": 0,
                           "arrival_window_minutes": 12, "closure_window_minutes": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset = root / "config.json"
            preset.write_text(json.dumps(config))
            with redirect_stdout(io.StringIO()):
                batch = run_experiment(preset, root / "results")
            self.assertEqual(list(batch.rglob("summary.csv")), [])
            with (batch / "runs.csv").open(encoding="utf-8-sig") as file:
                runs = list(csv.DictReader(file))
            with (batch / "aggregate.csv").open(encoding="utf-8-sig") as file:
                aggregates = {r["metric"]: r for r in csv.DictReader(file)}
            for run in runs:
                path = batch / f"run_{int(run['run']):03d}" / "letters.csv"
                with path.open(encoding="utf-8-sig") as file:
                    letters = list(csv.DictReader(file))
                self.assertNotIn("available_letters", run)
                self.assertNotIn("future_letters", run)
                late = sum(
                        bool(r["delivery_time"]) and int(r["lateness_minutes"]) > 0 for r in letters
                )
                self.assertEqual(
                    sum(int(run[k]) for k in (
                        "on_time_letters",
                        "awaiting_delivery_letters",
                        "not_yet_appeared_letters",
                    )) + late,
                    len(letters),
                )
                on_time = sum(r["status"] == "Вчасно" for r in letters)
                self.assertEqual(on_time, int(run["on_time_letters"]))
                self.assertEqual(float(run["on_time_rate"]), on_time / len(letters))
                for row in letters:
                    self.assertEqual(int(row["deadline"]),
                                     int(row["pickup_time"]) + int(row["delivery_allowance_minutes"]))
                    self.assertEqual(int(row["lateness_minutes"]),
                                     max(0, int(row["delivery_time"]) - int(row["deadline"])))
            expected = sum(float(r["on_time_rate"]) for r in runs) / len(runs)
            self.assertAlmostEqual(float(aggregates["on_time_rate"]["mean"]), expected)


if __name__ == '__main__':
    unittest.main()
