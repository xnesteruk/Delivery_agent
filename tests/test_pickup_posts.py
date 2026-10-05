import unittest

from agents.delivery_agent import DeliveryAgent
from experiments.scenario_generator import ScenarioGenerator
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.models import Action, ActionType, Letter, Position
from simulation.runner import Simulation


class PickupTests(unittest.TestCase):
    def test_all_letters_appear_without_post_capacity_limit(self):
        letters = [Letter(i, Position(0, 0), 0, 120) for i in range(20)]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
        self.assertEqual(len(env.get_observation().letters), 20)
        self.assertTrue(all(l.available_time == 0 and l.deadline is None for l in letters))
        for i in range(10):
            env.step(Action(ActionType.PICK_UP, letter_id=i))
        env.step(Action(ActionType.PICK_UP, letter_id=10))
        self.assertFalse(letters[10].is_picked_up)
        self.assertEqual(env.get_observation().remaining_capacity, 0)

    def test_waiting_does_not_use_delivery_allowance(self):
        letter = Letter(1, Position(3, 2), 0, 6)
        env = Environment(SimulationConfig(), [letter])
        for _ in range(150):
            env.step(Action(ActionType.WAIT))
        self.assertIsNone(letter.deadline)
        env.step(Action(ActionType.PICK_UP, letter_id=1))
        env.step(Action(ActionType.MOVE, destination=Position(3, 2)))
        env.step(Action(ActionType.DELIVER, letter_id=1))
        self.assertEqual(letter.deadline, 156)
        self.assertEqual(letter.lateness_minutes, 0)

    def test_arrivals_during_move_have_exact_scheduled_time(self):
        letters = [Letter(i, Position(0, 0), i + 2, 120) for i in range(3)]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
        env.step(Action(ActionType.MOVE, destination=Position(2, 3)))
        self.assertEqual([l.available_time for l in letters], [2, 3, 4])
        self.assertTrue(all(l.deadline is None for l in letters))

    def test_delivery_at_post_needs_no_departure(self):
        letters = [Letter(i, Position(2, 2), 0, 120) for i in range(15)]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
        result = Simulation(env, DeliveryAgent(5, 5), 40).run()
        self.assertTrue(result.finished)
        self.assertEqual(result.simulation_minutes, 0)

    def test_reproducible_random_posts_and_complete_delivery(self):
        config = SimulationConfig(map_width=10, map_height=10)
        generator = ScenarioGenerator(config, set())
        histories = []
        for _ in range(2):
            scenario = generator.generate(seed=31, closure_window_minutes=120,
                post_count=3, letter_count=15, arrival_window_minutes=30,
                building_count=12, closure_count=8)
            letters = scenario.create_letters()
            env = Environment(config, letters, blocked_cells=set(scenario.blocked_cells),
                scheduled_closures=list(scenario.closures), pickup_posts=scenario.pickup_posts,
                arrivals_seed=scenario.arrivals_seed)
            result = Simulation(env, DeliveryAgent(10, 10, set(scenario.blocked_cells)), 2000).run()
            self.assertTrue(result.finished)
            self.assertTrue(all(l.available_time == l.scheduled_time for l in letters))
            self.assertTrue(all(l.deadline == l.pickup_time + 120 for l in letters))
            self.assertTrue(all((l.pickup_position.x, l.pickup_position.y) in scenario.pickup_posts for l in letters))
            histories.append(result.history)
        self.assertEqual(*histories)


if __name__ == '__main__':
    unittest.main()
