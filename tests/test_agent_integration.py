"""Controlled integration scenarios, not the final statistical experiments."""
import unittest
from collections import Counter

from agents.delivery_agent import DeliveryAgent
from simulation.config import SimulationConfig
from simulation.environment import Environment
from simulation.evaluation import DeliveryEvaluator
from simulation.models import Action, ActionType, Letter, Position
from simulation.road_closures import RoadClosure
from simulation.runner import Simulation


def letter(identifier, destination, arrival=0, allowance=120):
    return Letter(identifier, Position(*destination), arrival, allowance)


def action_signature(action):
    return (action.action_type, action.letter_id,
            (action.destination.x, action.destination.y) if action.destination else None)


class AgentIntegrationTests(unittest.TestCase):
    def test_delivers_on_route_and_logs_exact_time(self):
        letters = [letter(1, (3, 2), allowance=6), letter(2, (4, 2), allowance=12)]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
        result = Simulation(env, DeliveryAgent(5, 5), 30).run()
        self.assertTrue(result.finished)
        self.assertEqual([l.delivery_time for l in letters], [6, 12])
        self.assertEqual([r.action for r in result.history],
                         ['pick_up', 'pick_up', 'move', 'deliver', 'move', 'deliver'])
        self.assertEqual(result.simulation_minutes, 12)
        self.assertEqual(DeliveryEvaluator.evaluate(letters, 12).on_time_letters, 2)

    def test_waits_for_closed_corridor_then_finishes(self):
        config = SimulationConfig(map_width=5, map_height=1)
        letters = [letter(1, (4, 0))]
        env = Environment(config, letters, pickup_posts=((2, 0),),
                          scheduled_closures=[RoadClosure((3, 0), 0, 10)])
        agent = DeliveryAgent(5, 1)
        result = Simulation(env, agent, 50).run()
        self.assertTrue(result.finished)
        self.assertEqual(sum(r.action == 'wait' for r in result.history), 10)
        self.assertEqual(letters[0].delivery_time, 22)
        self.assertNotIn((3, 0), agent.known_closures)
        self.assertFalse(any('failed' in (r.message or '') for r in result.history))

    def test_serves_other_letter_before_temporarily_blocked_address(self):
        letters = [letter(1, (3, 2), allowance=6), letter(2, (2, 3))]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),),
                          scheduled_closures=[RoadClosure((3, 2), 0, 18)])
        result = Simulation(env, DeliveryAgent(5, 5), 100).run()
        self.assertTrue(result.finished)
        self.assertLess(letters[1].delivery_time, letters[0].delivery_time)
        self.assertGreaterEqual(letters[0].delivery_time, 18)

    def test_future_letters_do_not_leak_into_actions(self):
        # The worlds differ only in information still hidden from the agent.
        worlds = []
        for destination in ((0, 0), (4, 4)):
            letters = [letter(1, (4, 2)), letter(2, destination, arrival=60)]
            env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
            worlds.append((env, DeliveryAgent(5, 5)))
        for _ in range(12):
            actions = []
            for env, agent in worlds:
                obs = env.get_observation()
                self.assertLess(obs.current_time, 60)
                self.assertEqual([l.letter_id for l in obs.letters], [1])
                action = agent.choose_action(obs)
                actions.append(action_signature(action))
                env.step(action)
            self.assertEqual(*actions)

    def test_observation_is_a_snapshot_not_live_letter_state(self):
        letters = [letter(1, (3, 2))]
        env = Environment(SimulationConfig(), letters)
        before = env.get_observation()
        env.step(Action(ActionType.PICK_UP, letter_id=1))
        self.assertFalse(before.letters[0].is_picked_up)
        self.assertTrue(env.get_observation().letters[0].is_picked_up)

    def test_future_arrivals_keep_simulation_alive(self):
        letters = [letter(1, (3, 2), arrival=20)]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
        self.assertFalse(env.is_finished())
        result = Simulation(env, DeliveryAgent(5, 5), 100).run()
        self.assertTrue(result.finished)
        self.assertEqual(letters[0].available_time, 20)
        self.assertGreaterEqual(letters[0].pickup_time, 20)

    def test_three_posts_and_courier_capacity(self):
        config = SimulationConfig(carrying_capacity=2)
        letters = [letter(i, ((i * 2) % 5, (i * 3) % 5)) for i in range(15)]
        env = Environment(config, letters, pickup_posts=((2, 2), (0, 2), (4, 2)), arrivals_seed=31)
        agent = DeliveryAgent(5, 5)
        for _ in range(400):
            obs = env.get_observation()
            self.assertLessEqual(sum(l.is_picked_up and not l.is_delivered for l in obs.letters), 2)
            if env.is_finished():
                break
            after = env.step(agent.choose_action(obs))
            self.assertNotIn("failed", after.last_action_result or "")
        self.assertTrue(env.is_finished())
        self.assertTrue(all(l.available_time == l.scheduled_time for l in letters))
        self.assertTrue(all(l.deadline == l.pickup_time + 120 for l in letters))

    def test_new_letters_appear_while_courier_stays_at_post(self):
        letters = [letter(1, (2, 2)), letter(2, (2, 2), arrival=1)]
        env = Environment(SimulationConfig(), letters, pickup_posts=((2, 2),))
        env.step(Action(ActionType.PICK_UP, letter_id=1))
        env.step(Action(ActionType.DELIVER, letter_id=1))
        env.step(Action(ActionType.WAIT))
        self.assertEqual(letters[1].available_time, 1)
        self.assertIsNone(letters[1].deadline)
        env.step(Action(ActionType.PICK_UP, letter_id=2))
        self.assertEqual(letters[1].deadline, 121)
        self.assertEqual(env.get_observation().courier_position, Position(2, 2))

    def test_permanent_unreachable_goal_stops_at_step_limit(self):
        letters = [letter(1, (4, 2))]
        buildings = {(3, y) for y in range(5)}
        env = Environment(SimulationConfig(), letters, blocked_cells=buildings)
        result = Simulation(env, DeliveryAgent(5, 5, buildings), max_steps=15).run()
        self.assertFalse(result.finished)
        self.assertEqual(result.stop_reason, 'step_limit')
        self.assertEqual(result.steps, 15)
        self.assertFalse(letters[0].is_delivered)
        metrics = DeliveryEvaluator.evaluate(letters, result.simulation_minutes)
        self.assertEqual(metrics.undelivered_letters, 1)


if __name__ == '__main__':
    unittest.main()
