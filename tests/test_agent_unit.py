"""Isolated agent tests: no Environment, random generation or file I/O."""
import unittest
from itertools import permutations

from agents.delivery_agent import DeliveryAgent
from simulation.models import ActionType, Letter, LetterInfo, Position
from simulation.observation import AgentObservation
from simulation.sensors import Direction, VisionObservation, VisionSensor


def info(identifier, destination, deadline=120, post=(2, 2), carried=True):
    letter = Letter(identifier, Position(*destination), 0, deadline)
    letter.defer_appearance()
    letter.appear(0, Position(*post))
    if carried:
        letter.mark_picked_up(0)
    return LetterInfo(letter)


def observation(letters=(), position=(2, 2), now=0, capacity=10,
                visible=(), closed=(), posts=((2, 2),)):
    return AgentObservation(
        current_time=now, courier_position=Position(*position),
        depot_position=Position(2, 2), remaining_capacity=capacity,
        letters=tuple(letters), direction=Direction.EAST,
        vision=VisionObservation(frozenset(visible), frozenset(closed)),
        pickup_posts=tuple(Position(*post) for post in posts),
    )


def signature(action):
    return (action.action_type, action.letter_id,
            (action.destination.x, action.destination.y) if action.destination else None)


class PathTests(unittest.TestCase):
    def test_shortest_open_path_and_adjacent_steps(self):
        path = DeliveryAgent(5, 5)._find_path(Position(0, 0), Position(4, 3))
        self.assertEqual(len(path), 7)
        self.assertEqual(path[-1], Position(4, 3))
        for before, after in zip([Position(0, 0), *path], path):
            self.assertEqual(abs(before.x-after.x)+abs(before.y-after.y), 1)

    def test_building_detour(self):
        agent = DeliveryAgent(5, 5, {(1, 0)})
        path = agent._find_path(Position(0, 0), Position(2, 0))
        self.assertEqual(len(path), 4)
        self.assertNotIn(Position(1, 0), path)

    def test_unreachable_target(self):
        agent = DeliveryAgent(5, 5, {(1, y) for y in range(5)})
        self.assertIsNone(agent._find_path(Position(0, 0), Position(2, 0)))

    def test_same_cell_requires_no_move(self):
        self.assertEqual(DeliveryAgent(5, 5)._find_path(Position(1, 1), Position(1, 1)), [])

    def test_invalid_goal(self):
        self.assertIsNone(DeliveryAgent(5, 5)._find_path(Position(1, 1), Position(5, 1)))

    def test_equal_length_route_passes_carried_delivery(self):
        carried = [info(1, (0, 1))]
        path = DeliveryAgent(5, 5)._find_path(Position(0, 0), Position(2, 2), carried)
        self.assertEqual(len(path), 4)
        self.assertIn(Position(0, 1), path)

    def test_known_closure_blocks_route(self):
        agent = DeliveryAgent(5, 5)
        agent._update_memory(observation(visible=((3, 2),), closed=((3, 2),)))
        path = agent._find_path(Position(2, 2), Position(4, 2))
        self.assertEqual(len(path), 4)
        self.assertNotIn(Position(3, 2), path)


class MemoryTests(unittest.TestCase):
    def test_remembers_until_expiry_boundary(self):
        agent = DeliveryAgent(5, 5, closure_memory_minutes=12)
        agent._update_memory(observation(visible=((3, 2),), closed=((3, 2),)))
        agent._update_memory(observation(now=11))
        self.assertIn((3, 2), agent.known_closures)
        agent._update_memory(observation(now=12))
        self.assertNotIn((3, 2), agent.known_closures)

    def test_seen_open_cell_clears_memory_immediately(self):
        agent = DeliveryAgent(5, 5)
        agent._update_memory(observation(visible=((3, 2),), closed=((3, 2),)))
        agent._update_memory(observation(now=1, visible=((3, 2),)))
        self.assertFalse(agent.known_closures)

    def test_repeated_sighting_refreshes_expiry(self):
        agent = DeliveryAgent(5, 5)
        for now in (0, 10):
            agent._update_memory(observation(now=now, visible=((3, 2),), closed=((3, 2),)))
        agent._update_memory(observation(now=12))
        self.assertIn((3, 2), agent.known_closures)
        agent._update_memory(observation(now=22))
        self.assertFalse(agent.known_closures)


class SensorTests(unittest.TestCase):
    def test_forward_side_range_and_no_rear_vision(self):
        vision = VisionSensor(2).observe((3, 3), Direction.EAST, 7, 7, set(), {(2, 3), (6, 3)})
        self.assertEqual(vision.visible_cells, frozenset({(3, 3), (4, 3), (5, 3), (6, 3),
                                                        (3, 2), (3, 1), (3, 4), (3, 5)}))
        self.assertEqual(vision.visible_closures, frozenset({(6, 3)}))

    def test_building_hides_cells_behind_it(self):
        vision = VisionSensor().observe((2, 2), Direction.EAST, 5, 5, {(3, 2)}, {(4, 2)})
        self.assertIn((3, 2), vision.visible_cells)
        self.assertNotIn((4, 2), vision.visible_cells)
        self.assertFalse(vision.visible_closures)


class DecisionTests(unittest.TestCase):
    def test_delivery_precedes_local_pickup(self):
        action = DeliveryAgent(5, 5).choose_action(observation([
            info(1, (2, 2)), info(2, (4, 2), carried=False)]))
        self.assertEqual(signature(action), (ActionType.DELIVER, 1, None))

    def test_pickup_oldest_available_at_current_post_only(self):
        letters = [info(1, (4, 2), 5, post=(0, 0), carried=False),
                   info(2, (4, 2), 30, carried=False), info(3, (4, 2), 20, carried=False)]
        action = DeliveryAgent(5, 5).choose_action(observation(letters))
        self.assertEqual(signature(action), (ActionType.PICK_UP, 2, None))

    def test_full_bag_moves_to_delivery(self):
        action = DeliveryAgent(5, 5).choose_action(observation(
            [info(1, (3, 2)), info(2, (4, 2), carried=False)], capacity=0))
        self.assertEqual(signature(action), (ActionType.MOVE, None, (3, 2)))

    def test_moves_to_remote_pickup_post(self):
        letter = info(1, (0, 0), post=(4, 2), carried=False)
        action = DeliveryAgent(5, 5).choose_action(observation([letter]))
        self.assertEqual(signature(action), (ActionType.MOVE, None, (3, 2)))

    def test_skips_unreachable_post(self):
        agent = DeliveryAgent(5, 5, {(3, y) for y in range(5)})
        letters = [info(1, (0, 0), 10, post=(4, 2), carried=False),
                   info(2, (0, 0), 120, post=(1, 2), carried=False)]
        action = agent.choose_action(observation(letters))
        self.assertEqual(signature(action), (ActionType.MOVE, None, (1, 2)))

    def test_replaces_target_when_new_closure_blocks_it(self):
        agent = DeliveryAgent(5, 5)
        first, second = info(1, (3, 2), 6), info(2, (2, 3), 120)
        agent.choose_action(observation([first, second]))
        self.assertEqual(agent.target_letter_id, 1)
        action = agent.choose_action(observation([first, second], visible=((3, 2),), closed=((3, 2),)))
        self.assertEqual(signature(action), (ActionType.MOVE, None, (2, 3)))
        self.assertEqual(agent.target_letter_id, 2)

    def test_no_work_away_from_posts_waits(self):
        self.assertEqual(DeliveryAgent(5, 5).choose_action(observation(position=(0, 0))).action_type, ActionType.WAIT)

    def test_empty_post_waits_for_new_arrivals(self):
        action = DeliveryAgent(5, 5).choose_action(observation())
        self.assertEqual(action.action_type, ActionType.WAIT)

    def test_waits_if_all_carried_destinations_are_closed(self):
        action = DeliveryAgent(5, 5).choose_action(observation([info(1, (3, 2))],
            visible=((3, 2),), closed=((3, 2),)))
        self.assertEqual(action.action_type, ActionType.WAIT)


class ForecastTests(unittest.TestCase):
    def test_delivery_strategy_is_injected(self):
        from agents.delivery_strategy import DeliveryStrategy
        class FixedStrategy(DeliveryStrategy):
            def __init__(self, result): self.result, self.calls = result, 0
            def select_delivery(self, position, carried, current_time, move_minutes, find_path):
                self.calls += 1
                return self.result
            def score_carried_route(self, position, current_time, carried, first, move_minutes, find_path):
                return (0, 0, current_time)
        target = info(44, (4, 2))
        strategy = FixedStrategy((target, [Position(3, 2), Position(4, 2)]))
        agent = DeliveryAgent(5, 5, delivery_strategy=strategy)
        action = agent.choose_action(observation([target]))
        self.assertEqual(signature(action), (ActionType.MOVE, None, (3, 2)))
        self.assertEqual(strategy.calls, 1)


    def test_exact_deadline_is_on_time_and_lateness_is_minutes(self):
        agent = DeliveryAgent(5, 5)
        letters = [info(1, (1, 0), 6), info(2, (2, 0), 10)]
        self.assertEqual(agent._delivery_strategy.score_carried_route(Position(0, 0), 0, letters, letters[1], agent._move_minutes, agent._find_path), (1, 2, 12))

    def test_multiple_letters_at_one_address_take_no_extra_time(self):
        letters = [info(1, (1, 0), 6), info(2, (1, 0), 6)]
        score = DeliveryAgent(5, 5)._delivery_strategy.score_carried_route(Position(0, 0), 0, letters, letters[0], 6, DeliveryAgent(5, 5)._find_path)
        self.assertEqual(score, (0, 0, 6))

    def test_forecast_does_not_mutate_agent_or_observation(self):
        agent = DeliveryAgent(5, 5)
        carried, waiting = [info(1, (4, 2))], [info(2, (0, 2), 30, post=(1, 2), carried=False)]
        obs = observation(carried+waiting, capacity=9)
        agent.choose_action(obs)
        # Cached paths are disposable computation, not decision memory.
        def state():
            return (agent.target_letter_id, agent.known_closures,
                    agent._pickup_target, repr([l.__dict__ for l in obs.letters]))
        before = state()
        agent._forecast_delivery_score(obs, carried[0], carried, waiting, Position(1, 2))
        self.assertEqual(state(), before)
        self.assertEqual((len(carried), len(waiting)), (1, 1))

    def test_waiting_letter_allowance_starts_at_forecast_pickup(self):
        agent = DeliveryAgent(5, 5)
        carried = info(1, (4, 2), 120)
        waiting = info(2, (0, 2), 18, post=(1, 2), carried=False)
        obs = observation([carried, waiting], capacity=9)
        self.assertEqual(agent._forecast_delivery_score(obs, carried, [carried], [waiting], None), (0, 0))
        self.assertEqual(agent._forecast_delivery_score(obs, carried, [carried], [waiting], Position(1, 2)), (0, 0))
        self.assertIsNone(waiting.deadline)

    def test_no_detour_when_current_urgent_delivery_would_be_late(self):
        letters = [info(1, (3, 2), 6), info(2, (0, 2), 120, post=(1, 2), carried=False)]
        self.assertEqual(signature(DeliveryAgent(5, 5).choose_action(observation(letters, capacity=9))),
                         (ActionType.MOVE, None, (3, 2)))

    def test_small_line_cases_against_independent_exhaustive_oracle(self):
        # Every delivery order is enumerated independently of the agent's search.
        # These controlled cases do not claim global optimality on arbitrary maps.
        for deadlines in ((6, 18, 36), (120, 12, 24), (5, 5, 5)):
            with self.subTest(deadlines=deadlines):
                letters = [info(i+1, (x, 0), d) for i, (x, d) in enumerate(zip((0, 3, 4), deadlines))]
                scores = {}
                for order in permutations(letters):
                    x, now, late, minutes = 2, 0, 0, 0
                    remaining = list(letters)
                    for target in order:
                        if target not in remaining: continue
                        direction = 1 if target.destination.x > x else -1
                        while x != target.destination.x:
                            x += direction; now += 6
                            for letter in list(remaining):
                                if letter.destination.x == x:
                                    delay = max(0, now-letter.deadline)
                                    late += int(delay > 0); minutes += delay
                                    remaining.remove(letter)
                    scores[tuple(l.letter_id for l in order)] = (late, minutes, now)
                agent = DeliveryAgent(5, 1)
                chosen, _ = agent._delivery_strategy.select_delivery(Position(2, 0), letters, 0, agent._move_minutes, agent._find_path)
                actual = agent._delivery_strategy.score_carried_route(Position(2, 0), 0, letters, chosen, agent._move_minutes, agent._find_path)
                self.assertEqual(actual, min(scores.values()))



class JointPlanningTests(unittest.TestCase):
    def test_first_stop_matches_exhaustive_small_pickup_delivery_problem(self):
        from agents.lookahead_strategy import LookaheadStrategy
        # One carried letter, two waiting letters, and only two bag slots.
        # Enumerate all precedence- and capacity-valid action orders independently.
        tasks = {
            'd1': (7, 24), 'p2': (2, None), 'd2': (0, 18),
            'p3': (5, None), 'd3': (6, 36),
        }
        candidates = []
        for order in permutations(tasks):
            bag = {1}
            position, now, late, lateness = 3, 0, 0, 0
            due = {1: 24}
            valid = True
            for action in order:
                identifier = int(action[1])
                destination, deadline = tasks[action]
                if action[0] == 'p':
                    if len(bag) == 2:
                        valid = False
                        break
                    bag.add(identifier)
                elif identifier not in bag:
                    valid = False
                    break
                else:
                    bag.remove(identifier)
                now += abs(destination - position) * 6
                position = destination
                if action[0] == 'p':
                    due[identifier] = now + {2: 18, 3: 36}[identifier]
                else:
                    delay = max(0, now - due[identifier])
                    late += delay > 0
                    lateness += delay
            if valid:
                candidates.append(((late, lateness, now), tasks[order[0]][0]))
        optimum = min(score for score, _ in candidates)
        best_first_stops = {first for score, first in candidates if score == optimum}
        self.assertEqual(optimum[0], 0)
        agent = DeliveryAgent(8, 1)
        obs = observation([
            info(1, (7, 0), 24, post=(3, 0)),
            info(2, (0, 0), 18, post=(2, 0), carried=False),
            info(3, (6, 0), 36, post=(5, 0), carried=False),
        ], position=(3, 0), capacity=1, posts=((2, 0), (5, 0)))
        goal = LookaheadStrategy().select_stop(obs, 6, agent._find_path, frozenset())
        self.assertIn(goal.x, best_first_stops)

    def test_replans_when_letter_deadline_changes(self):
        from agents.lookahead_strategy import LookaheadStrategy
        strategy = LookaheadStrategy()
        agent = DeliveryAgent(5, 5)
        initial = observation([info(1, (3, 2), 6), info(2, (2, 3), 120)])
        changed = observation([info(1, (3, 2), 120), info(2, (2, 3), 6)])
        self.assertEqual(strategy.select_stop(initial, 6, agent._find_path, frozenset()), Position(3, 2))
        self.assertEqual(strategy.select_stop(changed, 6, agent._find_path, frozenset()), Position(2, 3))

    def test_search_width_must_be_positive_integer(self):
        from agents.lookahead_strategy import LookaheadStrategy
        for width in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                LookaheadStrategy(width)


if __name__ == '__main__':
    unittest.main()
