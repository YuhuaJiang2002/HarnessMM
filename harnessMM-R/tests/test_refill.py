"""Replay the blocked first pick from the two training layouts."""
import json
from pathlib import Path
import unittest

from tabletop import Tabletop


class RefillExecutionTest(unittest.TestCase):
    def test_occupied_destination_blocks_pick_until_cleared(self):
        recipes = Path(__file__).resolve().parents[1] / "scenes.json"
        env = Tabletop(json.loads(recipes.read_text())["clear-and-refill"])
        self.addCleanup(env.renderer.close)
        for ordinal in (2, 5):
            with self.subTest(ordinal=ordinal):
                env.reset([20261122, 0, ordinal])
                before = env.observe()
                occupant = next(b for b, pad in before["resting_on"].items() if pad == "back")
                incoming = next(b for b, pad in env.goals.items() if pad == "back")
                reason, ticks = env.step(f"pick {incoming}")
                self.assertIn(occupant, reason)
                self.assertEqual(ticks, [])
                self.assertEqual(env.observe(), before)
                for action in (f"pick {occupant}", "place front", f"pick {incoming}", "place back"):
                    feedback, trace = env.step(action)
                    self.assertEqual(feedback, "executed")
                    self.assertTrue(trace)
                self.assertIsNone(env.held)
                for b, pad in env.goals.items():
                    self.assertEqual(env.resting_on(b), pad)
