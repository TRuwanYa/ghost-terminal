import json
import os
import tempfile
import unittest
from pathlib import Path

import importlib.util


MODULE_PATH = Path(__file__).resolve().parents[1] / "memory.py"
spec = importlib.util.spec_from_file_location("memory_module", MODULE_PATH)
memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_module)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_create_and_save_state(self):
        state = memory_module.create_new_state()
        self.assertIn("ghost", state)
        self.assertIn("user", state)

        saved = memory_module.update_user_profile({
            "name": "Ava",
            "nickname": "Star",
            "facts": {"likes": ["coffee"], "dislikes": ["loud rooms"]},
            "last_known_emotion": "curious",
        })

        self.assertEqual(saved["user"]["name"], "Ava")
        self.assertEqual(saved["user"]["nickname"], "Star")
        self.assertEqual(saved["user"]["facts"]["likes"], ["coffee"])

        loaded = memory_module.load_state()
        self.assertEqual(loaded["user"]["name"], "Ava")
        self.assertEqual(loaded["user"]["last_known_emotion"], "curious")


if __name__ == "__main__":
    unittest.main()
