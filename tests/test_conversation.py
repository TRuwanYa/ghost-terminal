import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "conversation.py"
spec = importlib.util.spec_from_file_location("conversation_module", MODULE_PATH)
conversation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversation_module)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_generate_response_requests_name_on_first_contact(self):
        state = conversation_module.create_state_from_memory()
        response = conversation_module.generate_response("hello", state=state)

        self.assertIn("name", response.lower())
        self.assertEqual(state["relationship"]["conversation_count"], 1)

    def test_generate_response_remembers_user_name(self):
        state = conversation_module.create_state_from_memory()
        conversation_module.generate_response("my name is Ava", state=state)

        self.assertEqual(state["user"]["name"], "Ava")
        self.assertIn("Ava", state["user"]["name"])


if __name__ == "__main__":
    unittest.main()
