from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dialogue import GhostDialogue
from memory import MemoryStore


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "test.db"
        self.memory = MemoryStore(self.db)
        self.dialogue = GhostDialogue(self.memory)
        self.dialogue.start()

    def finish_introduction(self):
        for text in ["Casper", "yes", "Ruwanya", "yes"]:
            self.dialogue.reply(text)

    def test_complete_introduction_and_returning_greeting(self):
        self.finish_introduction()
        self.assertEqual(self.memory.get_ghost_name(), "Casper")
        self.assertEqual(self.memory.get_user_name(), "Ruwanya")
        self.assertIsNone(self.memory.get_dialogue_state())
        greeting = GhostDialogue(MemoryStore(self.db)).start()
        self.assertIn("Back again, Ruwanya", greeting)
        self.assertIn("Casper", greeting)

    def test_suggestion_is_not_a_committed_name(self):
        self.dialogue.reply("Casper")
        self.assertIsNone(self.memory.get_ghost_name())
        self.assertEqual(self.memory.get_dialogue_state()["pending_value"], "Casper")

    def test_reject_ghost_name_then_accept_another(self):
        self.dialogue.reply("Casper")
        self.dialogue.reply("no")
        self.assertIsNone(self.memory.get_dialogue_state()["pending_value"])
        self.dialogue.reply("Ash")
        self.dialogue.reply("YES!")
        self.assertEqual(self.memory.get_ghost_name(), "Ash")

    def test_restart_at_each_introduction_stage(self):
        for entry in ["Casper", "yes", "Ruwanya", "yes"]:
            self.dialogue = GhostDialogue(MemoryStore(self.db))
            self.dialogue.start()
            self.dialogue.reply(entry)
        self.assertEqual(self.memory.get_user_name(), "Ruwanya")
        self.assertIsNone(self.memory.get_dialogue_state())

    def test_invalid_confirmation_does_not_change_state(self):
        self.dialogue.reply("Casper")
        before = self.memory.get_dialogue_state()
        self.assertIn("yes or no", self.dialogue.reply("maybe later"))
        self.assertEqual(self.memory.get_dialogue_state(), before)

    def test_commands_are_not_saved_as_names(self):
        for text in ["/help", "/whoami", "/about", "/unknown", "", "yes", "My name is Ava", "bad\x1bname"]:
            self.dialogue.reply(text)
        self.assertIsNone(self.memory.get_ghost_name())
        self.assertEqual(self.memory.get_dialogue_state()["pending_question_id"], "awaiting_ghost_name")

    def test_user_name_confirmation_and_correction(self):
        self.dialogue.reply("Casper")
        self.dialogue.reply("yes")
        self.dialogue.reply("Ava")
        self.assertIsNone(self.memory.get_user_name())
        self.dialogue.reply("no")
        self.dialogue.reply("Ruwanya")
        self.dialogue.reply("yes")
        self.dialogue.reply("/name Vimu")
        self.assertEqual(self.memory.get_user_name(), "Vimu")
        self.assertIsNone(self.memory.get_dialogue_state())

    def test_exit_preserves_candidate_for_restart(self):
        self.dialogue.reply("Casper")
        self.assertIsNone(self.dialogue.reply("/exit"))
        self.assertIn("Casper", GhostDialogue(MemoryStore(self.db)).start())
        self.assertIsNone(self.memory.get_ghost_name())

    def test_recovers_after_name_saved_before_next_question(self):
        self.memory.set_ghost_name("Casper")
        self.assertIn("What should I call you", self.dialogue.start())
        self.memory.set_dialogue_state("confirming_user_name", "Ava")
        self.memory.set_user_name("Ava")
        self.assertIn("Back again, Ava", self.dialogue.start())
        self.assertIsNone(self.memory.get_dialogue_state())

    def test_forget_and_reset_reenter_correct_stage(self):
        self.finish_introduction()
        self.memory.forget_user()
        self.assertIn("What should I call you", self.dialogue.start())
        self.assertEqual(self.dialogue.speaker, "Casper")
        self.memory.reset_installation()
        self.assertIn("misplaced my name", self.dialogue.start())
        self.assertEqual(self.dialogue.speaker, "Ghost")

    def test_existing_names_and_unhandled_question_are_preserved(self):
        self.finish_introduction()
        self.memory.set_dialogue_state("choose_support_mode")
        self.assertIn("doesn't handle", self.dialogue.start())
        self.dialogue.reply("yes")
        self.assertEqual(self.memory.get_dialogue_state()["pending_question_id"], "choose_support_mode")

    def test_cli_process_restart_from_different_directory(self):
        script = Path(__file__).resolve().with_name("main.py")
        def run(inputs):
            return subprocess.run(
                [sys.executable, str(script), "--db", str(self.db)],
                input=inputs, text=True, capture_output=True, check=True,
                cwd=self.temp.name,
            ).stdout
        first = run("Casper\n/exit\n")
        self.assertIn("Shall I keep it?", first)
        second = run("yes\nRuwanya\nyes\n/exit\n")
        self.assertIn("Casper:", second)
        self.assertEqual(self.memory.get_user_name(), "Ruwanya")
        self.assertIn("Back again, Ruwanya", run(""))


if __name__ == "__main__":
    unittest.main()
