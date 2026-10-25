from __future__ import annotations

import unicodedata

from memory import MemoryStore


YES = {"yes", "y", "yeah", "yep", "sure", "okay", "ok"}
NO = {"no", "n", "nope", "nah"}
GHOST_STATES = {"awaiting_ghost_name", "confirming_ghost_name"}
USER_STATES = {"awaiting_user_name", "confirming_user_name"}
HELP = (
    "For introductions, enter only the name, then answer yes or no.\n"
    "/help        Show these instructions\n"
    "/whoami      Show our saved names\n"
    "/name NAME   Correct your saved name\n"
    "/about       Explain what Ghost is\n"
    "/exit        Leave; unfinished introductions resume next time"
)


def _name_candidate(text: str) -> str:
    # Validate a name field, without claiming to extract names from prose.
    text = text.strip()
    if not 1 <= len(text) <= 60:
        raise ValueError("Please enter a name containing 1 to 60 characters.")
    allowed = all(
        char.isalpha() or unicodedata.category(char).startswith("M")
        or char in " '-.’" for char in text
    )
    normalized = text.casefold().strip(" .")
    if (
        not allowed or not any(char.isalpha() for char in text)
        or normalized in YES | NO | {"hello", "hi", "maybe", "skip", "exit", "quit"}
        or normalized.startswith(("i ", "my ", "call me ", "what ", "how ", "can ", "please "))
    ):
        raise ValueError("For now, type just the name, such as Casper or Ruwanya. Use /exit to leave.")
    return text


class GhostDialogue:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    @property
    def speaker(self) -> str:
        return self.memory.get_ghost_name() or "Ghost"

    def _current_state(self) -> dict | None:
        # Reconcile saved progress, including a restart between two writes.
        state = self.memory.get_dialogue_state()
        question = state["pending_question_id"] if state else None
        if self.memory.get_ghost_name() is None:
            if question not in GHOST_STATES or (
                question == "confirming_ghost_name" and not state["pending_value"]
            ):
                self.memory.set_dialogue_state("awaiting_ghost_name")
        elif self.memory.get_user_name() is None:
            if question not in USER_STATES or (
                question == "confirming_user_name" and not state["pending_value"]
            ):
                self.memory.set_dialogue_state("awaiting_user_name")
        elif question in GHOST_STATES | USER_STATES:
            self.memory.clear_dialogue_state()
        return self.memory.get_dialogue_state()

    def _prompt(self, state: dict) -> str:
        question = state["pending_question_id"]
        value = state["pending_value"]
        if question == "awaiting_ghost_name":
            return "You woke me up... and apparently I've misplaced my name. Any suggestions? (Name only.)"
        if question == "confirming_ghost_name":
            if value.casefold() == "casper":
                return "Casper... subtle. Shall I keep it? (yes/no)"
            return f"{value}... I could haunt a terminal under that name. Shall I keep it? (yes/no)"
        if question == "awaiting_user_name":
            return "Your turn. What should I call you? (Name only.)"
        if question == "confirming_user_name":
            return f"Shall I remember your name as {value}? (yes/no)"
        return "There's a saved question this introduction doesn't handle yet. Use /exit to leave; it will stay saved."

    def start(self) -> str:
        state = self._current_state()
        if state:
            return self._prompt(state)
        return (
            f"Back again, {self.memory.get_user_name()}? Good. "
            f"I'm still {self.speaker}... my memory appears to be behaving."
        )

    def reply(self, text: str) -> str | None:
        # Return a reply, or None when the user explicitly exits.
        text = text.strip()
        command, _, argument = text.partition(" ")
        command = command.casefold()
        if command == "/exit" and not argument:
            return None
        if command == "/help":
            return HELP
        if command == "/about":
            return "I'm a fictional ghost chatbot. The haunting is theatrical; the saved names are real application data."
        if command == "/whoami":
            return (
                f"Your saved name: {self.memory.get_user_name() or '(not set)'}. "
                f"My saved name: {self.memory.get_ghost_name() or '(not chosen)'}."
            )
        if command == "/name":
            try:
                name = _name_candidate(argument)
            except ValueError as error:
                return str(error)
            self.memory.set_user_name(name)
            state = self._current_state()
            suffix = "\n" + self._prompt(state) if state else ""
            return f"{name}. Corrected... even a ghost can update his notes." + suffix
        if text.startswith("/"):
            return "I don't recognize that command. Type /help for the available commands."

        state = self._current_state()
        if not state:
            return (
                "Introductions are working, but general conversation isn't built yet. "
                "Use /whoami, /name NAME, or /exit."
            )
        question = state["pending_question_id"]
        if not text:
            return self._prompt(state)

        if question in {"awaiting_ghost_name", "awaiting_user_name"}:
            try:
                name = _name_candidate(text)
            except ValueError as error:
                return str(error)
            next_question = (
                "confirming_ghost_name" if question == "awaiting_ghost_name"
                else "confirming_user_name"
            )
            self.memory.set_dialogue_state(next_question, name)
            return self._prompt(self.memory.get_dialogue_state())

        if question in {"confirming_ghost_name", "confirming_user_name"}:
            answer = text.casefold().rstrip(".!?").strip()
            if answer not in YES | NO:
                return "A yes or no will do for this question. " + self._prompt(state)
            if answer in NO:
                target = "awaiting_ghost_name" if question == "confirming_ghost_name" else "awaiting_user_name"
                self.memory.set_dialogue_state(target)
                return "Fair enough. " + self._prompt(self.memory.get_dialogue_state())
            name = state["pending_value"]
            if question == "confirming_ghost_name":
                self.memory.set_ghost_name(name)
                remaining = self._current_state()
                return "Settled, then. " + (self._prompt(remaining) if remaining else self.start())
            self.memory.set_user_name(name)
            self.memory.clear_dialogue_state()
            return f"{name}. I'll try to remember that better than my own name. Type /help whenever you need it."

        return self._prompt(state)


def conversation_loop(memory: MemoryStore | None = None) -> None:
    dialogue = GhostDialogue(memory if memory is not None else MemoryStore())
    print("Ghost Terminal | Naming prototype | /help for commands | /exit to leave")
    print(f"{dialogue.speaker}: {dialogue.start()}")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed. Saved progress will be available next time.")
            return
        answer = dialogue.reply(text)
        if answer is None:
            print(f"{dialogue.speaker}: Off you go... I'll keep our place.")
            return
        print(f"{dialogue.speaker}: {answer}")
