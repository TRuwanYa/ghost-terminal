from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterator
import unicodedata


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "ghost.db"


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("The name must be text.")
    name = name.strip()
    if not 1 <= len(name) <= 60:
        raise ValueError("Use a name containing 1 to 60 characters.")
    if any(unicodedata.category(char) == "Cc" for char in name):
        raise ValueError("Names cannot contain control characters or line breaks.")
    return name


class MemoryStore:
    """Store names and one pending dialogue state for a single local user."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_database()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize_database(self) -> None:
        """Create empty tables without replacing existing saved information."""
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported database version; keep the file for migration.")
            db.execute("""
                CREATE TABLE IF NOT EXISTS ghost_instance (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    chosen_name TEXT NOT NULL,
                    named_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    name TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS dialogue_state (
                    user_id INTEGER PRIMARY KEY CHECK (user_id = 1),
                    pending_question_id TEXT NOT NULL,
                    pending_value TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.execute("PRAGMA user_version = 1")

    def get_user_name(self) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT name FROM user_profile WHERE id = 1").fetchone()
            return row["name"] if row else None

    def set_user_name(self, name: str) -> None:
        name = _validate_name(name)
        with self._connection() as db:
            db.execute("""
                INSERT INTO user_profile (id, name) VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name, updated_at = CURRENT_TIMESTAMP
            """, (name,))
            db.execute("""
                DELETE FROM dialogue_state
                WHERE user_id = 1 AND pending_question_id = 'awaiting_user_name'
            """)

    def get_ghost_name(self) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT chosen_name FROM ghost_instance WHERE id = 1").fetchone()
            return row["chosen_name"] if row else None

    def set_ghost_name(self, name: str) -> None:
        """Commit an accepted name. Changing it requires an explicit reset."""
        name = _validate_name(name)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT chosen_name FROM ghost_instance WHERE id = 1").fetchone()
            if row and row["chosen_name"] != name:
                raise ValueError("Ghost already has a name. Use reset only if you want a fresh installation.")
            db.execute("INSERT OR IGNORE INTO ghost_instance (id, chosen_name) VALUES (1, ?)", (name,))
            db.execute("""
                DELETE FROM dialogue_state WHERE user_id = 1
                AND pending_question_id IN ('awaiting_ghost_name', 'confirming_ghost_name')
            """)

    def get_dialogue_state(self) -> dict | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM dialogue_state WHERE user_id = 1").fetchone()
            return dict(row) if row else None

    def set_dialogue_state(self, question_id: str, pending_value: str | None = None) -> None:
        """Save an identifier; dialogue.py will define its wording and answers."""
        if not isinstance(question_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", question_id):
            raise ValueError("Use a question ID of 1 to 64 lowercase letters, digits, or underscores, starting with a letter.")
        if pending_value is not None and (not isinstance(pending_value, str) or len(pending_value) > 500):
            raise ValueError("The pending value must be text of at most 500 characters, or None.")
        with self._connection() as db:
            db.execute("""
                INSERT INTO dialogue_state (user_id, pending_question_id, pending_value)
                VALUES (1, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    pending_question_id = excluded.pending_question_id,
                    pending_value = excluded.pending_value,
                    updated_at = CURRENT_TIMESTAMP
            """, (question_id, pending_value))

    def clear_dialogue_state(self) -> None:
        with self._connection() as db:
            db.execute("DELETE FROM dialogue_state")

    def forget_user(self) -> None:
        """Clear all currently stored user data and dialogue, keeping Ghost's name."""
        with self._connection() as db:
            db.execute("DELETE FROM user_profile")
            db.execute("DELETE FROM dialogue_state")

    def reset_installation(self) -> None:
        """Clear all application records, including Ghost's chosen name."""
        with self._connection() as db:
            db.execute("DELETE FROM user_profile")
            db.execute("DELETE FROM dialogue_state")
            db.execute("DELETE FROM ghost_instance")

    def snapshot(self) -> dict:
        return {
            "ghost_name": self.get_ghost_name(),
            "user_name": self.get_user_name(),
            "dialogue_state": self.get_dialogue_state(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and test Ghost's SQLite memory.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Optional alternative database path.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the database without clearing existing data.")
    commands.add_parser("show", help="Display saved records; null means nothing is saved.")
    commands.add_parser("set-user", help="Save or correct the user's name.").add_argument("name")
    commands.add_parser("set-ghost", help="Save Ghost's accepted name.").add_argument("name")
    pending = commands.add_parser("set-question", help="Save the pending question's identifier.")
    pending.add_argument("question_id")
    pending.add_argument("--value", default=None, help="Optional value awaiting confirmation.")
    commands.add_parser("clear-question", help="Clear the pending question.")
    commands.add_parser("forget-user", help="Delete user information and dialogue; keep Ghost's name.")
    commands.add_parser("reset", help="Delete all saved records, including Ghost's name.")
    args = parser.parse_args()
    try:
        memory = MemoryStore(args.db)
        if args.command == "show":
            print(json.dumps(memory.snapshot(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "set-user":
            memory.set_user_name(args.name)
        elif args.command == "set-ghost":
            memory.set_ghost_name(args.name)
        elif args.command == "set-question":
            memory.set_dialogue_state(args.question_id, args.value)
        elif args.command == "clear-question":
            memory.clear_dialogue_state()
        elif args.command == "forget-user":
            memory.forget_user()
        elif args.command == "reset":
            memory.reset_installation()
        print(f"Completed: {args.command}. Database: {memory.db_path}")
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
