# ghost's naming convention only

import argparse
from pathlib import Path
import sqlite3

from dialogue import conversation_loop
from memory import DEFAULT_DB_PATH, MemoryStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ghost Terminal: naming and memory prototype.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help="Optional alternative database; defaults to data/ghost.db beside memory.py.")
    args = parser.parse_args()
    try:
        conversation_loop(MemoryStore(args.db))
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Could not continue: {error}")
        print("No reset was requested. Check the problem before restarting.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
