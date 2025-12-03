from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@lru_cache
def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", choices=("bot", "interface"))
    parser.add_argument("--server-ws")
    parser.add_argument("--interface-port", type=int)
    parser.add_argument("--qtable-file")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.add_argument("--print-interval", type=int)
    parser.add_argument("--autosave-interval", type=int)
    parser.add_argument("--graph", dest="graph", action="store_true")
    parser.add_argument("--no-graph", dest="graph", action="store_false")
    parser.set_defaults(debug=None, graph=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def resolve_setting(
    env_name: str,
    cli_value: Any,
    default: Any,
    caster: Any = str,
) -> Any:
    if cli_value is not None:
        return cli_value
    raw = os.getenv(env_name)
    if raw is not None:
        return caster(raw)
    return default


args = cli_args()

MODE = resolve_setting("MODE", args.mode, default="bot")
if MODE not in ("bot", "interface"):
    print(f"Unknown MODE '{MODE}', defaulting to 'bot'.")
    MODE = "bot"

DEBUG_MODE = resolve_setting(
    "DEBUG_MODE",
    args.debug,
    default=False,
    caster=as_bool,
)
SERVER_WS = resolve_setting(
    "SERVER_WS",
    args.server_ws,
    default="ws://localhost:3000/bot",
)
INTERFACE_PORT = resolve_setting(
    "INTERFACE_PORT",
    args.interface_port,
    default=8765,
    caster=int,
)
QTABLE_FILE = resolve_setting("QTABLE_FILE", args.qtable_file, default="qtable.pkl")
PRINT_INTERVAL = resolve_setting(
    "PRINT_INTERVAL",
    args.print_interval,
    default=20,
    caster=int,
)
AUTOSAVE_INTERVAL = resolve_setting(
    "AUTOSAVE_INTERVAL",
    args.autosave_interval,
    default=300,
    caster=int,
)
GRAPH_ENABLED = resolve_setting(
    "GRAPH_ENABLED",
    args.graph,
    default=False,
    caster=as_bool,
)

ATTACK_RATIOS = [0.30, 0.35, 0.40]
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995

CONQUEST_WIN_THRESHOLD = 80

REWARD_SPAWN_SUCCESS = 100.0
REWARD_SPAWN_FAIL = -1000.0
REWARD_MISSED_SPAWN = -150.0
REWARD_SMALL_STEP = -0.1
REWARD_TILE_WON = 10.0
REWARD_TILE_LOST = -50.0
REWARD_VERY_LOW_POPULATION = -100.0
REWARD_VERY_HIGH_POPULATION = -100.0
REWARD_CONQUEST_WIN = 500.0

# Q-Table state matching configuration
DISTANCE_THRESHOLD = 10  # Maximum distance for state matching (returns 0.0 if exceeded)

# Environment thresholds
LOW_POPULATION_THRESHOLD = 0.20
HIGH_POPULATION_THRESHOLD = 0.80
SPAWN_PHASE_DURATION = 100
MAX_NEIGHBORS_DISPLAY = 5
