import asyncio
import json
import os
import pickle
import random
import string
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from websockets.asyncio.server import ServerConnection, serve
import websockets

from lib.bot_connection import BotConnection
from lib.server_interface import ServerInterface


MODE = "bot"


SERVER_WS = "ws://localhost:3000/bot"
INTERFACE_PORT = 8765
QTABLE_FILE = "qtable.pkl"


PRUNE_MAX_TARGETS = 50
ATTACK_RATIOS = [0.20, 0.40, 0.60, 0.80]
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995


AUTOSAVE_INTERVAL = 300
PRINT_INTERVAL = 20


REWARD_TERRITORY_GAIN = 10.0
REWARD_TERRITORY_LOSS = -10.0
REWARD_SPAWN_SUCCESS = 50.0
REWARD_MISSED_SPAWN = -50.0
REWARD_SMALL_STEP = -0.1
REWARD_WIN = 100.0
REWARD_LOST = -100.0


class Action(Enum):
    SPAWN = "spawn"
    ATTACK = "attack"
    NONE = "none"


def make_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def safe_num(v: Optional[float], fmt: str = "{:.2f}") -> str:
    if v is None:
        return "N/A"
    try:
        return fmt.format(v)
    except Exception:
        try:
            return str(float(v))
        except Exception:
            return str(v)


def arg_max(table: Dict[str, float]) -> str:
    return max(table, key=table.get)


def get_action_key(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    if action_type == "spawn":
        return f"spawn:{action.get('x')},{action.get('y')}"
    elif action_type == "attack":
        return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
    return "none"


class Environment:
    def __init__(self):
        self.current_state: Optional[Dict[str, Any]] = None
        self.previous_state: Optional[Dict[str, Any]] = None

    def update_state(self, state: Dict[str, Any]) -> None:
        try:
            self.previous_state = self.current_state
            self.current_state = state
        except Exception:
            self.previous_state = None
            self.current_state = state

    def do(self, action: Dict[str, Any]) -> float:
        if self.previous_state is None or self.current_state is None:
            return 0.0
        return self.calculate_reward(self.previous_state, self.current_state, action)

    def calculate_reward(
        self,  
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        action: Optional[Dict[str, Any]],
    ) -> float:
        reward = REWARD_SMALL_STEP

        try:
            prev_small = old_state.get("me", {}).get("smallID")
            new_small = new_state.get("me", {}).get("smallID")
            if (
                action
                and action.get("type") == "spawn"
                and (not prev_small)
                and new_small
            ):
                reward += REWARD_SPAWN_SUCCESS
                print(f"Reward: spawn success detected -> +{REWARD_SPAWN_SUCCESS}")
        except Exception:
            pass

        try:
            prev_in_spawn = bool(old_state.get("inSpawnPhase", False))
            prev_candidates = (old_state.get("candidates") or {}).get(
                "emptyNeighbors"
            ) or []
            if prev_in_spawn and len(prev_candidates) > 0:
                if not action or action.get("type") != "spawn":
                    reward += REWARD_MISSED_SPAWN
                    
        except Exception:
            pass
        
        return reward

    def get_possible_actions(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates = state.get("candidates") or {}
        empty = candidates.get("emptyNeighbors") or []
        enemy = candidates.get("enemyNeighbors") or []

        if state.get("inSpawnPhase"):
            if empty:
                actions: List[Dict[str, Any]] = []
                for cell in empty[:PRUNE_MAX_TARGETS]:
                    actions.append({"type": "spawn", "x": cell["x"], "y": cell["y"]})
                return actions
            else:
                return [{"type": "none"}]

        targets = (enemy or []) + (empty or [])
        if not targets:
            return [{"type": "none"}]

        actions = [{"type": "none"}]
        prioritized = (enemy or []) + (empty or [])
        for cell in prioritized[:PRUNE_MAX_TARGETS]:
            for ratio in ATTACK_RATIOS:
                actions.append(
                    {"type": "attack", "x": cell["x"], "y": cell["y"], "ratio": ratio}
                )

        return actions


class Agent:
    def __init__(self,env):
        self.env = env
        self.reward = 0
        self.qtable = {}
        self.score = None
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2

        self.previous_state = None
        self.previous_action = None

        self.reset()
        self.history = []

    def reset(self):
        if self.score is not None:
            self.history.append(self.score)
        self.iterations = 0
        self.score = 0
        self.pos = getattr(self.env, "start", None)
        self.state = self.get_state()

    def get_state(self):
        """Return a compact, safe tuple representing the bot's observable state.

        Keeps the signature unchanged (no parameters). The method reads the
        current state from `self.env.current_state` and returns a tuple:
        (in_spawn, population, maxPopulation, troops, rank, conquestPercent, ownedCount, empty_count, enemy_count)

        Missing or malformed values are represented as the string "N/A".
        """
        try:
            s = getattr(self.env, "current_state", {}) or {}
        except Exception:
            s = {}

        me = s.get("me") or {}
        candidates = s.get("candidates") or {}

        def safe_num(v):
            """Return numeric value when possible, otherwise None.

            - ints are preserved
            - floats parsed from strings are returned as float
            - integers represented as floats are converted to int
            - non-numeric or missing values -> None
            """
            if v is None:
                return None
            try:
                if isinstance(v, int):
                    return v
                f = float(v)
                if abs(f - int(f)) < 1e-9:
                    return int(f)
                return f
            except Exception:
                return None

        in_spawn = bool(s.get("inSpawnPhase", False))
        population = safe_num(me.get("population"))
        max_population = safe_num(me.get("maxPopulation"))
        troops = safe_num(me.get("troops"))
        rank = safe_num(me.get("rank"))
        conquest = safe_num(me.get("conquestPercent"))
        owned_count = safe_num(me.get("ownedCount"))

        empty_count = len((candidates.get("emptyNeighbors") or []))
        enemy_count = len((candidates.get("enemyNeighbors") or []))

        state = (
            in_spawn,
            population,
            max_population,
            troops,
            rank,
            conquest,
            owned_count,
            empty_count,
            enemy_count,
        )

        print(f"Current state: {state}")

        return state


    def get_q_value(self, state_key: str, action_key: str) -> float:
        if state_key not in self.qtable:
            return 0.0
        return self.qtable[state_key].get(action_key, 0.0)
    

    def best_action(
        self, state: Dict[str, Any], possible_actions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not possible_actions:
            return None

        state_key = self.get_state()

        if state_key not in self.qtable:
            self.qtable[state_key] = {}
            for action in possible_actions:
                action_key = get_action_key(action)
                self.qtable[state_key][action_key] = 0.0
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        possible_actions_keys = {
            get_action_key(action): action for action in possible_actions
        }
        for action_key in possible_actions_keys:
            if action_key not in self.qtable[state_key]:
                self.qtable[state_key][action_key] = 0.0

        valid_q_actions = [
            (action_key, q_value)
            for action_key, q_value in self.qtable[state_key].items()
            if action_key in possible_actions_keys
        ]

        if not valid_q_actions:
            return random.choice(possible_actions)

        best_action_key = max(valid_q_actions, key=lambda item: item[1])[0]
        return possible_actions_keys.get(
            best_action_key, random.choice(possible_actions)
        )

    def do(self, action: Dict[str, Any], state: Dict[str, Any]) -> None:
        if self.previous_state is not None and self.previous_action is not None:
            reward = self.env.calculate_reward(
                self.previous_state, state, self.previous_action
            )
            self.update(state, reward)
            self.score += reward
        self.previous_state = state
        self.previous_action = action

    def update(self, state: Dict[str, Any], reward: float) -> None:
        if self.previous_state is None or self.previous_action is None:
            return

        # Build the previous-state key without changing get_state's signature:
        # temporarily set env.current_state to previous_state so get_state() reads it.
        env_saved = getattr(self.env, "current_state", None)
        try:
            self.env.current_state = self.previous_state
            prev_state_key = self.get_state()
        finally:
            # restore original env state
            self.env.current_state = env_saved
        prev_action_key = get_action_key(self.previous_action)
        current_state_key = self.get_state()
        current_q = self.get_q_value(prev_state_key, prev_action_key)
        max_next_q = 0.0
        if current_state_key in self.qtable and self.qtable[current_state_key]:
            max_next_q = max(self.qtable[current_state_key].values())
        delta = self.alpha * (reward + self.gamma * max_next_q - current_q)
        if prev_state_key not in self.qtable:
            self.qtable[prev_state_key] = {}
        self.qtable[prev_state_key][prev_action_key] = current_q + delta

    def save(self, filename: str = QTABLE_FILE) -> None:
        try:
            with open(filename, "wb") as f:
                pickle.dump((self.qtable, self.history), f)
            print(f"Q-table saved to {filename} ({len(self.qtable)} states)")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load(self, filename: str = QTABLE_FILE) -> None:
        if not Path(filename).exists():
            print(f"No saved Q-table found at {filename}, starting fresh")
            return
        try:
            with open(filename, "rb") as f:
                self.qtable, self.history = pickle.load(f)
            print(f"Q-table loaded from {filename} ({len(self.qtable)} states)")
        except Exception as e:
            print(f"Error loading Q-table: {e}, starting fresh")
            self.qtable = {}
            self.history = []


async def run_main() -> None:
    env = Environment()
    agent = Agent(env)

    try:
        agent.load(QTABLE_FILE)
    except Exception:
        pass

    if MODE == "bot":
        bot = BotConnection(agent, env)
        try:
            await bot.run()
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            print("Interrupted, saving qtable...")
            agent.save()
    else:
        server_iface = ServerInterface(agent, env)
        print(f"Starting interface websocket server on 0.0.0.0:{INTERFACE_PORT}")
        async with serve(server_iface.handle_connection, "0.0.0.0", INTERFACE_PORT):
            try:
                await asyncio.Future()
            except KeyboardInterrupt:
                print("Interface interrupted, saving qtable...")
                agent.save()


if __name__ == "__main__":
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("Shutting down, saving qtable...")

        try:
            a = Agent(Environment())
            a.load(QTABLE_FILE)
            a.save(QTABLE_FILE)
        except Exception:
            pass
