import asyncio
import math
import random
import sys
from typing import Any

from websockets.asyncio.server import serve

from lib.bot_connection import BotConnection
from lib.constants import (
    ATTACK_RATIOS,
    DEBUG_MODE,
    INTERFACE_PORT,
    MODE,
    PRUNE_MAX_TARGETS,
    REWARD_MISSED_SPAWN,
    REWARD_SMALL_STEP,
    REWARD_SPAWN_SUCCESS,
)
from lib.qtable import QTable
from lib.server_interface import ServerInterface
from lib.utils import Action, get_action_key, normalize_number


class Environment:
    def __init__(self):
        self.current_state: dict[str, Any] | None = None
        self.previous_state: dict[str, Any] | None = None

    def update_state(self, state: dict[str, Any]) -> None:
        self.previous_state = self.current_state
        self.current_state = state

    def calculate_reward(
        self,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        action: dict[str, Any] | None,
    ) -> float:
        reward = REWARD_SMALL_STEP

        print(f"State: {self.current_state}") # @ TODO: Keep the state minimal
        print(f"Previous State: {self.previous_state}")

        if (
            action
            and action.get("type") == Action.SPAWN.value
        ):
            reward += REWARD_SPAWN_SUCCESS
            print(f"Reward: spawn success detected -> +{REWARD_SPAWN_SUCCESS}")

        prev_in_spawn = bool((old_state or {}).get("inSpawnPhase", False))
        prev_candidates = ((old_state or {}).get("candidates") or {}).get(
            "emptyNeighbors"
        ) or []
        if (
            prev_in_spawn
            and len(prev_candidates) > 0
            and (not action or action.get("type") != Action.SPAWN.value)
        ):
            reward += REWARD_MISSED_SPAWN

        return reward

    def get_possible_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = state.get("candidates") or {}
        empty = candidates.get("emptyNeighbors") or []
        enemy = candidates.get("enemyNeighbors") or []

        if state.get("inSpawnPhase"):
            if empty:
                return [
                    {"type": Action.SPAWN.value, "x": cell["x"], "y": cell["y"]}
                    for cell in empty[:PRUNE_MAX_TARGETS]
                ]
            return [{"type": Action.NONE.value}]

        targets = (enemy or []) + (empty or [])
        if not targets:
            return [{"type": Action.NONE.value}]

        actions = [{"type": Action.NONE.value}]
        prioritized = (enemy or []) + (empty or [])
        actions.extend(
            {
                "type": Action.ATTACK.value,
                "x": cell["x"],
                "y": cell["y"],
                "ratio": ratio,
            }
            for cell in prioritized[:PRUNE_MAX_TARGETS]
            for ratio in ATTACK_RATIOS
        )

        return actions


class Agent:
    def __init__(self, env):
        self.env = env
        self.qtable = QTable.get_instance()
        self.score = None
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.previous_state = None
        self.previous_action = None
        self.history = []
        self._state_lock = asyncio.Lock()
        self.reset()

    def reset(self):
        if self.score is not None:
            self.history.append(self.score)
        self.score = 0

    def get_state(self):
        s = getattr(self.env, "current_state", {}) or {}
        me = s.get("me") or {}
        candidates = s.get("candidates") or {}

        in_spawn = bool(s.get("inSpawnPhase", False))
        population = math.floor(normalize_number(me.get("population")) or 0)
        max_population = math.floor(normalize_number(me.get("maxPopulation")) or 0)
        troops = math.floor(normalize_number(me.get("troops")) or 0)
        conquest = math.floor(normalize_number(me.get("conquestPercent")) or 0)
        owned_count = math.floor(normalize_number(me.get("ownedCount")) or 0)
        empty_count = len(candidates.get("emptyNeighbors") or [])
        enemy_count = len(candidates.get("enemyNeighbors") or [])

        state = (
            in_spawn,
            population,
            max_population,
            troops,
            conquest,
            owned_count,
            empty_count,
            enemy_count,
        )

        if DEBUG_MODE:
            print(f"Current state: {state}")

        return state

    async def get_previous_state_action_if_ready(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        async with self._state_lock:
            if self.previous_state is not None and self.previous_action is not None:
                return (self.previous_state, self.previous_action.copy())
        return None

    async def set_previous_state_action(
        self, state: dict[str, Any], action: dict[str, Any]
    ) -> None:
        async with self._state_lock:
            self.previous_state = state
            self.previous_action = action

    async def get_q_value(self, state_key: Any, action_key: str) -> float:
        return await self.qtable.get_q_value(state_key, action_key)

    async def best_action(
        self, _state: dict[str, Any], possible_actions: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not possible_actions:
            return None

        async with self._state_lock:
            state_key = self.get_state()
        action_keys = [get_action_key(action) for action in possible_actions]
        await self.qtable.ensure_state_exists(state_key, action_keys)

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        possible_actions_keys = {
            get_action_key(action): action for action in possible_actions
        }
        state_actions = await self.qtable.get_state_actions(
            state_key, list(possible_actions_keys.keys())
        )
        valid_q_actions = list(state_actions.items())

        if not valid_q_actions:
            return random.choice(possible_actions)

        best_action_key = max(valid_q_actions, key=lambda item: item[1])[0]
        return possible_actions_keys.get(
            best_action_key, random.choice(possible_actions)
        )

    async def do(self, action: dict[str, Any], state: dict[str, Any]) -> None:
        if self.previous_state is not None and self.previous_action is not None:
            reward = self.env.calculate_reward(
                self.previous_state, state, self.previous_action
            )
            await self.update(state, reward)
            self.score += reward
        await self.set_previous_state_action(state, action)

    async def update(self, _state: dict[str, Any], reward: float) -> None:
        async with self._state_lock:
            if self.previous_state is None or self.previous_action is None:
                return

            env_saved = getattr(self.env, "current_state", None)
            try:
                self.env.current_state = self.previous_state
                prev_state_key = self.get_state()
            finally:
                self.env.current_state = env_saved

            prev_action_key = get_action_key(self.previous_action)
            current_state_key = self.get_state()

        current_q = await self.get_q_value(prev_state_key, prev_action_key)
        max_next_q = await self.qtable.get_max_q_value(current_state_key)
        delta = self.alpha * (reward + self.gamma * max_next_q - current_q)
        new_q_value = current_q + delta
        await self.qtable.set_q_value(prev_state_key, prev_action_key, new_q_value)

    async def save(self) -> None:
        await self.qtable.save()

    async def load(self) -> None:
        await self.qtable.load()


async def run_main() -> None:
    env = Environment()
    agent = Agent(env)
    await agent.load()

    if MODE == "bot":
        bot = BotConnection(agent, env)
        await bot.run()
    elif MODE == "interface":
        server_iface = ServerInterface(agent, env)
        print(f"Starting interface websocket server on 0.0.0.0:{INTERFACE_PORT}")
        async with serve(server_iface.handle_connection, "0.0.0.0", INTERFACE_PORT):
            await asyncio.Future()
    else:
        print(f"Unknown MODE '{MODE}', exiting.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        print("Shutting down, saving qtable...")

        async def save_on_exit():
            a = Agent(Environment())
            await a.load()
            await a.save()

        asyncio.run(save_on_exit())
