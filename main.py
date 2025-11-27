import asyncio
import math
import random
import sys
from enum import Enum
from typing import Any

from websockets.asyncio.server import serve

from lib.bot_connection import BotConnection
from lib.constants import (
    ATTACK_RATIOS,
    INTERFACE_PORT,
    MODE,
    PRUNE_MAX_TARGETS,
    REWARD_MISSED_SPAWN,
    REWARD_SMALL_STEP,
    REWARD_SPAWN_SUCCESS,
)
from lib.qtable import QTable
from lib.server_interface import ServerInterface
from lib.utils import normalize_number


class Action(Enum):
    SPAWN = "spawn"
    ATTACK = "attack"
    NONE = "none"


def arg_max(table):
    return max(table, key=table.get)


class Environment:
    def __init__(self):
        self.current_state: dict[str, Any] | None = None
        self.previous_state: dict[str, Any] | None = None
        self._state_event = asyncio.Event()
        self._action_queue = asyncio.Queue()

    def update_state(self, state: dict[str, Any]) -> None:
        self.previous_state = self.current_state
        self.current_state = state
        self._state_event.set()

    async def do(self, _pos, action):
        await self._action_queue.put(action)

        self._state_event.clear()
        await self._state_event.wait()

        reward = self.calculate_reward(
            self.previous_state or {}, self.current_state or {}, action
        )

        return self.current_state, reward

    def calculate_reward(
        self,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        action: dict[str, Any] | None,
    ) -> float:
        reward = REWARD_SMALL_STEP

        # Create a concise status line that updates on the same line
        state = self.current_state or {}
        tick = state.get("tick", 0)
        in_spawn = state.get("inSpawnPhase", False)
        me = state.get("me", {})
        pop = me.get("population", 0)
        owned = me.get("ownedCount", 0)
        action_type = action.get("type", "none") if action else "none"
        
        status = f"\rTick: {tick:4d} | Phase: {'SPAWN' if in_spawn else 'BATTLE'} | Pop: {pop:4d} | Owned: {owned:3d} | Action: {action_type:6s} | Reward: {reward:+.2f}"
        print(status, end='', flush=True)

        if action and action.get("type") == Action.SPAWN.value:
            reward += REWARD_SPAWN_SUCCESS

        prev_in_spawn = bool((old_state or {}).get("inSpawnPhase", False))
        prev_candidates = (old_state or {}).get("candidates") or []
        
        prev_empty = [c for c in prev_candidates if c.get("troops", 0) == 0]
        
        if (
            prev_in_spawn
            and len(prev_empty) > 0
            and (not action or action.get("type") != Action.SPAWN.value)
        ):
            reward += REWARD_MISSED_SPAWN

        return reward

    def get_possible_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = state.get("candidates") or []
        
        # Handle both old format (dict with emptyNeighbors/enemyNeighbors) and new format (list)
        if isinstance(candidates, dict):
            empty = candidates.get("emptyNeighbors") or []
            enemy = candidates.get("enemyNeighbors") or []
        else:
            # New format: candidates is a list, wilderness cells have troops=0
            empty = [c for c in candidates if c.get("troops", 0) == 0]
            enemy = [c for c in candidates if c.get("troops", 0) > 0]

        if state.get("inSpawnPhase"):
            if empty:
                return [
                    {"type": Action.SPAWN.value, "x": cell["x"], "y": cell["y"]}
                    for cell in empty[:PRUNE_MAX_TARGETS]
                ]
            return [{"type": Action.NONE.value}]

        # Check if we can attack: population must be > 70% of max population
        me = state.get("me") or {}
        population = me.get("population", 0)
        max_population = me.get("maxPopulation", 0)
        
        can_attack = False
        if max_population > 0:
            population_ratio = population / max_population
            can_attack = population_ratio > 0.70
        
        targets = (enemy or []) + (empty or [])
        if not targets:
            return [{"type": Action.NONE.value}]

        actions = [{"type": Action.NONE.value}]
        
        # Only add attack actions if we meet the population threshold
        if can_attack:
            prioritized = (enemy or []) + (empty or [])
            # Use attack ratios between 30% and 40% to keep population between 40-70%
            for ratio in ATTACK_RATIOS:
                actions.extend(
                    {
                        "type": Action.ATTACK.value,
                        "x": cell["x"],
                        "y": cell["y"],
                        "ratio": ratio,
                    }
                    for cell in prioritized[:PRUNE_MAX_TARGETS]
                )

        return actions



class Agent:
    def __init__(self, env):
        self.env = env
        self.reward = 0
        self.qtable = QTable.get_instance()
        self.score = None
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.history = []
        self._state_lock = asyncio.Lock()
        self.reset()

    def reset(self):
        if self.score is not None:
            self.history.append(self.score)
        self.iterations = 0
        self.score = 0
        self.state = None

    def get_state(self):
        s = getattr(self.env, "current_state", {}) or {}
        me = s.get("me") or {}
        candidates = s.get("candidates") or []

        in_spawn = bool(s.get("inSpawnPhase", False))
        population = math.floor(normalize_number(me.get("population")) or 0)
        pop_bin = int(math.log2(population + 1))

        # Handle both old and new format
        if isinstance(candidates, dict):
            empty_count = len(candidates.get("emptyNeighbors") or [])
            enemy_count = len(candidates.get("enemyNeighbors") or [])
        else:
            empty_count = len([c for c in candidates if c.get("troops", 0) == 0])
            enemy_count = len([c for c in candidates if c.get("troops", 0) > 0])

        return (
            in_spawn,
            pop_bin,
            empty_count,
            enemy_count,
        )

    async def do(self, action):
        previous_state = self.state

        self.state, self.reward = await self.env.do(self.state, action)

        new_state_key = self.get_state()
        prev_state_key = previous_state

        action_key = self._get_action_key(action)

        current_q = await self.qtable.get_q_value(prev_state_key, action_key)
        max_next_q = await self.qtable.get_max_q_value(new_state_key)

        delta = self.alpha * (self.reward + self.gamma * max_next_q - current_q)
        new_q = current_q + delta

        await self.qtable.set_q_value(prev_state_key, action_key, new_q)

        self.score += self.reward
        self.iterations += 1

        self.state = new_state_key

    async def best_action(self):
        self.state = self.get_state()
        possible_actions = self.env.get_possible_actions(self.env.current_state or {})

        if not possible_actions:
            return {"type": Action.NONE.value}

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        action_keys = [self._get_action_key(a) for a in possible_actions]
        q_values = await self.qtable.get_state_actions(self.state, action_keys)

        if not q_values:
            return random.choice(possible_actions)

        best_action_key = max(q_values, key=q_values.get)

        for a in possible_actions:
            if self._get_action_key(a) == best_action_key:
                return a

        return random.choice(possible_actions)

    def _get_action_key(self, action: dict[str, Any]) -> str:
        action_type = action.get("type")
        if action_type == Action.SPAWN.value:
            return f"spawn:{action.get('x')},{action.get('y')}"
        if action_type == Action.ATTACK.value:
            return f"attack:{action.get('x')},{action.get('y')}|ratio:{action.get('ratio')}"
        return Action.NONE.value

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
