import asyncio
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
    REWARD_HIGH_POPULATION,
    REWARD_LOW_POPULATION,
    REWARD_MISSED_SPAWN,
    REWARD_SMALL_STEP,
    REWARD_SPAWN_SUCCESS,
    REWARD_TILE_LOST,
    REWARD_TILE_WON,
)
from lib.qtable import QTable
from lib.server_interface import ServerInterface

# Population thresholds
LOW_POPULATION_THRESHOLD = 0.20
HIGH_POPULATION_THRESHOLD = 0.80
ATTACK_POPULATION_THRESHOLD = 0.70


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

        old_me = (old_state or {}).get("me", {})
        new_me = (new_state or {}).get("me", {})
        old_owned = old_me.get("ownedCount", 0)
        new_owned = new_me.get("ownedCount", 0)

        tiles_diff = new_owned - old_owned
        if tiles_diff > 0:
            reward += tiles_diff * REWARD_TILE_WON
        elif tiles_diff < 0:
            reward += abs(tiles_diff) * REWARD_TILE_LOST

        # Population threshold rewards
        population = new_me.get("population", 0)
        max_population = new_me.get("maxPopulation", 1)

        if max_population > 0:
            pop_ratio = population / max_population
            if pop_ratio < LOW_POPULATION_THRESHOLD:
                reward += REWARD_LOW_POPULATION
            elif pop_ratio > HIGH_POPULATION_THRESHOLD:
                reward += REWARD_HIGH_POPULATION

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

        if isinstance(candidates, dict):
            empty = candidates.get("emptyNeighbors") or []
            enemy = candidates.get("enemyNeighbors") or []
        else:
            # New format: candidates is a list, wilderness cells have troops=0
            empty = [c for c in candidates if c.get("troops", 0) == 0]
            enemy = [c for c in candidates if c.get("troops", 0) > 0]

        if state.get("inSpawnPhase"):
            me = state.get("me") or {}
            has_spawned = me.get("ownedCount", 0) > 0
            if empty and not has_spawned:
                return [{"type": Action.SPAWN.value, "x": -1, "y": -1}]
            return [{"type": Action.NONE.value}]

        me = state.get("me") or {}
        population = me.get("population", 0)
        max_population = me.get("maxPopulation", 0)

        can_attack = False
        if max_population > 0:
            population_ratio = population / max_population
            can_attack = population_ratio > ATTACK_POPULATION_THRESHOLD

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
                    for cell in prioritized
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
        self.total_reward = 0
        self._state_lock = asyncio.Lock()
        self.reset()

    def reset(self):
        if self.score is not None:
            self.history.append(self.total_reward)
        self.iterations = 0
        self.score = 0
        self.total_reward = 0
        self.state = None

    def get_state(self):
        s = getattr(self.env, "current_state", {}) or {}
        me = s.get("me") or {}
        candidates = s.get("candidates") or []

        in_spawn = bool(s.get("inSpawnPhase", False))
        population = me.get("population", 0)
        max_population = me.get("maxPopulation", 1)
        conquest_pct = me.get("conquestPercent", 0)

        # Extract neighbor populations from candidates
        neighbor_populations = []
        if isinstance(candidates, dict):
            # Old format
            enemy_neighbors = candidates.get("enemyNeighbors") or []
            neighbor_populations = [n.get("troops", 0) for n in enemy_neighbors]
        else:
            # New format: candidates is a list
            neighbor_populations = [c.get("troops", 0) for c in candidates]

        return (
            in_spawn,
            population,
            max_population,
            conquest_pct,
            tuple(neighbor_populations),
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
        self.total_reward += self.reward
        self.iterations += 1

        # Display concise status
        state = self.env.current_state or {}
        tick = state.get("tick", 0)
        me = state.get("me", {})
        pop = me.get("population", 0)
        max_pop = me.get("maxPopulation", 1)
        conquest_pct = me.get("conquestPercent", 0)

        candidates = state.get("candidates") or []
        neighbor_count = len(candidates)

        status = f"\rTick: {tick:4d} | State: {new_state_key} | Pop: {pop:7d}/{max_pop:7d} | Conquest: {conquest_pct:2d}% | Neighbors: {neighbor_count:2d}"
        print(status, end="", flush=True)

        self.state = new_state_key

    async def best_action(self):
        self.state = self.get_state()
        possible_actions = self.env.get_possible_actions(self.env.current_state or {})

        if not possible_actions:
            return {"type": Action.NONE.value}

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        closest_state = await self.qtable.find_closest_state(self.state)
        if closest_state is None:
            return random.choice(possible_actions)

        action_keys = [self._get_action_key(a) for a in possible_actions]
        q_values = await self.qtable.get_state_actions(closest_state, action_keys)

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
        print("\nShutdown complete.")
