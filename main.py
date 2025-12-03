import asyncio
import math
import random
import sys
from typing import Any

from websockets.asyncio.server import serve

from lib.bot_connection import BotConnection
from lib.constants import (
    ATTACK_RATIOS,
    CONQUEST_WIN_THRESHOLD,
    HIGH_POPULATION_THRESHOLD,
    INTERFACE_PORT,
    LOW_POPULATION_THRESHOLD,
    MAX_NEIGHBORS_DISPLAY,
    MODE,
    REWARD_CONQUEST_WIN,
    REWARD_MISSED_SPAWN,
    REWARD_SMALL_STEP,
    REWARD_SPAWN_SUCCESS,
    REWARD_TILE_LOST,
    REWARD_TILE_WON,
    REWARD_VERY_HIGH_POPULATION,
    REWARD_VERY_LOW_POPULATION,
    SPAWN_PHASE_DURATION,
)
from lib.qtable import QTable
from lib.server_interface import ServerInterface
from lib.utils import Action


def calculate_neighbor_ratio(my_troops: int, enemy_troops: int) -> int:
    if enemy_troops == 0:
        return 100
    if my_troops == 0:
        return -100

    ratio = my_troops / enemy_troops
    log_ratio = math.log10(ratio)
    clamped = max(-1.0, min(1.0, log_ratio))  # Clamp to [-1, 1]

    return int(clamped * 100)


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

    async def do(self, action):
        await self._action_queue.put(action)

        self._state_event.clear()
        await self._state_event.wait()

        reward = self.calculate_reward(
            self.previous_state or {}, self.current_state or {}, action
        )

        return self.current_state, reward

    def rewards_territory(
        self, old_state: dict[str, Any], new_state: dict[str, Any]
    ) -> float:
        old_me = (old_state or {}).get("me", {})
        new_me = (new_state or {}).get("me", {})
        old_owned = old_me.get("ownedCount", 0)
        new_owned = new_me.get("ownedCount", 0)
        tiles_diff = new_owned - old_owned

        if tiles_diff > 0:
            return tiles_diff * REWARD_TILE_WON
        if tiles_diff < 0:
            return abs(tiles_diff) * REWARD_TILE_LOST
        return 0.0

    def rewards_population(self, new_state: dict[str, Any]) -> float:
        new_me = (new_state or {}).get("me", {})
        population = new_me.get("population", 0)
        max_population = new_me.get("maxPopulation", 1)

        if max_population > 0:
            pop_ratio = population / max_population
            if pop_ratio < LOW_POPULATION_THRESHOLD:
                return REWARD_VERY_LOW_POPULATION
            if pop_ratio > HIGH_POPULATION_THRESHOLD:
                return REWARD_VERY_HIGH_POPULATION
        return 0.0

    def rewards_conquest(self, new_state: dict[str, Any]) -> float:
        new_me = (new_state or {}).get("me", {})
        conquest_pct = new_me.get("conquestPercent", 0)
        if conquest_pct >= CONQUEST_WIN_THRESHOLD:
            return REWARD_CONQUEST_WIN
        return 0.0

    def rewards_spawn(
        self,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        action: dict[str, Any] | None,
    ) -> float:
        reward = 0.0

        if action and action.get("type") == Action.SPAWN.value:
            reward += REWARD_SPAWN_SUCCESS

        prev_in_spawn = bool((old_state or {}).get("inSpawnPhase", False))
        prev_candidates = (old_state or {}).get("candidates") or []
        prev_empty = [c for c in prev_candidates if c.get("troops", 0) == 0]

        new_me = (new_state or {}).get("me", {})
        has_population = new_me.get("population", 0) > 0

        if prev_in_spawn and len(prev_empty) > 0 and not has_population:
            reward += REWARD_MISSED_SPAWN

        return reward

    def calculate_reward(
        self,
        old_state: dict[str, Any],
        new_state: dict[str, Any],
        action: dict[str, Any] | None,
    ) -> float:
        tick = new_state.get("tick", 0)

        if tick < SPAWN_PHASE_DURATION:
            return self.rewards_spawn(old_state, new_state, action)

        return (
            REWARD_SMALL_STEP
            + self.rewards_territory(old_state, new_state)
            + self.rewards_population(new_state)
            + self.rewards_conquest(new_state)
        )

    def get_possible_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = state.get("candidates") or []

        if isinstance(candidates, dict):
            empty = candidates.get("emptyNeighbors") or []
            enemy = candidates.get("enemyNeighbors") or []
        else:
            empty = [c for c in candidates if c.get("troops", 0) == 0]
            enemy = [c for c in candidates if c.get("troops", 0) > 0]

        if state.get("inSpawnPhase"):
            me = state.get("me") or {}
            has_spawned = me.get("ownedCount", 0) > 0
            if empty and not has_spawned:
                return [{"type": Action.SPAWN.value, "x": -1, "y": -1}]
            return [{"type": Action.NONE.value}]

        targets = (enemy or []) + (empty or [])
        if not targets:
            return [{"type": Action.NONE.value}]

        actions = [{"type": Action.NONE.value}]
        prioritized = (enemy or []) + (empty or [])
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
        self.gamma = 0.99
        self.epsilon = 0.05
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
        self.random_actions = 0
        self.qtable_actions = 0
        self.wait_actions = 0
        self.attack_actions = 0

    def get_state(self):
        s = getattr(self.env, "current_state", {}) or {}
        me = s.get("me") or {}
        candidates = s.get("candidates") or []

        in_spawn = bool(s.get("inSpawnPhase", False))
        population = me.get("population", 0)
        max_population = me.get("maxPopulation", 1)
        conquest_pct = me.get("conquestPercent", 0)

        if max_population > 0:
            population_pct = int((population / max_population) * 100)
        else:
            population_pct = 0

        neighbor_ratios = []
        if isinstance(candidates, dict):
            enemy_neighbors = candidates.get("enemyNeighbors") or []
            for neighbor in enemy_neighbors:
                enemy_troops = neighbor.get("troops", 0)
                ratio = calculate_neighbor_ratio(population, enemy_troops)
                neighbor_ratios.append(ratio)
        else:
            for candidate in candidates:
                enemy_troops = candidate.get("troops", 0)
                ratio = calculate_neighbor_ratio(population, enemy_troops)
                neighbor_ratios.append(ratio)

        return (
            in_spawn,
            population_pct,
            max_population,
            conquest_pct,
            tuple(neighbor_ratios),
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

        state = self.env.current_state or {}
        tick = state.get("tick", 0)
        me = state.get("me", {})
        pop = me.get("population", 0)
        max_pop = me.get("maxPopulation", 1)
        conquest_pct = me.get("conquestPercent", 0)

        in_spawn, pop_pct, max_pop_state, conquest_state, neighbor_ratios = (
            new_state_key
        )
        neighbors_str = ",".join(
            str(n) for n in neighbor_ratios[:MAX_NEIGHBORS_DISPLAY]
        )
        if len(neighbor_ratios) > MAX_NEIGHBORS_DISPLAY:
            neighbors_str += "..."
        state_str = f"S:({int(in_spawn)},{pop_pct},{max_pop_state},{conquest_state},({neighbors_str}))"

        status = f"\rTick: {tick:4d} | Pop: {pop:7d}/{max_pop:7d} | Conquest: {conquest_pct:2d}% | Reward: {self.reward:7.1f} | Total: {self.total_reward:8.1f} | R:{self.random_actions}/Q:{self.qtable_actions} | W:{self.wait_actions}/A:{self.attack_actions} | {state_str}"
        print(status + " " * 20, end="", flush=True)

        self.state = new_state_key

    async def best_action(self):
        self.state = self.get_state()
        possible_actions = self.env.get_possible_actions(self.env.current_state or {})

        if not possible_actions:
            return {"type": Action.NONE.value}

        if (
            random.random() < self.epsilon
            or (closest_state := await self.qtable.find_closest_state(self.state))
            is None
        ):
            self.random_actions += 1
            action = random.choice(possible_actions)
        else:
            action_keys = [self._get_action_key(a) for a in possible_actions]
            q_values = await self.qtable.get_state_actions(closest_state, action_keys)

            if not q_values:
                self.random_actions += 1
                action = random.choice(possible_actions)
            else:
                best_action_key = max(q_values, key=lambda k: q_values[k])
                action = None
                for a in possible_actions:
                    if self._get_action_key(a) == best_action_key:
                        self.qtable_actions += 1
                        action = a
                        break
                if action is None:
                    self.random_actions += 1
                    action = random.choice(possible_actions)

        if action.get("type") == Action.NONE.value:
            self.wait_actions += 1
        elif action.get("type") == Action.ATTACK.value:
            self.attack_actions += 1

        return action

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
