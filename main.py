import asyncio
import random
import sys
from typing import Any

from websockets.asyncio.server import serve

from lib.bot_connection import BotConnection
from lib.constants import (
    ALPHA,
    ATTACK_RATIOS,
    CONQUEST_WIN_THRESHOLD,
    EPSILON,
    GAMMA,
    HIGH_POPULATION_THRESHOLD,
    INTERFACE_PORT,
    LOW_POPULATION_THRESHOLD,
    MAX_NEIGHBORS_DISPLAY,
    MODE,
    RATIO_2_TIMES_HIGHER,
    RATIO_2_TIMES_LOWER,
    RATIO_3_TIMES_HIGHER,
    RATIO_3_TIMES_LOWER,
    RATIO_4_TIMES_HIGHER,
    RATIO_4_TIMES_LOWER,
    RATIO_EQUAL,
    REWARD_ATTACK_AT_LOW_POPULATION,
    REWARD_CONQUEST_GAIN,
    REWARD_CONQUEST_LOSS,
    REWARD_SMALL_STEP,
    REWARD_SPAWN_FAILED,
    REWARD_SPAWN_SUCCESS,
    REWARD_VERY_HIGH_POPULATION,
    REWARD_VERY_LOW_POPULATION,
    REWARD_VICTORY,
    SPAWN_PHASE_DURATION,
)
from lib.player_state import PlayerState
from lib.qtable import QTable
from lib.server_interface import ServerInterface
from lib.utils import Action, BuildingType, calculate_building_cost, get_action_key


def calculate_neighbor_ratio(my_troops: int, enemy_troops: int) -> int:
    if enemy_troops == 0:
        return 3
    if my_troops == 0:
        return -3

    ratio = my_troops / enemy_troops
    thresholds = [
        (RATIO_4_TIMES_HIGHER, 3),
        (RATIO_3_TIMES_HIGHER, 2),
        (RATIO_2_TIMES_HIGHER, 1),
        (RATIO_EQUAL, 0),
        (RATIO_2_TIMES_LOWER, -1),
        (RATIO_3_TIMES_LOWER, -2),
        (RATIO_4_TIMES_LOWER, -3),
    ]

    for threshold, value in thresholds:
        if ratio >= threshold:
            return value
    return -4


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
        old_conquest = old_me.get("conquestPercent", 0)
        new_conquest = new_me.get("conquestPercent", 0)

        conquest_diff = new_conquest - old_conquest

        if conquest_diff > 0:
            return conquest_diff * REWARD_CONQUEST_GAIN
        if conquest_diff < 0:
            return abs(conquest_diff) * REWARD_CONQUEST_LOSS
        return 0.0

    def rewards_population(self, new_state: dict[str, Any]) -> float:
        player = PlayerState(new_state or {})
        pop_ratio = player.population_ratio

        if pop_ratio < LOW_POPULATION_THRESHOLD:
            return REWARD_VERY_LOW_POPULATION
        if pop_ratio > HIGH_POPULATION_THRESHOLD:
            return REWARD_VERY_HIGH_POPULATION
        return 0.0

    def rewards_conquest(self, new_state: dict[str, Any]) -> float:
        player = PlayerState(new_state or {})
        if player.conquest_percent >= CONQUEST_WIN_THRESHOLD:
            return REWARD_VICTORY
        return 0.0

    def rewards_attack_at_low_population(
        self, old_state: dict[str, Any], action: dict[str, Any] | None
    ) -> float:
        if action and action.get("type") == Action.ATTACK.value:
            player = PlayerState(old_state or {})
            if player.population_ratio < LOW_POPULATION_THRESHOLD:
                return REWARD_ATTACK_AT_LOW_POPULATION
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
            reward += REWARD_SPAWN_FAILED

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
            + self.rewards_attack_at_low_population(old_state, action)
        )

    def can_afford_building(
        self, building_type: BuildingType, state: dict[str, Any]
    ) -> bool:
        me = state.get("me", {})
        gold = me.get("gold", 0)
        buildings = me.get("buildings", {})
        count = buildings.get(building_type.value.lower() + "s", 0)
        cost = calculate_building_cost(building_type, count)
        return gold >= cost

    def get_possible_actions(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = state.get("candidates") or []

        if state.get("inSpawnPhase"):
            me = state.get("me") or {}
            has_spawned = me.get("ownedCount", 0) > 0
            if not has_spawned:
                return [{"type": Action.SPAWN.value, "x": -1, "y": -1}]
            return [{"type": Action.NONE.value}]

        if not candidates:
            return [{"type": Action.NONE.value}]

        actions = [{"type": Action.NONE.value}]

        actions.append(
            {"type": Action.BUILD.value, "unit": BuildingType.CITY.value}
        )

        for ratio in ATTACK_RATIOS:
            for idx, candidate in enumerate(candidates):
                actions.append(
                    {
                        "type": Action.ATTACK.value,
                        "neighbor_index": idx,
                        "x": candidate.get("x"),
                        "y": candidate.get("y"),
                        "ratio": ratio,
                    }
                )

        return actions


class Agent:
    def __init__(self, env):
        self.env = env
        self.reward = 0
        self.qtable = QTable.get_instance()
        self.score = None
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON
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
        conquest_pct = round(me.get("conquestPercent", 0) / 5) * 5

        if max_population > 0:
            population_pct = round((population / max_population) * 100 / 5) * 5
        else:
            population_pct = 0

        neighbor_ratios = []
        for candidate in candidates:
            enemy_troops = candidate.get("troops", 0)
            ratio = calculate_neighbor_ratio(population, enemy_troops)
            neighbor_ratios.append(ratio)

        buildings = me.get("buildings", {})
        gold = me.get("gold", 0)
        city_count = buildings.get("cities", 0)
        city_cost = calculate_building_cost(BuildingType.CITY, city_count)
        can_afford_city = int(gold >= city_cost)

        return (
            in_spawn,
            population_pct,
            conquest_pct,
            can_afford_city,
            tuple(neighbor_ratios),
        )

    async def _update_q_value(self, prev_state, action, new_state):
        action_key = get_action_key(action)
        current_q = await self.qtable.get_q_value(prev_state, action_key)
        max_next_q = await self.qtable.get_max_q_value(new_state)
        delta = self.alpha * (self.reward + self.gamma * max_next_q - current_q)
        new_q = current_q + delta
        await self.qtable.set_q_value(prev_state, action_key, new_q)

    def _format_status_line(self, state_key, state):
        tick = state.get("tick", 0)
        player = PlayerState(state)
        in_spawn, pop_pct, conquest_state, can_afford_city, neighbor_ratios = state_key
        neighbors_str = ",".join(
            str(n) for n in neighbor_ratios[:MAX_NEIGHBORS_DISPLAY]
        )
        if len(neighbor_ratios) > MAX_NEIGHBORS_DISPLAY:
            neighbors_str += "..."
        state_str = f"S:({int(in_spawn)},{pop_pct},{conquest_state},({neighbors_str}))"
        return f"\rTick: {tick:4d} | Pop: {player.population:7d}/{player.max_population:7d} | Conquest: {player.conquest_percent:2d}% | Gold: {player.gold:6d} | Cities: {player.city_count} | Reward: {self.reward:7.1f} | Total: {self.total_reward:8.1f} | R:{self.random_actions}/Q:{self.qtable_actions} | W:{self.wait_actions}/A:{self.attack_actions} | {state_str}"

    async def do(self, action):
        previous_state = self.state
        self.state, self.reward = await self.env.do(action)
        new_state_key = self.get_state()

        await self._update_q_value(previous_state, action, new_state_key)

        self.score += self.reward
        self.total_reward += self.reward
        self.iterations += 1

        status = self._format_status_line(new_state_key, self.env.current_state or {})
        print(status + " " * 20, end="", flush=True)

        self.state = new_state_key

    async def best_action(self):
        self.state = self.get_state()
        possible_actions = self.env.get_possible_actions(self.env.current_state or {})

        if not possible_actions:
            return {"type": Action.NONE.value}

        if random.random() < self.epsilon:
            self.random_actions += 1
            action = random.choice(possible_actions)
        else:
            action_keys = [get_action_key(a) for a in possible_actions]
            q_values = await self.qtable.get_state_actions(self.state, action_keys)

            if not q_values:
                self.random_actions += 1
                action = random.choice(possible_actions)
            else:
                best_action_key = max(q_values, key=lambda k: q_values[k])
                action = None
                for a in possible_actions:
                    if get_action_key(a) == best_action_key:
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
