import asyncio
import math
import random
import sys
from typing import Any

from websockets.asyncio.server import serve

from lib.bot_connection import BotConnection
from lib.constants import (
    ALPHA,
    CONQUEST_WIN_THRESHOLD,
    EPSILON,
    GAMMA,
    HIGH_POPULATION_THRESHOLD,
    INTERFACE_PORT,
    LEARNING_ASSISTANCE,
    LOW_POPULATION_THRESHOLD,
    MAX_NEIGHBORS_DISPLAY,
    MODE,
    PRECISION,
    REWARD_ATTACK_AT_LOW_POPULATION,
    REWARD_CONQUEST_GAIN,
    REWARD_CONQUEST_LOSS,
    REWARD_INVALID_ACTION,
    REWARD_SMALL_STEP,
    REWARD_SPAWN_FAILED,
    REWARD_SPAWN_SUCCESS,
    REWARD_VERY_HIGH_POPULATION,
    REWARD_VERY_LOW_POPULATION,
    REWARD_VICTORY,
)
from lib.player_state import PlayerState
from lib.qtable import QTable
from lib.server_interface import ServerInterface
from lib.utils import (
    Action,
    BuildingType,
    calculate_building_cost,
    get_action_key,
)


def calculate_neighbor_ratio(my_troops: int, enemy_troops: int) -> int:
    if enemy_troops == 0:
        return PRECISION
    if my_troops == 0:
        return -PRECISION
    ratio = my_troops / enemy_troops
    log_ratio = math.log2(ratio)
    scaled_value = (log_ratio / 2.0) * PRECISION
    return int(max(-PRECISION, min(PRECISION, round(scaled_value))))


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

        action_type = action.get("type") if action else None
        old_player = PlayerState(self.previous_state or {})
        new_player = PlayerState(self.current_state or {})
        reward = self.calculate_reward(old_player, new_player, action_type)

        return self.current_state, reward

    def rewards_territory(self, old: PlayerState, new: PlayerState) -> float:
        conquest_diff = new.conquest_percent - old.conquest_percent

        if conquest_diff > 0:
            return conquest_diff * REWARD_CONQUEST_GAIN
        if conquest_diff < 0:
            return abs(conquest_diff) * REWARD_CONQUEST_LOSS
        return 0.0

    def rewards_population(self, player: PlayerState) -> float:
        pop_ratio = player.population_ratio

        if pop_ratio < LOW_POPULATION_THRESHOLD:
            return REWARD_VERY_LOW_POPULATION
        if pop_ratio > HIGH_POPULATION_THRESHOLD:
            return REWARD_VERY_HIGH_POPULATION
        return 0.0

    def rewards_conquest(self, player: PlayerState) -> float:
        if player.conquest_percent >= CONQUEST_WIN_THRESHOLD:
            return REWARD_VICTORY
        return 0.0

    def rewards_attack_at_low_population(
        self, player: PlayerState, action: str | None
    ) -> float:
        if (
            action == Action.ATTACK.value
            and player.population_ratio < LOW_POPULATION_THRESHOLD
        ):
            return REWARD_ATTACK_AT_LOW_POPULATION
        return 0.0

    def rewards_spawn(
        self, old: PlayerState, new: PlayerState, action: str | None
    ) -> float:
        reward = 0.0

        if action == Action.SPAWN.value:
            reward += REWARD_SPAWN_SUCCESS

        prev_empty = [e for e in old.enemies if e.troops == 0]

        if old.in_spawn_phase and len(prev_empty) > 0 and new.population == 0:
            reward += REWARD_SPAWN_FAILED

        return reward

    def calculate_reward(
        self, old: PlayerState, new: PlayerState, action: str | None
    ) -> float:
        if new.in_spawn_phase:
            return self.rewards_spawn(old, new, action)

        if (
            LEARNING_ASSISTANCE == "low"
            and action is not None
            and (
                (action == Action.SPAWN.value and not old.in_spawn_phase)
                or (action == Action.ATTACK.value and old.population == 0)
                or (
                    action == Action.BUILD.value
                    and old.gold
                    < calculate_building_cost(BuildingType.CITY, old.city_count)
                )
            )
        ):
            return REWARD_INVALID_ACTION

        return (
            REWARD_SMALL_STEP
            + self.rewards_territory(old, new)
            + self.rewards_population(new)
            + self.rewards_conquest(new)
            + self.rewards_attack_at_low_population(old, action)
        )


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
        player = PlayerState(s)

        step_size = 10 / PRECISION
        conquest_pct = int(round(player.conquest_percent / step_size) * step_size)

        population_pct = int(
            round((player.population / player.max_population) * 100 / step_size)
            * step_size
            if player.max_population > 0
            else 0
        )

        neighbor_ratios = []
        for enemy in player.enemies:
            ratio = calculate_neighbor_ratio(player.population, enemy.troops)
            neighbor_ratios.append(ratio)

        city_cost = calculate_building_cost(BuildingType.CITY, player.city_count)
        can_afford_city = int(player.gold >= city_cost)

        return (
            player.in_spawn_phase,
            population_pct,
            conquest_pct,
            can_afford_city,
            tuple(neighbor_ratios),
        )

    async def do(self, action):
        previous_state = self.state
        self.state, self.reward = await self.env.do(action)
        new_state_key = self.get_state()
        prev_state_key = previous_state

        player = PlayerState(self.env.current_state or {})
        action_key = get_action_key(action, player, calculate_neighbor_ratio)

        current_q = await self.qtable.get_q_value(prev_state_key, action_key)
        max_next_q = await self.qtable.get_max_q_value(new_state_key)
        delta = self.alpha * (self.reward + self.gamma * max_next_q - current_q)
        new_q = current_q + delta
        await self.qtable.set_q_value(prev_state_key, action_key, new_q)

        self.score += self.reward
        self.total_reward += self.reward
        self.iterations += 1

        in_spawn, pop_pct, conquest_state, can_afford_city, neighbor_ratios = (
            new_state_key
        )
        neighbors_str = ",".join(
            str(n) for n in neighbor_ratios[:MAX_NEIGHBORS_DISPLAY]
        )
        if len(neighbor_ratios) > MAX_NEIGHBORS_DISPLAY:
            neighbors_str += "..."
        state_str = f"S:({int(in_spawn)},{pop_pct},{conquest_state},{can_afford_city},({neighbors_str}))"
        status = f"\rTick: {player.tick:4d} | Pop: {player.population:7d}/{player.max_population:7d} | Conquest: {player.conquest_percent:2d}% | Gold: {player.gold:6d} | Cities: {player.city_count} | Reward: {self.reward:7.1f} | Total: {self.total_reward:8.1f} | R:{self.random_actions}/Q:{self.qtable_actions} | W:{self.wait_actions}/A:{self.attack_actions} | {state_str}"
        print(status + " " * 20, end="", flush=True)

        self.state = new_state_key

    async def best_action(self):
        self.state = self.get_state()
        player = PlayerState(self.env.current_state or {})
        possible_actions = [{"type": Action.NONE.value}]

        SPAWN_REQUEST = {"type": Action.SPAWN.value, "x": -1, "y": -1}
        match LEARNING_ASSISTANCE:
            case "low":
                possible_actions.append(SPAWN_REQUEST)
                if not player.in_spawn_phase:
                    possible_actions.append(
                        {"type": Action.BUILD.value, "unit": BuildingType.CITY.value}
                    )
                    for idx, enemy in enumerate(player.enemies):
                        possible_actions.append(
                            {
                                "type": Action.ATTACK.value,
                                "neighbor_index": idx,
                                "x": enemy.x,
                                "y": enemy.y,
                                "ratio": 0.2,
                            }
                        )
            case "high" | _:
                if player.in_spawn_phase and player.owned_count == 0:
                    possible_actions = [SPAWN_REQUEST]
                elif player.enemies:
                    city_cost = calculate_building_cost(
                        BuildingType.CITY, player.city_count
                    )

                    if player.gold >= city_cost:
                        possible_actions.append(
                            {
                                "type": Action.BUILD.value,
                                "unit": BuildingType.CITY.value,
                            }
                        )

                    can_attack = (
                        LEARNING_ASSISTANCE != "high"
                        or player.population
                        >= HIGH_POPULATION_THRESHOLD * player.max_population
                    )
                    if can_attack:
                        for idx, enemy in enumerate(player.enemies):
                            possible_actions.append(
                                {
                                    "type": Action.ATTACK.value,
                                    "neighbor_index": idx,
                                    "x": enemy.x,
                                    "y": enemy.y,
                                    "ratio": 0.2,
                                }
                            )

        if random.random() < self.epsilon:
            self.random_actions += 1
            action = random.choice(possible_actions)
        else:
            action_keys = [
                get_action_key(a, player, calculate_neighbor_ratio)
                for a in possible_actions
            ]

            q_values = await self.qtable.get_state_actions(self.state, action_keys)

            if not q_values:
                self.random_actions += 1
                action = random.choice(possible_actions)
            else:
                best_action_key = max(q_values, key=lambda k: q_values[k])
                action = None
                for i, a in enumerate(possible_actions):
                    if action_keys[i] == best_action_key:
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
