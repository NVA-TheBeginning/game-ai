from enum import Enum
from typing import TYPE_CHECKING, Any

from lib.constants import (
    CITY_BASE_COST,
    CITY_MAX_COST,
    DEFENSE_POST_BASE_COST,
    DEFENSE_POST_MAX_COST,
    PORT_BASE_COST,
    PORT_MAX_COST,
)

if TYPE_CHECKING:
    from lib.player_state import PlayerState

FLOAT_TOLERANCE = 1e-9


class Action(Enum):
    SPAWN = "spawn"
    ATTACK = "attack"
    BUILD = "build_unit"
    NONE = "none"


class BuildingType(Enum):
    CITY = "City"
    PORT = "Port"
    POST = "Defense Post"


def get_action_key(action: dict[str, Any], player: PlayerState, ratio_fn) -> str:
    action_type = action.get("type")

    if action_type == Action.SPAWN.value:
        return Action.SPAWN.value

    if action_type == Action.BUILD.value:
        return f"build:{action.get('unit')}"

    if action_type == Action.ATTACK.value:
        neighbor_idx = action.get("neighbor_index")
        troop_ratio = action.get("ratio")

        if (
            neighbor_idx is None
            or not isinstance(neighbor_idx, int)
            or neighbor_idx >= len(player.enemies)
        ):
            return Action.NONE.value

        strength_ratio = ratio_fn(
            player.population, player.enemies[neighbor_idx].troops
        )
        return f"attack:{strength_ratio}|{troop_ratio}"

    return Action.NONE.value


def calculate_building_cost(building_type: BuildingType, current_count: int) -> int:
    if building_type == BuildingType.CITY:
        return min(CITY_MAX_COST, 2**current_count * CITY_BASE_COST)
    if building_type == BuildingType.PORT:
        return min(PORT_MAX_COST, 2**current_count * PORT_BASE_COST)
    if building_type == BuildingType.POST:
        return min(DEFENSE_POST_MAX_COST, (current_count + 1) * DEFENSE_POST_BASE_COST)
    return 0
