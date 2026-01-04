from typing import Any

from lib.enemy import Enemy


class PlayerState:
    def __init__(self, state: dict[str, Any]):
        self._state = state
        self._me = state.get("me", {})

    @property
    def gold(self) -> int:
        return self._me.get("gold", 0)

    @property
    def population(self) -> int:
        return self._me.get("population", 0)

    @property
    def max_population(self) -> int:
        return self._me.get("maxPopulation", 1)

    @property
    def conquest_percent(self) -> int:
        return self._me.get("conquestPercent", 0)

    @property
    def owned_count(self) -> int:
        return self._me.get("ownedCount", 0)

    @property
    def buildings(self) -> dict:
        return self._me.get("buildings", {})

    @property
    def city_count(self) -> int:
        return self.buildings.get("cities", 0)

    @property
    def population_ratio(self) -> float:
        if self.max_population == 0:
            return 0.0
        return self.population / self.max_population

    @property
    def small_id(self) -> int | None:
        return self._me.get("smallID")

    @property
    def in_spawn_phase(self) -> bool:
        return bool(self._state.get("inSpawnPhase", False))

    @property
    def tick(self) -> int:
        return self._state.get("tick", 0)

    @property
    def enemies(self) -> list[Enemy]:
        candidates_data = self._state.get("candidates") or []
        return [Enemy(c) for c in candidates_data]

    @property
    def players(self) -> list:
        return self._state.get("players", [])

    @property
    def player_id(self) -> str | None:
        if self.small_id is None or self.small_id < 0:
            return None

        for player in self.players:
            if player.get("smallID") == self.small_id:
                return player.get("playerID")

        return None
