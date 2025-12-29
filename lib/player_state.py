from typing import Any


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
