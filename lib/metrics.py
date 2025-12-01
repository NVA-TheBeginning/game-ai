from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Any

mpl.use("Agg")


class GameMetrics:
    def __init__(self) -> None:
        self.game_scores: list[float] = []
        self.game_durations: list[int] = []

        self.current_game_start_tick: int | None = None
        self.current_game_score: float = 0.0

    def start_game(self) -> None:
        self.current_game_start_tick = None
        self.current_game_score = 0.0

    def update_tick(self, tick: int) -> None:
        if self.current_game_start_tick is None:
            self.current_game_start_tick = tick

    def add_reward(self, reward: float) -> None:
        self.current_game_score += reward

    def end_game(self, final_tick: int) -> None:
        duration = (
            final_tick - self.current_game_start_tick
            if self.current_game_start_tick is not None
            else final_tick
        )

        print(
            f"\nGame ended - Score sent to graph: {self.current_game_score:.2f}, Duration: {duration} ticks"
        )
        self.game_scores.append(self.current_game_score)
        self.game_durations.append(duration)

    def generate_graphs(
        self, output_dir: str = ".", filename: str = "metrics.png"
    ) -> str:
        if not self.game_scores:
            return ""

        output_path = Path(output_dir) / filename

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("AI Learning Progress", fontsize=16, fontweight="bold")

        game_numbers = list(range(1, len(self.game_scores) + 1))

        # Plot 1: Total Reward per Game
        axes[0].plot(game_numbers, self.game_scores, marker="o", linewidth=2)
        axes[0].set_title("Total Reward per Game")
        axes[0].set_xlabel("Game Number")
        axes[0].set_ylabel("Total Reward")
        axes[0].grid(visible=True, alpha=0.3)
        if len(self.game_scores) > 1:
            avg_score = sum(self.game_scores) / len(self.game_scores)
            axes[0].axhline(
                y=avg_score,
                color="r",
                linestyle="--",
                label="Average",
                alpha=0.7,
            )
            axes[0].legend()

        # Plot 2: Game Duration (ticks)
        axes[1].plot(
            game_numbers, self.game_durations, marker="s", color="green", linewidth=2
        )
        axes[1].set_title("Game Duration (ticks)")
        axes[1].set_xlabel("Game Number")
        axes[1].set_ylabel("Ticks")
        axes[1].grid(visible=True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return str(output_path)

    def get_summary(self) -> dict[str, Any]:
        if not self.game_scores:
            return {}

        return {
            "total_games": len(self.game_scores),
            "avg_score": sum(self.game_scores) / len(self.game_scores),
            "avg_duration": sum(self.game_durations) / len(self.game_durations),
        }
