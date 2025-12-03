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
        self.game_wins: list[int] = []
        self.qtable_sizes: list[int] = []
        self.epsilons: list[float] = []

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

    def end_game(
        self,
        final_tick: int,
        win: bool = False,
        qtable_size: int | None = None,
        epsilon: float | None = None,
    ) -> None:
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
        self.game_wins.append(1 if win else 0)
        self.qtable_sizes.append(qtable_size if qtable_size is not None else 0)
        self.epsilons.append(epsilon if epsilon is not None else 0.0)

    def generate_graphs(
        self, output_dir: str = ".", filename: str = "metrics.png"
    ) -> str:
        if not self.game_scores:
            return ""

        output_path = Path(output_dir) / filename

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("AI Learning Progress", fontsize=16, fontweight="bold")

        game_numbers = list(range(1, len(self.game_scores) + 1))

        def moving_average(values: list[float], window: int = 5) -> list[float]:
            if not values:
                return []
            smoothed: list[float] = []
            for i in range(len(values)):
                start = max(0, i - window + 1)
                window_slice = values[start : i + 1]
                smoothed.append(sum(window_slice) / len(window_slice))
            return smoothed

        # Plot 1: Total Reward per Game + moving average
        axes[0, 0].plot(game_numbers, self.game_scores, marker="o", linewidth=2)
        axes[0, 0].set_title("Total Reward per Game")
        axes[0, 0].set_xlabel("Game Number")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(visible=True, alpha=0.3)
        if len(self.game_scores) > 1:
            smoothed_scores = moving_average(self.game_scores, window=5)
            axes[0, 0].plot(
                game_numbers,
                smoothed_scores,
                color="r",
                linestyle="--",
                label="Moving Avg (w=5)",
                linewidth=2,
            )
            axes[0, 0].legend()

        # Plot 2: Win rate (cumulative)
        if self.game_wins:
            cumulative_win_rate = [
                sum(self.game_wins[: i + 1]) / (i + 1) for i in range(len(game_numbers))
            ]
            axes[0, 1].plot(
                game_numbers,
                cumulative_win_rate,
                marker="s",
                color="green",
                linewidth=2,
            )
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title("Win Rate (cumulative)")
        axes[0, 1].set_xlabel("Game Number")
        axes[0, 1].set_ylabel("Win Rate")
        axes[0, 1].grid(visible=True, alpha=0.3)

        # Plot 3: Game Duration (ticks)
        axes[1, 0].plot(
            game_numbers, self.game_durations, marker="s", color="green", linewidth=2
        )
        axes[1, 0].set_title("Game Duration (ticks)")
        axes[1, 0].set_xlabel("Game Number")
        axes[1, 0].set_ylabel("Ticks")
        axes[1, 0].grid(visible=True, alpha=0.3)

        # Plot 4: Q-table size per game (with epsilon on twin axis)
        if self.qtable_sizes:
            axes[1, 1].plot(
                game_numbers,
                self.qtable_sizes,
                marker="^",
                color="purple",
                linewidth=2,
                label="Q-table size",
            )
        axes[1, 1].set_title("Q-table Size & Epsilon per Game")
        axes[1, 1].set_xlabel("Game Number")
        axes[1, 1].set_ylabel("States", color="purple")
        axes[1, 1].tick_params(axis="y", labelcolor="purple")
        axes[1, 1].grid(visible=True, alpha=0.3)

        ax_eps = axes[1, 1].twinx()
        if self.epsilons:
            ax_eps.plot(
                game_numbers,
                self.epsilons,
                marker="d",
                color="orange",
                linewidth=2,
                label="Epsilon",
            )
        ax_eps.set_ylabel("Epsilon", color="orange")
        ax_eps.tick_params(axis="y", labelcolor="orange")

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return str(output_path)

    def get_summary(self) -> dict[str, Any]:
        if not self.game_scores:
            return {}

        win_rate = sum(self.game_wins) / len(self.game_wins) if self.game_wins else 0.0
        avg_qtable_size = (
            sum(self.qtable_sizes) / len(self.qtable_sizes)
            if self.qtable_sizes
            else 0.0
        )

        return {
            "total_games": len(self.game_scores),
            "avg_score": sum(self.game_scores) / len(self.game_scores),
            "avg_duration": sum(self.game_durations) / len(self.game_durations),
            "win_rate": win_rate,
            "avg_qtable_size": avg_qtable_size,
            "last_epsilon": self.epsilons[-1] if self.epsilons else None,
        }
