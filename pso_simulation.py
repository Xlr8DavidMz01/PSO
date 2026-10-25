

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class PSOConfig:

    n_particles: int = 30
    dimensions: int = 2
    lower_bound: float | Sequence[float] = -5.12
    upper_bound: float | Sequence[float] = 5.12
    iterations: int = 200
    inertia: float = 0.7298
    cognitive: float = 1.49618
    social: float = 1.49618
    velocity_clamp: Optional[float | Sequence[float]] = None
    seed: Optional[int] = 42

    def validate(self) -> None:

        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.inertia < 0:
            raise ValueError("inertia must be non-negative")
        if self.cognitive < 0 or self.social < 0:
            raise ValueError("cognitive and social coefficients must be non-negative")


@dataclass
class PSOResult:
    """Outputs collected from one optimization run."""

    best_position: np.ndarray
    best_value: float
    best_value_history: np.ndarray
    global_best_positions_history: np.ndarray
    mean_value_history: np.ndarray
    positions_history: np.ndarray
    velocity_history: np.ndarray
    iterations_completed: int

    @property
    def convergence_history(self) -> np.ndarray:
        """Alias useful when discussing the global-best curve."""

        return self.best_value_history


class PSO:

    def __init__(self, objective: ObjectiveFunction, config: PSOConfig):
        config.validate()
        self.objective = objective
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.lower = self._as_dimension_vector(config.lower_bound, "lower_bound")
        self.upper = self._as_dimension_vector(config.upper_bound, "upper_bound")
        if np.any(self.lower >= self.upper):
            raise ValueError("Every lower_bound must be smaller than upper_bound")

        if config.velocity_clamp is None:
            # A conservative default prevents a particle from crossing its
            # entire search range in one update.
            self.velocity_max = 0.2 * (self.upper - self.lower)
        else:
            self.velocity_max = self._as_dimension_vector(
                config.velocity_clamp, "velocity_clamp"
            )
            if np.any(self.velocity_max <= 0):
                raise ValueError("velocity_clamp values must be positive")

        self.positions = self.rng.uniform(
            self.lower, self.upper, size=(config.n_particles, config.dimensions)
        )
        velocity_range = self.velocity_max
        self.velocities = self.rng.uniform(
            -velocity_range, velocity_range,
            size=(config.n_particles, config.dimensions),
        )
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = self._evaluate(self.positions)
        best_index = int(np.argmin(self.personal_best_values))
        self.global_best_position = self.personal_best_positions[best_index].copy()
        self.global_best_value = float(self.personal_best_values[best_index])

    def _as_dimension_vector(
        self, value: float | Sequence[float], name: str
    ) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.ndim == 0:
            vector = np.full(self.config.dimensions, float(vector))
        if vector.shape != (self.config.dimensions,):
            raise ValueError(
                f"{name} must be a scalar or a sequence of length "
                f"{self.config.dimensions}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray([self.objective(position) for position in positions], dtype=float)
        if values.shape != (len(positions),) or not np.all(np.isfinite(values)):
            raise ValueError("objective must return one finite scalar per position")
        return values

    def _apply_bounds(self) -> None:

        below = self.positions < self.lower
        above = self.positions > self.upper
        self.positions = np.clip(self.positions, self.lower, self.upper)
        self.velocities[below | above] *= -0.5

    def step(self) -> float:
        """Perform one PSO iteration and return the current global best value."""

        n_particles = self.config.n_particles
        dimensions = self.config.dimensions
        r1 = self.rng.random((n_particles, dimensions))
        r2 = self.rng.random((n_particles, dimensions))

        self.velocities = (
            self.config.inertia * self.velocities
            + self.config.cognitive * r1
            * (self.personal_best_positions - self.positions)
            + self.config.social * r2
            * (self.global_best_position - self.positions)
        )
        self.velocities = np.clip(
            self.velocities, -self.velocity_max, self.velocity_max
        )
        self.positions += self.velocities
        self._apply_bounds()

        values = self._evaluate(self.positions)
        improved = values < self.personal_best_values
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_values[improved] = values[improved]

        best_index = int(np.argmin(self.personal_best_values))
        if self.personal_best_values[best_index] < self.global_best_value:
            self.global_best_value = float(self.personal_best_values[best_index])
            self.global_best_position = self.personal_best_positions[best_index].copy()
        return self.global_best_value

    def optimize(self) -> PSOResult:
        """Run the configured number of iterations and collect diagnostics."""

        best_history = []
        global_best_positions_history = [self.global_best_position.copy()]
        mean_history = []
        positions_history = [self.positions.copy()]
        velocity_history = [self.velocities.copy()]

        for _ in range(self.config.iterations):
            self.step()
            best_history.append(self.global_best_value)
            global_best_positions_history.append(self.global_best_position.copy())
            mean_history.append(float(np.mean(self.personal_best_values)))
            positions_history.append(self.positions.copy())
            velocity_history.append(self.velocities.copy())

        return PSOResult(
            best_position=self.global_best_position.copy(),
            best_value=float(self.global_best_value),
            best_value_history=np.asarray(best_history),
            global_best_positions_history=np.asarray(global_best_positions_history),
            mean_value_history=np.asarray(mean_history),
            positions_history=np.asarray(positions_history),
            velocity_history=np.asarray(velocity_history),
            iterations_completed=self.config.iterations,
        )


def sphere(position: np.ndarray) -> float:
    """Sphere function: global minimum f(0)=0."""

    return float(np.sum(position**2))


def rastrigin(position: np.ndarray) -> float:
    """Multimodal Rastrigin function: global minimum f(0)=0."""

    dimension = len(position)
    return float(10 * dimension + np.sum(position**2 - 10 * np.cos(2 * np.pi * position)))


def rosenbrock(position: np.ndarray) -> float:
    """Rosenbrock's curved valley function: global minimum f(1,...,1)=0."""

    return float(np.sum(100 * (position[1:] - position[:-1] ** 2) ** 2
                       + (1 - position[:-1]) ** 2))


OBJECTIVES: dict[str, tuple[ObjectiveFunction, tuple[float, float]]] = {
    "sphere": (sphere, (-5.12, 5.12)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-2.0, 2.0)),
}


def save_history_csv(result: PSOResult, path: Path) -> None:
    """Save iteration-level convergence metrics in a spreadsheet-friendly file."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "global_best", "mean_personal_best"])
        for iteration, (best, mean) in enumerate(
            zip(result.best_value_history, result.mean_value_history), start=1
        ):
            writer.writerow([iteration, f"{best:.12g}", f"{mean:.12g}"])


def save_result_json(result: PSOResult, config: PSOConfig, path: Path) -> None:
    """Save the final answer and convergence history as JSON."""

    payload = {
        "config": asdict(config),
        "best_position": result.best_position.tolist(),
        "best_value": result.best_value,
        "iterations_completed": result.iterations_completed,
        "best_value_history": result.best_value_history.tolist(),
        "mean_value_history": result.mean_value_history.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_result(
    result: PSOResult,
    objective: ObjectiveFunction,
    lower: np.ndarray,
    upper: np.ndarray,
    path: Path,
) -> None:
    """Save convergence and, for two dimensions, swarm movement plots."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install requirements.txt or use --no-plot."
        ) from error

    if len(result.best_position) == 2:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(result.best_value_history, label="global best")
        axes[0].plot(result.mean_value_history, label="mean personal best", alpha=0.8)
        axes[0].set(xlabel="iteration", ylabel="objective value", title="Convergence")
        axes[0].set_yscale("symlog", linthresh=1e-10)
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        grid_x = np.linspace(lower[0], upper[0], 180)
        grid_y = np.linspace(lower[1], upper[1], 180)
        xx, yy = np.meshgrid(grid_x, grid_y)
        zz = np.asarray([objective(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
        axes[1].contourf(xx, yy, zz.reshape(xx.shape), levels=35, cmap="viridis")
        positions = result.positions_history
        axes[1].plot(
            positions[:, :, 0], positions[:, :, 1], color="white", alpha=0.18, linewidth=0.7
        )
        axes[1].scatter(
            positions[0, :, 0], positions[0, :, 1], label="initial", c="orange", s=25
        )
        axes[1].scatter(
            positions[-1, :, 0], positions[-1, :, 1], label="final", c="red", s=25
        )
        axes[1].scatter(
            result.best_position[0], result.best_position[1], label="best", c="cyan", marker="*", s=150
        )
        axes[1].set(xlabel="x[0]", ylabel="x[1]", title="Particle movement")
        axes[1].legend()
    else:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(result.best_value_history, label="global best")
        axis.plot(result.mean_value_history, label="mean personal best", alpha=0.8)
        axis.set(xlabel="iteration", ylabel="objective value", title="Convergence")
        axis.set_yscale("symlog", linthresh=1e-10)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def animate_result(
    result: PSOResult,
    objective: ObjectiveFunction,
    lower: np.ndarray,
    upper: np.ndarray,
    path: Path,
    fps: int = 12,
    max_frames: int = 100,
) -> None:


    if len(result.best_position) != 2:
        raise ValueError("Animation requires exactly two dimensions")
    if fps < 1 or max_frames < 2:
        raise ValueError("fps must be positive and max_frames must be at least 2")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as error:
        raise RuntimeError(
            "Animation requires matplotlib and Pillow. Install requirements.txt."
        ) from error

    positions = result.positions_history
    velocities = result.velocity_history
    total_steps = len(positions) - 1
    frame_indices = np.unique(
        np.linspace(0, total_steps, min(total_steps + 1, max_frames), dtype=int)
    )

    # Include the initial state in the animated convergence curve.
    initial_values = np.asarray([objective(position) for position in positions[0]])
    best_values = np.concatenate(([float(np.min(initial_values))], result.best_value_history))

    grid_x = np.linspace(lower[0], upper[0], 180)
    grid_y = np.linspace(lower[1], upper[1], 180)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = np.asarray(
        [objective(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())]
    ).reshape(xx.shape)

    figure, (swarm_axis, convergence_axis) = plt.subplots(1, 2, figsize=(13, 5))
    swarm_axis.contourf(xx, yy, zz, levels=35, cmap="viridis")
    swarm_axis.set_xlim(lower[0], upper[0])
    swarm_axis.set_ylim(lower[1], upper[1])
    swarm_axis.set(xlabel="x[0]", ylabel="x[1]", title="PSO swarm movement")
    swarm_axis.text(
        0.02,
        0.98,
        "arrows = velocity\nstar = global best",
        transform=swarm_axis.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 4},
    )

    convergence_axis.set_xlim(0, total_steps)
    finite_values = best_values[np.isfinite(best_values)]
    y_min = min(float(np.min(finite_values)), 0.0)
    y_max = float(np.max(finite_values))
    convergence_axis.set_ylim(y_min - 0.05 * max(1.0, abs(y_min)), y_max + 0.05 * max(1.0, abs(y_max)))
    convergence_axis.set(xlabel="iteration", ylabel="global-best objective", title="Convergence")
    convergence_axis.grid(alpha=0.25)

    particles = swarm_axis.scatter(
        positions[0, :, 0], positions[0, :, 1], c="white", edgecolors="black", s=42
    )
    best_marker, = swarm_axis.plot(
        [result.global_best_positions_history[0, 0]],
        [result.global_best_positions_history[0, 1]],
        marker="*",
        color="cyan",
        markeredgecolor="black",
        markersize=16,
        linestyle="",
    )
    trail_lines = [
        swarm_axis.plot([], [], color="white", alpha=0.3, linewidth=0.8)[0]
        for _ in range(len(positions[0]))
    ]
    velocity_arrows = swarm_axis.quiver(
        positions[0, :, 0],
        positions[0, :, 1],
        velocities[0, :, 0],
        velocities[0, :, 1],
        color="yellow",
        alpha=0.75,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
    )
    convergence_line, = convergence_axis.plot(
        [0], [best_values[0]], color="tab:red", linewidth=2, label="global best"
    )
    convergence_axis.legend()

    def update(frame_number: int):
        step = int(frame_indices[frame_number])
        current_positions = positions[step]
        current_velocities = velocities[step]
        particles.set_offsets(current_positions)
        best_marker.set_data(
            [result.global_best_positions_history[step, 0]],
            [result.global_best_positions_history[step, 1]],
        )
        velocity_arrows.set_offsets(current_positions)
        velocity_arrows.set_UVC(current_velocities[:, 0], current_velocities[:, 1])
        for particle_index, trail in enumerate(trail_lines):
            trail.set_data(
                positions[: step + 1, particle_index, 0],
                positions[: step + 1, particle_index, 1],
            )
        convergence_line.set_data(
            frame_indices[: frame_number + 1], best_values[frame_indices[: frame_number + 1]]
        )
        figure.suptitle(
            f"Particle Swarm Optimization | iteration {step}/{total_steps} | "
            f"best = {best_values[step]:.6g}"
        )
        return [particles, best_marker, velocity_arrows, convergence_line, *trail_lines]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(figure)


def run_demo(args: argparse.Namespace) -> None:
    objective, default_bounds = OBJECTIVES[args.function]
    config = PSOConfig(
        n_particles=args.particles,
        dimensions=args.dimensions,
        lower_bound=default_bounds[0],
        upper_bound=default_bounds[1],
        iterations=args.iterations,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        seed=args.seed,
    )
    result = PSO(objective, config).optimize()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(result, output_dir / f"{args.function}_history.csv")
    save_result_json(result, config, output_dir / f"{args.function}_result.json")
    lower = np.asarray(
        config.lower_bound
        if np.ndim(config.lower_bound)
        else [config.lower_bound] * config.dimensions
    )
    upper = np.asarray(
        config.upper_bound
        if np.ndim(config.upper_bound)
        else [config.upper_bound] * config.dimensions
    )
    if not args.no_plot:
        plot_result(
            result,
            objective,
            lower,
            upper,
            output_dir / f"{args.function}_pso.png",
        )
    if args.animate:
        animate_result(
            result,
            objective,
            lower,
            upper,
            output_dir / f"{args.function}_pso.gif",
            fps=args.animation_fps,
            max_frames=args.animation_frames,
        )

    print(f"Function: {args.function}")
    print(f"Best value: {result.best_value:.12g}")
    print(f"Best position: {np.array2string(result.best_position, precision=6)}")
    print(f"Iterations: {result.iterations_completed}")
    print(f"Artifacts: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Particle Swarm Optimization simulation.")
    parser.add_argument("--function", choices=sorted(OBJECTIVES), default="rastrigin")
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--inertia", type=float, default=0.7298)
    parser.add_argument("--cognitive", type=float, default=1.49618)
    parser.add_argument("--social", type=float, default=1.49618)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib output")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Save an animated 2-D particle-search GIF",
    )
    parser.add_argument("--animation-fps", type=int, default=12)
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=100,
        help="Maximum GIF frames; long runs are sampled evenly",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
