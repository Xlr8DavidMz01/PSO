

from __future__ import annotations

import argparse

import numpy as np

from pso_simulation import PSO, PSOConfig


ANALYTICAL_SOLUTION = np.array([1.0, 2.0, -1.0])
LOWER_BOUND = np.array([-5.0, -5.0, -5.0])
UPPER_BOUND = np.array([5.0, 5.0, 5.0])


def coupled_objective(position: np.ndarray) -> float:


    x, y, z = position
    return float(
        (x - 1.0) ** 2
        + (y - 2.0) ** 2
        + (z + 1.0) ** 2
        + 10.0 * (x * y - 2.0) ** 2
        + 5.0 * (x + z) ** 2
    )


def parse_args() -> argparse.Namespace:
    """Parse optional settings while keeping reproducible defaults."""

    parser = argparse.ArgumentParser(
        description="Compare PSO's numerical solution with a known solution."
    )
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed position and objective errors for the comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Run PSO and compare its result with the analytical solution."""

    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    config = PSOConfig(
        n_particles=args.particles,
        dimensions=3,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        iterations=args.iterations,
        seed=args.seed,
    )

    analytical_value = coupled_objective(ANALYTICAL_SOLUTION)
    result = PSO(coupled_objective, config).optimize()

    position_error = float(np.linalg.norm(result.best_position - ANALYTICAL_SOLUTION))
    objective_error = abs(result.best_value - analytical_value)
    within_tolerance = (
        position_error <= args.tolerance
        and objective_error <= args.tolerance
    )

    print("Coupled three-variable objective")
    print("F(x, y, z) = (x - 1)^2 + (y - 2)^2 + (z + 1)^2")
    print("             + 10 * (x*y - 2)^2 + 5 * (x + z)^2")
    print()
    print(f"Analytical solution: {np.array2string(ANALYTICAL_SOLUTION, precision=10)}")
    print(f"Analytical minimum:  {analytical_value:.12g}")
    print()
    print(f"PSO solution:        {np.array2string(result.best_position, precision=10)}")
    print(f"PSO minimum:         {result.best_value:.12g}")
    print()
    print(f"Position error:      {position_error:.12g}")
    print(f"Objective error:     {objective_error:.12g}")
    print(f"Tolerance:           {args.tolerance:.12g}")
    print(f"Match within tolerance: {within_tolerance}")
    print(f"Particles:           {config.n_particles}")
    print(f"Iterations:          {config.iterations}")
    print(f"Seed:                {config.seed}")


if __name__ == "__main__":
    main()
