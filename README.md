# Particle Swarm Optimization (PSO): an expert, practical guide

This guide explains PSO from intuition to equations to production choices. It
is written to be read alongside [`pso_simulation.py`](pso_simulation.py), where
each important equation is implemented directly with NumPy.

---

## 1. What problem does PSO solve?

PSO searches for a good solution to an optimization problem of the form

\[
\min_{x \in \Omega} f(x),
\]

where:

- `x` is a vector of decision variables, for example `[temperature, speed]`;
- `f(x)` is the objective (loss, cost, error, energy, or risk);
- `Ω` is the feasible search region, usually represented by lower and upper
  bounds for each variable.

PSO is especially useful when `f` is a **black box**: it may be nonlinear,
non-convex, noisy, discontinuous, simulation-based, expensive, or impossible
to differentiate. Instead of following a gradient, PSO evaluates candidate
solutions and moves them using experience shared by a population.

The algorithm is a **stochastic, population-based, derivative-free,
bounded optimizer**. It is a heuristic: it can find excellent solutions, but
unless the problem has special mathematical structure, it does not prove that

---

## 2. The flocking intuition

Imagine a flock searching a landscape for the lowest valley:

1. Each bird is a **particle** and represents one candidate solution.
2. A particle remembers its best location so far: its **personal best**
   (`pbest`).
3. The flock knows the best location found by any particle: the **global best**
   (`gbest`).
4. Each particle has a velocity. It is pulled by:
   - its current direction of travel;
   - its own successful memory;
   - the successful memory of the swarm.
5. Random coefficients make particles explore different routes rather than
   moving identically.

The resulting tension between exploration and exploitation is the central idea:

- **Exploration** samples new regions and reduces the chance of missing a
  promising basin.
- **Exploitation** concentrates evaluations near good solutions.

---

## 3. Core vocabulary and state

For `N` particles and `D` dimensions, the swarm stores:

| Symbol | Shape | Meaning |
|---|---:|---|
| `x_i(t)` | `D` | Position of particle `i` at iteration `t` |
| `v_i(t)` | `D` | Velocity of particle `i` |
| `p_i` | `D` | Best position ever visited by particle `i` |
| `f(p_i)` | scalar | Objective value at `p_i` |
| `g` | `D` | Best personal-best position in the whole swarm |
| `f(g)` | scalar | Global-best objective value |
| `w` | scalar | Inertia weight |
| `c_1` | scalar | Cognitive/personal coefficient |
| `c_2` | scalar | Social/swarm coefficient |
| `r_1, r_2` | `D` | Independent random values in `[0, 1]` |

`pbest` and `gbest` must be compared using the correct direction. This project
implements minimization, so a smaller objective is better. For maximization,
minimize the negated objective: `lambda x: -score(x)`.

---

## 4. The canonical PSO equations

### 4.1 Velocity update

For each particle and each coordinate:

\[
v_i(t+1) = wv_i(t)
 + c_1r_1(t)\left[p_i-x_i(t)\right]
 + c_2r_2(t)\left[g-x_i(t)\right].
\]

Interpretation:

- `w v_i(t)` is inertia. A large `w` preserves momentum and promotes search;
  a small `w` makes the swarm settle more quickly.
- `c1 r1 (p_i - x_i)` is the cognitive pull toward the particle's own best.
- `c2 r2 (g - x_i)` is the social pull toward the swarm's best.
- `r1` and `r2` are normally redrawn for every particle and coordinate at every
  iteration. This vector form is more exploratory than using one random scalar
  for the entire swarm.

### 4.2 Position update

\[
x_i(t+1) = x_i(t) + v_i(t+1).
\]

After moving, the implementation handles bounds, evaluates the objective,
updates `pbest`, and then potentially updates `gbest`.

### 4.3 What happens in one iteration?

The exact logical sequence is:

```text
draw r1 and r2
calculate new velocity from inertia, pbest, and gbest
optionally clamp velocity
move each particle
repair positions that left the bounds
evaluate every new position
replace pbest only when the new value improves it
replace gbest only when a pbest improves it
record diagnostics
```

The order matters. `gbest` for an iteration is based on knowledge available
before the update; newly evaluated positions become memories after the move.

---

## 5. A worked numerical update

Suppose a one-dimensional particle has:

```text
x(t)       = 4.0
v(t)       = 0.5
pbest      = 2.0
gbest      = 1.0
w          = 0.7
c1         = 1.4
c2         = 1.8
r1         = 0.25
r2         = 0.50
```

Then:

```text
inertia  = 0.7 * 0.5                       =  0.350
cognitive = 1.4 * 0.25 * (2.0 - 4.0)       = -0.700
social    = 1.8 * 0.50 * (1.0 - 4.0)       = -2.700
v(t+1) = 0.350 - 0.700 - 2.700             = -3.050
x(t+1) = 4.0 - 3.050                       =  0.950
```

The particle moves toward the two known good locations. If the legal interval
were `[0, 1]`, the new position would be valid. If it were `[0, 0.8]`, a
boundary policy would repair it, such as clipping it to `0.8` and damping or
reversing the outward velocity.

---

## 6. What is implemented in this project?

### `PSOConfig`

`PSOConfig` validates the main choices:

- number of particles and dimensions;
- lower and upper bounds (scalar or a per-dimension vector);
- iteration count;
- `inertia`, `cognitive`, and `social` coefficients;
- optional maximum velocity per coordinate;
- random seed for reproducibility.

The default coefficients (`0.7298`, `1.49618`, `1.49618`) are a commonly used
constricted/stable setting. They are a useful starting point, not a universal
law.

### Initialization

Positions are sampled uniformly within the bounds. Velocities are sampled in
`[-vmax, vmax]`, where the default `vmax` is 20% of each search-range width.
Every initial position is also its particle's initial `pbest`; the best of
those is the initial `gbest`.

### Evaluation

The objective is a Python callable:

```python
def objective(x: np.ndarray) -> float:
    return float(...)
```

It must return one finite scalar. Non-finite values are rejected because NaN
or infinity can silently corrupt comparisons and convergence statistics.

### Boundary handling

`pso_simulation.py` uses clipping:

```python
x = np.clip(x, lower, upper)
```

The outward velocity component is multiplied by `-0.5`. This both brings a
particle back into the search space and reduces repeated wall collisions.
Other legitimate policies include reflection, random reinitialization, or a
penalty objective. The right choice depends on whether a boundary is a hard
physical limit or merely a convenient search limit.

### Diagnostics

`PSOResult` records:

- final best position and objective value;
- global-best history;
- mean personal-best history;
- all positions and velocities, which enable trajectory plots.

For a large production run, recording every particle at every iteration may
consume memory. Store only sampled snapshots or aggregate metrics in that case.

---

## 7. Running the examples

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a two-dimensional multimodal problem:

```bash
python pso_simulation.py --function rastrigin --particles 30 --iterations 200
```

Run without plotting, which is useful on a server:

```bash
python pso_simulation.py --function sphere --dimensions 10 \
    --particles 40 --iterations 300 --no-plot
```

Try a curved valley:

```bash
python pso_simulation.py --function rosenbrock --iterations 500 --seed 7
```

### Animated visual explanation

For a visual explanation of the algorithm, generate a GIF from a 2-D run:

```bash
python pso_simulation.py \
    --function rastrigin \
    --iterations 120 \
    --animate \
    --animation-frames 80 \
    --output-dir outputs
```

The resulting `rastrigin_pso.gif` has two synchronized panels:

- **Left:** the objective landscape, moving particles, white position trails,
  yellow velocity arrows, and a cyan star for the current `gbest`.
- **Right:** the global-best objective curve, showing the best-so-far value as
  the iterations progress.

The animation shows the central PSO loop in order: particles move according to
inertia, personal memory, and social memory; the objective is evaluated; and
the best-so-far marker is retained or improved. `--animation-frames` samples
long runs to control GIF size, while `--animation-fps` controls playback speed.
Animation is limited to two dimensions because higher-dimensional particle
movement cannot be represented directly on a 2-D landscape.

The three included objectives have known global minima:

| Objective | Domain used by the demo | Global minimizer | Minimum |
|---|---|---|---:|
| Sphere | `[-5.12, 5.12]^D` | `(0, ..., 0)` | `0` |
| Rastrigin | `[-5.12, 5.12]^D` | `(0, ..., 0)` | `0` |
| Rosenbrock | `[-2, 2]^D` | `(1, ..., 1)` | `0` |

Use a seed when learning or comparing parameter choices. Remove or vary it when
you want to measure robustness across independent stochastic runs.

### Use PSO from another Python file

```python
import numpy as np

from pso_simulation import PSO, PSOConfig


def tuning_loss(x: np.ndarray) -> float:
    learning_rate, regularization = x
    return (learning_rate - 0.01) ** 2 + 4 * (regularization - 0.2) ** 2


config = PSOConfig(
    n_particles=25,
    dimensions=2,
    lower_bound=[0.0001, 0.0],
    upper_bound=[0.1, 1.0],
    iterations=100,
    seed=123,
)
result = PSO(tuning_loss, config).optimize()
print(result.best_position, result.best_value)
```

If dimensions have very different numerical scales, use meaningful per-variable
bounds. Normalizing variables to `[0, 1]` is often even better because PSO's
velocity and coefficients then have comparable effects across dimensions.

---

## 8. Understanding the hyperparameters

### Number of particles (`N`)

More particles improve spatial coverage but increase objective evaluations per
iteration. Start around 20–50 for a small continuous problem. Use more when:

- the dimension is high;
- the landscape is strongly multimodal;
- evaluations are cheap;
- diversity collapses too quickly.

There is no formula that guarantees the correct swarm size. Compare several
values using repeated seeds and a fixed evaluation budget.

### Number of iterations

Iterations are a budget of population updates. The actual expensive quantity is
usually `N × iterations` objective evaluations. For expensive simulations,
compare configurations with the same evaluation budget rather than blindly
increasing iterations.

### Inertia (`w`)

- Larger inertia: longer moves, more exploration, slower settling.
- Smaller inertia: shorter moves, more exploitation, possible premature
  convergence.

A linearly decreasing schedule, for example from `0.9` to `0.4`, is a common
experiment: explore early, refine late. Adaptive schedules should be validated
against a fixed baseline rather than assumed to be better.

### Cognitive coefficient (`c1`)

Controls the pull to personal experience. A larger value encourages particles
to preserve independent discoveries and can maintain diversity.

### Social coefficient (`c2`)

Controls the pull to the swarm's discovery. A larger value accelerates sharing
but can make every particle rush to one region and get trapped in a local
minimum.

### Velocity clamp (`vmax`)

Velocity clamping limits a single jump in each coordinate. It is especially
useful when the objective is sensitive, bounds are tight, or numerical scale is
poor. Too small a clamp makes exploration painfully slow; too large a clamp
makes the clamp ineffective.

### Random seed

PSO's randomness is part of its search mechanism, not a bug. A seed provides
reproducibility for debugging and demos. For a credible experiment, run many
seeds and report mean, median, standard deviation, and best/worst performance.

---

## 9. Exploration, exploitation, and premature convergence

The swarm can fail in two opposite ways:

1. **Too much exploration:** positions keep moving widely and do not refine a
   good area.
2. **Too much exploitation:** particles become nearly identical around an early
   attractive point and never investigate better basins.

Useful diagnostics include:

- global-best curve: should be non-increasing for minimization;
- mean personal-best curve: shows whether the whole swarm is improving;
- coordinate-wise swarm spread or standard deviation;
- number of distinct/unique candidates for discrete problems;
- repeated-run distribution of final objective values.

Possible remedies for collapse include decreasing social pressure, increasing
inertia early, increasing swarm size, using a neighborhood topology, restarting
stagnant particles, or adding a mutation/diversity mechanism. Every remedy
adds behavior that should be evaluated on the actual problem.

---

## 10. Where and why should you use PSO?

PSO is a strong candidate when:

- evaluating a candidate is straightforward but deriving a gradient is hard;
- the objective is non-convex or has many local minima;
- variables are continuous and naturally bounded;
- the objective is noisy or discontinuous;
- a reasonably good solution is more valuable than a mathematical proof of
  optimality;
- parallel objective evaluations are possible.

Typical applications include:

| Area | Example decision variables |
|---|---|
| Engineering design | dimensions, material parameters, component sizes |
| Control | controller gains, trajectory parameters |
| Machine learning | hyperparameters, feature weights, architecture settings |
| Signal/image processing | filter parameters, segmentation thresholds |
| Energy systems | generator dispatch, storage sizing, controller settings |
| Scheduling | ordering, assignment, timing; usually requires discrete adaptations |
| Robotics | path waypoints, motion-planning parameters |
| Finance | portfolio weights or model parameters, subject to constraints |
| Simulation calibration | parameters that make a model match observations |
| Feature selection | binary inclusion decisions, using binary PSO rather than this continuous version |

PSO is often a good **outer-loop optimizer** around a simulator or a model
whose internal gradients are unavailable. It can also tune a gradient-based
model's hyperparameters, while the model itself is trained with a different
algorithm.

### When another method may be better

- A smooth, differentiable, high-dimensional objective with reliable gradients:
  use gradient-based optimization first.
- A convex problem: use a convex solver for guarantees and efficiency.
- A tiny discrete search space: enumerate or use a combinatorial optimizer.
- Very expensive evaluations: consider Bayesian optimization, surrogate models,
  caching, or reduced-order modeling.
- A strict global-optimality or feasibility certificate is required: use a
  method with the appropriate mathematical guarantees.

PSO is not automatically superior to Differential Evolution, CMA-ES, simulated
annealing, Bayesian optimization, or domain-specific solvers. Select by
objective type, cost, constraints, dimensionality, and required guarantees.

---

## 11. Constraints and real-world modeling

Simple box bounds are only one kind of constraint. For a constraint
`g_j(x) <= 0`, common approaches are:

1. **Penalty:** optimize `f(x) + λ * violation(x)`. Easy, but `λ` is difficult
   to calibrate and can distort the landscape.
2. **Repair:** map an infeasible candidate to a feasible one. Effective when a
   sensible projection exists.
3. **Feasibility-first comparison:** prefer any feasible point over an
   infeasible point; among feasible points compare objective value; among
   infeasible points compare total violation.
4. **Specialized constrained PSO:** maintain feasibility or use multipliers and
   constraint-handling rules.

Do not silently encode an invalid solution as a very good objective. Explicitly
test constraints after every repair and log violation statistics.

For categorical, integer, permutation, or binary variables, continuous
position updates are not enough. Use rounding only when it is defensible;
otherwise use a discrete/binary/permutation representation with operators that
preserve validity.

---

## 12. Complexity and engineering considerations

For `N` particles, `D` dimensions, and `T` iterations:

- PSO arithmetic is approximately `O(TND)`;
- objective cost is usually `O(TN × cost(f))` and dominates;
- in this implementation, stored position and velocity histories require
  `O(TND)` memory.

Practical improvements:

- vectorize cheap objective functions;
- parallelize independent objective evaluations for expensive simulations;
- cache repeated evaluations when the objective is deterministic;
- stop on a target objective or stagnation threshold when appropriate;
- record snapshots rather than full histories for long runs;
- scale/normalize variables and choose bounds carefully;
- use independent seeds for statistical comparisons.

The `PSO` class currently calls the objective once per particle using a Python
loop so that arbitrary Python callables work. A domain-specific implementation
can accept a batch objective that evaluates an `(N, D)` matrix at once.

---

## 13. Correctness checklist for an implementation

When implementing or reviewing PSO, verify:

- positions are initialized inside bounds;
- personal bests start at the initial positions;
- global best is the best personal best, not merely the last evaluated point;
- `r1` and `r2` are regenerated as intended;
- the signs in `(pbest - position)` and `(gbest - position)` are correct;
- velocity and position updates happen in the intended order;
- the objective direction (minimize/maximize) is explicit;
- non-finite objective results are handled;
- bounds and constraints are enforced;
- global best is monotonic non-increasing for minimization;
- a fixed seed gives repeatable output;
- a known benchmark can be recovered within a reasonable tolerance;
- multiple seeds are used before claiming robustness.

---

## 14. Variants worth knowing

### Local-best (lbest) PSO

Instead of every particle seeing one global best, a particle sees the best in a
neighborhood. This usually spreads information more slowly and can preserve
diversity on multimodal landscapes.

### Constriction-factor PSO

A constriction factor `χ` scales the velocity update to help control dynamic
stability:

\[
v(t+1)=\chi\left[v(t)+c_1r_1(pbest-x)+c_2r_2(gbest-x)\right].
\]

The inertia-weight and constriction-factor forms are related but should not be
combined casually; their parameter settings come from different analyses.

### Adaptive PSO

Changes inertia, coefficients, neighborhood, or mutation based on progress or
diversity. Adaptive methods can help, but their extra choices create more
opportunities for overfitting a benchmark.

### Binary and discrete PSO

Uses a velocity-derived probability or problem-specific move operator. The
continuous equation cannot directly represent a valid permutation schedule.

### Multi-objective PSO (MOPSO)

Maintains a Pareto archive rather than one `gbest`. A solution is better if it
is non-dominated; diversity in objective space is also managed.

### Hybrid PSO

Uses PSO for broad search and a local optimizer for final refinement. This is
often useful when the objective has both a difficult global landscape and a
smooth local basin.

---


