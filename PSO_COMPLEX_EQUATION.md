# Solving a Coupled Multivariable Equation with PSO

This walkthrough shows how to use the existing implementation in
[`pso_simulation.py`](pso_simulation.py) to minimize a new, coupled
three-variable objective.

The example is intentionally placed in a separate script:

```text
pso_complex_equation.py
```

The original PSO implementation is **not modified**. The new script imports
the existing `PSO` and `PSOConfig` classes and supplies a new objective
function to them.

## 1. Problem definition

We will minimize the following function:

\[
\begin{aligned}
F(x,y,z)={}&(x-1)^2+(y-2)^2+(z+1)^2\\
&+10(xy-2)^2+5(x+z)^2.
\end{aligned}
\]

This is a useful teaching example because it contains both independent and
coupled terms:

| Term | Role |
|---|---|
| `(x - 1)^2` | Encourages `x` to approach `1` |
| `(y - 2)^2` | Encourages `y` to approach `2` |
| `(z + 1)^2` | Encourages `z` to approach `-1` |
| `10 * (x*y - 2)^2` | Couples `x` and `y` |
| `5 * (x + z)^2` | Couples `x` and `z` |

The objective is a sum of squared terms. Therefore:

\[
F(x,y,z)\geq 0
\]

for every real `x`, `y`, and `z`.

At the point:

\[
(x,y,z)=(1,2,-1),
\]

every term is zero:

\[
\begin{aligned}
(1-1)^2 &= 0,\\
(2-2)^2 &= 0,\\
(-1+1)^2 &= 0,\\
(1\cdot 2-2)^2 &= 0,\\
(1+(-1))^2 &= 0.
\end{aligned}
\]

Consequently, the known analytical solution is:

\[
\boxed{(x,y,z)=(1,2,-1)}
\]

and the known minimum is:

\[
\boxed{F(1,2,-1)=0}.
\]

Because the first three squared terms can all be zero only at this point, the
global minimizer is unique for this example.

## 2. Python objective function

The mathematical function is represented in Python as:

```python
def coupled_objective(position: np.ndarray) -> float:
    x, y, z = position
    return float(
        (x - 1.0) ** 2
        + (y - 2.0) ** 2
        + (z + 1.0) ** 2
        + 10.0 * (x * y - 2.0) ** 2
        + 5.0 * (x + z) ** 2
    )
```

The PSO class expects an objective with this general interface:

```python
def objective(position: np.ndarray) -> float:
    return one_finite_scalar
```

Here, `position` is a NumPy vector with three elements:

```text
position = [x, y, z]
```

For example, evaluating the point `[3, 1, 0]` gives:

\[
\begin{aligned}
F(3,1,0)
&=(3-1)^2+(1-2)^2+(0+1)^2\\
&\quad+10(3\cdot1-2)^2+5(3+0)^2\\
&=4+1+1+10+45\\
&=61.
\end{aligned}
\]

The PSO algorithm does not need to know how the objective is calculated. It
only compares the returned values. A smaller value is better because this is a
minimization problem.

## 3. Reusing the original PSO implementation

The new script imports the existing classes directly:

```python
from pso_simulation import PSO, PSOConfig
```

No copy of the PSO algorithm is needed, and no changes are required in
`pso_simulation.py`.

The three-dimensional configuration is:

```python
config = PSOConfig(
    n_particles=40,
    dimensions=3,
    lower_bound=[-5.0, -5.0, -5.0],
    upper_bound=[5.0, 5.0, 5.0],
    iterations=300,
    seed=42,
)
```

The settings mean:

| Setting | Value | Meaning |
|---|---:|---|
| `n_particles` | `40` | 40 candidate solutions search simultaneously |
| `dimensions` | `3` | Each candidate contains `x`, `y`, and `z` |
| `lower_bound` | `[-5, -5, -5]` | Smallest allowed coordinate values |
| `upper_bound` | `[5, 5, 5]` | Largest allowed coordinate values |
| `iterations` | `300` | Number of swarm updates |
| `seed` | `42` | Makes this run reproducible |

The bounds contain the known solution `[1, 2, -1]`.

## 4. What happens during the run?

Initially, the swarm contains a matrix with shape `(40, 3)`:

```text
particle 1: [x1, y1, z1]
particle 2: [x2, y2, z2]
...
particle 40: [x40, y40, z40]
```

Each position is initialized randomly within the bounds. The objective is
evaluated for every particle. The initial position of each particle becomes
its personal best, and the best personal best becomes the initial global best.

For every iteration, the existing `PSO.step()` method performs this sequence:

```text
1. Generate independent random values r1 and r2.
2. Update every particle velocity.
3. Limit velocity to the configured maximum.
4. Move every particle.
5. Clip positions that leave the bounds.
6. Dampen and reverse outward velocity at a boundary.
7. Evaluate the new positions with coupled_objective.
8. Update personal bests when a value improves.
9. Update the global best when a value improves.
```

### 4.1 Velocity update

For particle `i` and coordinate `d`, the existing implementation uses:

\[
v_{i,d}(t+1)=wv_{i,d}(t)
+c_1r_{1,i,d}(p_{i,d}-x_{i,d}(t))
+c_2r_{2,i,d}(g_d-x_{i,d}(t)).
\]

The three terms represent:

1. **Inertia:** continue moving in the current direction.
2. **Cognitive pull:** move toward the particle's personal best.
3. **Social pull:** move toward the swarm's global best.

The position is then updated with:

\[
x_{i,d}(t+1)=x_{i,d}(t)+v_{i,d}(t+1).
\]

The objective function is evaluated after the movement. The PSO update does
not require derivatives of `F`; it only needs the value of `F` at candidate
positions.

## 5. Comparing PSO with the analytical solution

The script independently stores the known solution:

```python
ANALYTICAL_SOLUTION = np.array([1.0, 2.0, -1.0])
```

It evaluates the objective at that point:

```python
analytical_value = coupled_objective(ANALYTICAL_SOLUTION)
```

Then it runs PSO:

```python
result = PSO(coupled_objective, config).optimize()
```

The returned result contains:

```python
result.best_position
result.best_value
```

The script calculates the position error using the Euclidean norm:

\[
e_x=\left\|x_{\text{PSO}}-x_{\text{analytical}}\right\|_2.
\]

In Python:

```python
position_error = float(
    np.linalg.norm(result.best_position - ANALYTICAL_SOLUTION)
)
```

It also calculates the objective error:

\[
e_F=|F_{\text{PSO}}-F_{\text{analytical}}|.
\]

In Python:

```python
objective_error = abs(result.best_value - analytical_value)
```

The result is reported as matching when both errors are within the selected
tolerance:

```python
within_tolerance = (
    position_error <= tolerance
    and objective_error <= tolerance
)
```

This checks both important aspects:

- Did PSO find approximately the correct variables?
- Did those variables produce approximately the correct minimum value?

## 6. Run the example

From the PSO directory, install the existing dependencies if necessary:

```bash
python -m pip install -r requirements.txt
```

Then run the new script:

```bash
python pso_complex_equation.py
```

The script does not generate a plot. It is a direct numerical comparison and
does not require Matplotlib or Pillow at runtime; it only needs NumPy and the
existing `pso_simulation.py` module.

The default command uses:

```text
40 particles
3 dimensions
300 iterations
random seed 42
tolerance 0.001
```

The number of particles, iterations, seed, and comparison tolerance can be
changed from the command line:

```bash
python pso_complex_equation.py \
    --particles 60 \
    --iterations 500 \
    --seed 7 \
    --tolerance 0.001
```

On Windows PowerShell, the command can also be written on one line:

```powershell
python pso_complex_equation.py --particles 60 --iterations 500 --seed 7 --tolerance 0.001
```

## 7. Interpreting the output

The output has this structure:

```text
Coupled three-variable objective
F(x, y, z) = ...

Analytical solution: [ 1.  2. -1.]
Analytical minimum:  0

PSO solution:        [approximately 1, approximately 2, approximately -1]
PSO minimum:         a small non-negative value

Position error:      a small value
Objective error:     a small value
Tolerance:           0.001
Match within tolerance: True
```

The PSO solution may look like:

```text
[ 1.00001  1.99998 -1.00002]
```

rather than exactly:

```text
[1.0, 2.0, -1.0]
```

That is normal. PSO is an iterative stochastic optimizer, not a symbolic
algebra system. The important checks are that the errors are small and that
the objective value is close to zero.

## 8. Why the result can vary

PSO uses random initialization, random velocities, and random coefficients in
the velocity update. The script uses `seed=42` by default so that the default
run is reproducible.

To test robustness, run multiple seeds:

```powershell
python pso_complex_equation.py --seed 1
python pso_complex_equation.py --seed 2
python pso_complex_equation.py --seed 3
```

Each run may follow a different path. A stronger experiment records the final
objective and position errors across many seeds instead of relying on one
successful run.

If a run does not reach the desired tolerance, possible adjustments include:

- increasing `--iterations`;
- increasing `--particles`;
- checking that the bounds include the expected solution;
- trying a different seed;
- changing the inertia, cognitive, or social coefficients in the new script's
  `PSOConfig`.

The objective has a known global minimum and moderate bounds, so the default
settings should generally be sufficient for this demonstration.

## 9. Complete data flow

The complete data flow is:

```text
known equation
      |
      v
coupled_objective(position) -> one scalar objective value
      |
      v
PSO(coupled_objective, config)
      |
      v
random three-dimensional particles
      |
      v
velocity and position updates
      |
      v
best position found by the swarm
      |
      +-----------------------------+
      |                             |
      v                             v
compare position with         compare objective value with
      |                             |
      +-------------+---------------+
                    v
          print errors and tolerance result
```

The objective is passed as a callable. This is the key design that allows the
same PSO implementation to work with Sphere, Rastrigin, Rosenbrock, or a
user-defined equation without changing the optimizer.
