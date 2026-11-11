#!/usr/bin/env python3
"""Differentiable optimal control: swing a pendulum up on a cart, for thousands of pendulum lengths at once (JAX).

PHYSICAL SET-UP.  A cart of mass M slides on a horizontal rail under a horizontal force F (our control).  A pendulum of point mass m on a
massless rod of length l hangs from the cart; theta is its angle from the UPWARD vertical (theta = pi: hanging down, theta = 0: balanced up).

EQUATIONS (Lagrange).  Positions: cart (x, 0); bob (x + l sin(theta), l cos(theta)).
  kinetic   T = (M+m) xdot^2/2 + m l xdot thetadot cos(theta) + m l^2 thetadot^2/2
  potential V = m g l cos(theta)
  d/dt dL/dxdot - dL/dx = F     ->  (M+m) xddot + m l thetaddot cos(theta) - m l thetadot^2 sin(theta) = F
  d/dt dL/dthetadot - dL/dtheta = 0 ->  m l xddot cos(theta) + m l^2 thetaddot - m g l sin(theta) = 0
  Solving the two for the accelerations:
      xddot     = [ F + m l thetadot^2 sin(theta) - m g sin(theta) cos(theta) ] / ( M + m sin^2(theta) )
      thetaddot = ( g sin(theta) - xddot cos(theta) ) / l
ANALYTIC CHECK.  Linearise about hanging (theta = pi + phi): thetaddot = -(g/l)(1 + m/M) phi, so a free small swing has
  omega^2 = (g/l)(M+m)/M.   With F = 0 the energy E = T + V is conserved.

CONTROL PROBLEM.  F(t) piecewise constant in N intervals over time Tend; start at rest hanging (x = 0, theta = pi); minimise
  J = integral F^2 dt (effort)  +  wT [ x^2 + (1 - cos theta) * c1 + (xdot^2 + c2 thetadot^2) ]_end  (arrive upright, at rest, at x = 0).
The gradient dJ/dF_k through the integrator comes from reverse-mode automatic differentiation (jax.grad); it is checked against finite differences.
Adam (written out below, no extra library) updates F.  vmap runs many (length, initial-guess) pairs in parallel: that is what the GPU is for.

Run:  python3 swingup.py            (CPU, small: about a minute)
      LENGTHS=64 SEEDS=64 ITERS=3000 python3 swingup.py     (the Kaggle GPU run: 4096 optimisations)
"""
import os, json, time, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", os.environ.get("X64", "1") == "1")
dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# ---- parameters (named, SI units) ----------------------------------------------------------------------------------------
M, m, g = 1.0, 0.2, 9.81            # kg, kg, m/s^2
l_min, l_max = 0.25, 1.0            # m, range of pendulum lengths swept
Tend, N, SUB = 2.0, 80, 4           # s, control intervals, RK4 sub-steps per interval
wT, c1, c2 = 5000.0, 1.0, 0.1       # terminal penalty weights
LENGTHS = int(os.environ.get("LENGTHS", 8))
SEEDS = int(os.environ.get("SEEDS", 8))
ITERS = int(os.environ.get("ITERS", 800))
LR = float(os.environ.get("LR", 0.15))
OUT = os.environ.get("OUT", "out"); os.makedirs(OUT, exist_ok=True)
dt = Tend / N / SUB

def deriv(s, F, l):
    x, th, xd, thd = s
    sn, cs = jnp.sin(th), jnp.cos(th)
    xdd = (F + m * l * thd**2 * sn - m * g * sn * cs) / (M + m * sn**2)
    thdd = (g * sn - xdd * cs) / l
    return jnp.array([xd, thd, xdd, thdd])

def rk4(s, F, l, h):
    k1 = deriv(s, F, l); k2 = deriv(s + h / 2 * k1, F, l); k3 = deriv(s + h / 2 * k2, F, l); k4 = deriv(s + h * k3, F, l)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout(F, l, s0, h):
    """F: (N,) forces; returns states at the control-interval ends, shape (N+1, 4)."""
    def interval(s, f):
        s = jax.lax.fori_loop(0, SUB, lambda i, st: rk4(st, f, l, h), s)
        return s, s
    _, traj = jax.lax.scan(interval, s0, F)
    return jnp.concatenate([s0[None], traj])

def energy(s, l):
    x, th, xd, thd = s
    return 0.5 * (M + m) * xd**2 + m * l * xd * thd * jnp.cos(th) + 0.5 * m * l**2 * thd**2 + m * g * l * jnp.cos(th)

def cost(F, l):
    s0 = jnp.array([0.0, jnp.pi, 0.0, 0.0], dtype)
    traj = rollout(F, l, s0, dt)
    x, th, xd, thd = traj[-1]
    effort = jnp.sum(F**2) * (Tend / N)
    terminal = x**2 + c1 * (1 + jnp.cos(th - jnp.pi) * -1) * 0 + c1 * (1 - jnp.cos(th)) + xd**2 + c2 * thd**2
    return effort + wT * terminal

# ---------------------------------------------------------------------------------------------------------------- 1. physics tests
rep = {}
l0 = 0.5
# (a) small free swing about hanging: period against 2 pi / omega
phi0 = 0.01
s0 = jnp.array([0.0, jnp.pi + phi0, 0.0, 0.0], dtype)
hh = 1e-3
nsteps = int(3.0 / hh)
def step(s, _):
    s = rk4(s, 0.0, l0, hh); return s, s
_, tr = jax.lax.scan(step, s0, None, length=nsteps)
ph = np.asarray(tr[:, 1] - jnp.pi); t = hh * np.arange(1, nsteps + 1)
up = np.where((ph[:-1] < 0) & (ph[1:] >= 0))[0]
tc = t[up] - ph[up] * hh / (ph[up + 1] - ph[up])
T_num = np.mean(np.diff(tc)); T_th = 2 * np.pi / np.sqrt(g / l0 * (M + m) / M)
rep["period_numeric_s"], rep["period_analytic_s"], rep["period_rel_error"] = float(T_num), float(T_th), float(T_num / T_th - 1)
# (b) energy drift for a large free swing (F = 0), and RK4 order from halving the step
s0 = jnp.array([0.0, jnp.pi - 2.0, 0.0, 0.0], dtype)
def run(hh, T=2.0):
    n = int(round(T / hh)); _, trj = jax.lax.scan(lambda s, _: (rk4(s, 0.0, l0, hh), None), s0, None, length=n)
    return _
fin = [run(h_) for h_ in (0.02, 0.01, 0.005, 0.0025)]
ref = run(0.00125)
errs = [float(jnp.abs(f - ref).max()) for f in fin]
rep["rk4_errors"] = errs; rep["rk4_observed_orders"] = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
E0, E1 = float(energy(s0, l0)), float(energy(run(0.005), l0)); rep["energy_drift_rel"] = (E1 - E0) / abs(E0)
# (c) gradient of the cost against central finite differences
F_test = 3.0 * jnp.sin(jnp.linspace(0, 6, N)).astype(dtype)
gA = jax.grad(cost)(F_test, l0)
idx = [3, 20, 41, 77]
eps = 1e-6 if dtype == jnp.float64 else 3e-3     # a step below the float32 resolution would make both cost values identical
gN = np.array([(cost(F_test.at[i].add(eps), l0) - cost(F_test.at[i].add(-eps), l0)) / (2 * eps) for i in idx])
rep["gradient_rel_error"] = float(np.abs(np.asarray(gA)[idx] - gN).max() / np.abs(gN).max())
print("physics tests:", json.dumps(rep, indent=1))
assert abs(rep["period_rel_error"]) < 2e-3 and abs(rep["energy_drift_rel"]) < 1e-6 and rep["gradient_rel_error"] < (1e-5 if dtype == jnp.float64 else 5e-2), "a physics test failed"

# ---------------------------------------------------------------------------------------------------------------- 2. optimisation, batched
lengths = jnp.linspace(l_min, l_max, LENGTHS).astype(dtype)
key = jax.random.key(0)
kA, kw, kp = jax.random.split(key, 3)
A = jax.random.uniform(kA, (LENGTHS, SEEDS), minval=2.0, maxval=12.0)           # N, amplitude of the initial pumping force
w = jax.random.uniform(kw, (LENGTHS, SEEDS), minval=0.6, maxval=2.0) * jnp.sqrt(g / lengths)[:, None] * jnp.sqrt((M + m) / M)   # rad/s, near the free swing frequency
p = jax.random.uniform(kp, (LENGTHS, SEEDS), minval=0, maxval=2 * jnp.pi)
tk = (jnp.arange(N) + 0.5) * Tend / N
F0 = (A[..., None] * jnp.sin(w[..., None] * tk + p[..., None])).astype(dtype)          # (LENGTHS, SEEDS, N)

val_grad = jax.value_and_grad(cost)
def adam_step(carry, i):
    F, mom, vel, l = carry
    J, gr = val_grad(F, l)
    mom = 0.9 * mom + 0.1 * gr; vel = 0.999 * vel + 0.001 * gr**2
    mh = mom / (1 - 0.9 ** (i + 1)); vh = vel / (1 - 0.999 ** (i + 1))
    lr = LR * (0.02 + 0.98 * jnp.cos(0.5 * jnp.pi * i / ITERS) ** 2)      # decays to 2 % of LR so the last steps settle
    F = F - lr * mh / (jnp.sqrt(vh) + 1e-8)
    return (F, mom, vel, l), J
def optimise(F, l):
    (F, _, _, _), Js = jax.lax.scan(adam_step, (F, jnp.zeros_like(F), jnp.zeros_like(F), l), jnp.arange(ITERS))
    return F, Js
batched = jax.jit(jax.vmap(jax.vmap(optimise, in_axes=(0, None)), in_axes=(0, 0)))
t0 = time.time(); Fopt, Js = batched(F0, lengths); Fopt.block_until_ready(); t_first = time.time() - t0
t0 = time.time(); Fopt, Js = batched(F0, lengths); Fopt.block_until_ready(); t_run = time.time() - t0     # second call: no compilation
Jfin = np.asarray(Js[..., -1])                                                                           # (LENGTHS, SEEDS)
best = Jfin.argmin(axis=1)
Fbest = np.asarray(Fopt)[np.arange(LENGTHS), best]
def summary(Fk, l):
    traj = np.asarray(rollout(jnp.asarray(Fk, dtype), l, jnp.array([0.0, jnp.pi, 0.0, 0.0], dtype), dt))
    x, th, xd, thd = traj[-1]
    return dict(effort_N2s=float(np.sum(Fk**2) * Tend / N), x_end=float(x), theta_end_deg=float(np.degrees(((th + np.pi) % (2 * np.pi)) - np.pi)),
                xd_end=float(xd), thd_end=float(thd), peak_force_N=float(np.abs(Fk).max()))
res = [dict(length_m=float(lengths[i]), **summary(Fbest[i], lengths[i]), J=float(Jfin[i, best[i]])) for i in range(LENGTHS)]
arrived = [r for r in res if abs(r["theta_end_deg"]) < 2.0 and abs(r["x_end"]) < 0.05 and abs(r["thd_end"]) < 0.5]
print(f"{LENGTHS*SEEDS} optimisations of {ITERS} Adam steps: {t_run:.1f} s (first call with compilation {t_first:.1f} s) on {jax.devices()[0]}")
print(f"{len(arrived)} of {LENGTHS} lengths reach upright within 2 degrees; effort by length:",
      ", ".join(f"{r['length_m']:.2f} m: {r['effort_N2s']:.1f}" for r in res))
timing = dict(device=str(jax.devices()[0]), precision="float64" if dtype == jnp.float64 else "float32", optimisations=LENGTHS * SEEDS, adam_steps=ITERS,
              seconds_second_call=t_run, seconds_first_call=t_first, seconds_per_opt_step_per_problem=t_run / ITERS / (LENGTHS * SEEDS), jax=jax.__version__)
json.dump(dict(parameters=dict(M=M, m=m, g=g, l_min=l_min, l_max=l_max, Tend=Tend, N=N, SUB=SUB, wT=wT, LR=LR, ITERS=ITERS, LENGTHS=LENGTHS, SEEDS=SEEDS),
               physics_tests=rep, results=res, arrived=len(arrived), timing=timing), open(os.path.join(OUT, "result.json"), "w"), indent=1)
json.dump(timing, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
# data file for the website: the trajectory of the cheapest length that arrived (x, theta over time), small
cand = arrived or res
k = int(np.argmin([r["effort_N2s"] for r in cand])); li = [r["length_m"] for r in res].index(cand[k]["length_m"])
traj = np.asarray(rollout(jnp.asarray(Fbest[li], dtype), lengths[li], jnp.array([0.0, jnp.pi, 0.0, 0.0], dtype), dt))
json.dump(dict(l=float(lengths[li]), M=M, m=m, g=g, T=Tend, t=np.linspace(0, Tend, N + 1).round(5).tolist(), x=traj[:, 0].round(5).tolist(),
               theta=traj[:, 1].round(5).tolist(), F=np.append(Fbest[li], Fbest[li][-1]).round(4).tolist()), open(os.path.join(OUT, "swingup_data.json"), "w"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot([r["length_m"] for r in res], [r["effort_N2s"] for r in res], "o-"); ax[0].set_xlabel("pendulum length l (m)"); ax[0].set_ylabel(r"effort $\int F^2 dt$ (N$^2$ s)")
ax[1].plot(np.linspace(0, Tend, N + 1), np.degrees(traj[:, 1]), label=r"$\theta$ (deg)"); ax[1].plot(np.linspace(0, Tend, N + 1), traj[:, 0] * 100, label="x (cm)")
ax[1].set_xlabel("t (s)"); ax[1].legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "swingup.png"), dpi=130)
