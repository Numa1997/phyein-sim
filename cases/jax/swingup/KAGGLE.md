# Running the swing-up on the Kaggle GPU: exact clicks

What this runs: 4096 differentiable optimisations (64 pendulum lengths × 64 random starts, 3000 Adam steps each) of a cart-pendulum swing-up.
Physics, equations and checks are in the docstring of `swingup.py`. Output: `result.json`, `timing.json`, `swingup_data.json`, `swingup.png`.

## Before you start (what was measured here, and what was not)

| | measured | where |
|---|---|---|
| Forward model against theory | small-swing period within 2·10⁻⁵ of the analytic value; RK4 order 4.0 / 3.0 / 3.8 (64-bit); energy drift 5·10⁻⁹ (64-bit), 6·10⁻⁷ (32-bit) | this sandbox, CPU |
| Gradient (reverse-mode AD) against central finite differences | 4·10⁻⁸ (64-bit), 4·10⁻³ (32-bit) | this sandbox, CPU |
| 256 optimisations × 3000 steps, 32-bit | 95 s; 6 of 8 lengths arrive upright within about 2.2°, at rest | this sandbox, CPU |
| Physics tests on a Kaggle T4, 32-bit (12 Nov 2026) | period within 2·10⁻⁵, energy drift 1.3·10⁻⁶ (inside the 32-bit rounding bound 2.4·10⁻⁵), gradient 5·10⁻³ | Kaggle |
| The full run on the Kaggle GPU | see `HANDOVER.md` (the first run's `timing.json`) | Kaggle |

Why 32-bit: Kaggle's default GPU (T4) is fast in 32-bit numbers and about 32 times slower in 64-bit. The 32-bit forward model and gradient pass their checks
(numbers above), so the run uses `X64 = "0"`. Which random start wins differs between precisions (the search is not convex), so compare the best-of-64 efforts,
not single runs.

Expected time: this job is 1.2·10⁷ optimisation steps. On this sandbox's CPU in 32-bit that would take about 25 minutes. A T4 should be faster; a hard limit
is Kaggle's 12-hour session. Estimates are not measurements.

## Route A: by hand on kaggle.com (needs no connector)

Screen labels change from time to time; cell 1 of the notebook is the real check that the GPU is there.

1. Sign in at kaggle.com. The account must be phone-verified, or GPUs are not offered.
2. Left menu: **Create** → **New Notebook**.
3. In the notebook: **File** → **Import Notebook** (or the upload icon), choose `kaggle_swingup.ipynb` from this folder.
4. Right-hand panel **Session options** (or **Settings**) → **Accelerator** → choose **GPU T4 ×2**. (Not P100: it is being retired.)
5. Same panel: **Internet** → **on** (cell 1 installs the CUDA build of JAX if Kaggle's preinstalled one sees no GPU)
   `!pip install -U "jax[cuda12]"`, then restart the session.
6. **Run All** (top bar). Cell 1 prints the GPU name and fails at once if none is seen. Cell 4 runs a few-second smoke test, then the full run.
7. When it has finished, open **Output** in the right panel (or the **Data** tab → **Output**), and download `full/swingup_data.json`, `full/swingup.png`,
   `full/timing.json`, `full/result.json`.
8. Send the four files to Claude, or put `swingup_data.json` into the website page (see `docs/website-delivery.md`).

GPU quota: the weekly GPU hours of the account are shared by all notebooks; only two GPU sessions can be active at once.

## Route B: Claude launches it through Kaggle's connector

One-time set-up (claude.ai or the desktop app): Settings → Connectors → Add custom connector → Name `Kaggle`, URL `https://www.kaggle.com/mcp`, leave the advanced fields
empty → Add → Connect and sign in → switch the connector on for the chat. Then say, in a new session:

> Run `cases/jax/swingup/kaggle_swingup.ipynb` on the Kaggle GPU with the kaggle-gpu-run skill. Show me the quota first.

Claude shows the notebook title, visibility (private), accelerator, estimated hours and the remaining quota, and waits for a yes for that exact run.

## If something goes wrong

| What you see | Meaning | What to do |
|---|---|---|
| Cell 1 stops: "No GPU visible" | Accelerator not set, or JAX has no CUDA build | Step 4; then step 5 (Internet on, pip line) |
| `Out of memory` | 4096 problems at once is too many for the card | In cell 4 set `SEEDS = 32` (halves the memory), or `LENGTHS = 32` |
| Fewer than 6 of 8 lengths upright | Search is not convex | Raise `SEEDS` (more starts), or `ITERS` to 6000 |
| Much slower than the CPU estimate | The 64-bit setting is on | `X64 = "0"` in cell 4 |
