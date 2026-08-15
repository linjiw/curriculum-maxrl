# Hopper lane recipe

What a working CPU campaign on Hopper actually requires, written from the
Acrobot U64 lane (2026-08-15). Follow this and a new lane should reach a
running array in one submission instead of four.

## Cluster facts

| item | value |
|---|---|
| host | `lwang44@hopper.orc.gmu.edu` |
| account | `xiao` |
| CPU partition | `normal`, 7-day limit, ~90 nodes (`amd*` and `hop*`) |
| GPU partition | `gpuq` + `--qos=gpu` + explicit MIG `--gres` |
| scratch | `/scratch/lwang44`, unbacked, 388 TB free |
| login Python | 3.11.5 |
| compute-node CPU seen | Intel Xeon Gold 6240R @ 2.40 GHz |

**Compute nodes are ~6x slower per core than the lab workstation** for
single-threaded NumPy. A 2M-transition Acrobot run is ~106 s locally and ~413 s
on `hop061`. Budget accordingly; do not extrapolate from local timings.

## The three defects that cost four submissions

1. **`/usr/bin/time` does not exist on compute nodes.** This also killed UED job
   9366863. Use the shell clock (`SECONDS`) and read memory from
   `sacct -j <id> -o MaxRSS`. Never put GNU `time` in an sbatch.

2. **`UV_OFFLINE=1` needs the interpreter cached where the job looks.** Setting
   `UV_PYTHON_INSTALL_DIR` at run time does not move an interpreter that was
   installed to uv's default path during warm-up. Warm up with the *same*
   environment variables the job will use, and verify offline on the login node
   before submitting.

3. **A whole-directory digest is self-invalidating.** Hashing every file under a
   staged bundle breaks as soon as the bundle runs, because CPython writes
   `__pycache__` into it. Exclude `__pycache__`/`*.pyc` **in both** the staging
   verifier and the sbatch verifier, using identical predicates, and export
   `PYTHONDONTWRITEBYTECODE=1` so the interpreter cannot mutate the tree it is
   being audited against.

## Environment: uv, offline, on scratch

`uv` is not installed on Hopper. Install it user-level (no sudo) and keep both
the package cache and the interpreters on scratch so compute nodes can read
them without network:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # -> ~/.local/bin/uv
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/scratch/lwang44/uv-cache            # ~91 MB for this lane
export UV_PYTHON_INSTALL_DIR=/scratch/lwang44/uv-python  # ~97 MB per interpreter
uv python install 3.12.13                                # MUST use the same vars
UV_OFFLINE=1 uv run --python 3.12.13 --with numpy==2.5.1 \
  --with 'gymnasium[classic-control]==1.3.0' python -c "print('offline ok')"
```

The last line is the gate: if it fails on the login node it will fail on every
compute node, and you will discover it one queue wait at a time.

## Pinning a runtime that a cluster can actually satisfy

Pin the fields that change results and record the rest:

- **compare**: `python_implementation`, `python`, `numpy`, `gymnasium`, `machine`
- **record only**: `platform` — it embeds the kernel build string
  (`6.8.0-124-generic` on the workstation vs `4.18.0-553…el8_10` on Hopper), is
  not a scientific variable, and is never uniform across a real cluster.

Requiring full `platform` equality makes a protocol unrunnable off the machine
that sealed it. That is exactly why the sealed V2 Acrobot tournament, locked on
macOS arm64, fail-closes on Hopper and on the lab workstation alike.

## Pairing and node heterogeneity

`normal` mixes `amd*` and `hop*` nodes, so a paired contrast can straddle CPU
models. **Run every arm of one paired unit inside one task on one node**
(one array task per seed, all arms in-process). Then record `cpu_model` and
`hostname` per run and have the analyzer refuse a campaign whose within-seed
arms landed on different CPUs. Pairing becomes exact by construction rather
than by assumption.

## The workflow

```bash
# 1. freeze, then stage a content-addressed bundle (refuses a dirty tree)
hopper/stage_acrobot_u64.sh

# 2. submit ONE task first; its output is real campaign data either way
./hopper/hopper.sh submit hopper/sbatch/<lane>.sbatch --array=0-0 \
  "--export=ALL,<LANE>_BUNDLE_DIR=...,<LANE>_BUNDLE_SHA256=..."

# 3. when it completes, submit the remainder
./hopper/hopper.sh submit hopper/sbatch/<lane>.sbatch --array=1-19 "--export=..."

# 4. scheduler metadata only while running
./hopper/hopper.sh status <job_id>
sacct -j <job_id> -X -P -o JobID,State,Elapsed,MaxRSS

# 5. fetch, verifying completeness before any analysis
hopper/fetch_acrobot_u64.sh <job_id> [<job_id>...]
```

`hopper.sh submit` refuses resource/identity flags on the command line so they
stay auditable in the sbatch file. `--array` and `--export` are allowed.

Each array submission writes under its own `SLURM_ARRAY_JOB_ID`, so a campaign
split across submissions lands in several directories; the fetch script merges
them, and the analyzer still requires a single shared lock digest across all
cells, so merging mismatched submissions fails closed.

## Make the sbatch fail closed

The sbatch should refuse to produce anything unless it can prove what it ran:

1. re-verify the bundle digest on the node before doing work;
2. derive the seed from `SLURM_ARRAY_TASK_ID` and reject it if outside the
   frozen block;
3. verify the source lock and run the arm-sanity check as preflight;
4. pin `OMP_NUM_THREADS=1` (and OPENBLAS/MKL/NUMEXPR) so concurrent tasks do
   not oversubscribe and BLAS threading cannot add nondeterminism;
5. assert the expected number of output files at the end and exit nonzero
   otherwise;
6. print a unique completion sentinel (`<LANE>_TASK_COMPLETE seed=…`) so
   completion is greppable and cannot be inferred from exit status alone.

## Timings for this lane

- one seed, four arms, 2M transitions each: **1,653 s** on `hop061`
- peak RSS: tens of MB; 4 GB request is generous
- 20 concurrent tasks scheduled immediately on `normal`
