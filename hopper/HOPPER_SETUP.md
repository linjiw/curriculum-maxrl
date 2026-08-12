# GMU Hopper cluster — access, resource request, and job scripts

**Prepared 2026-08-12** from the ORC wiki (Hopper_Quick_Start_Guide, Running_GPU_Jobs, Getting_an_ORC_Account). Everything below is ready; the two starred steps require your GMU NetID and cannot be done by an agent.

## What Hopper is for in this project (and what it is NOT for)

| Workload | Hopper? | Why |
|---|---|---|
| **E2c (frozen Countdown protocol)** | **NO — never** | The preregistration locks the local runtime, model/data paths, launcher, and GPU gate; B1/B2 comparator seeds were trained on the local RTX 5090. Moving it breaks comparator parity and the preflight rejects environment drift by design. E2c completes only via the unchanged local driver. |
| ICRA BARN 4-arm × 5-seed matrix + N-ablation (CPU) | **YES — best use** | Gazebo/ROS2 campaign is CPU-only; Hopper `normal` partition nodes (48 cores, 3-day limit) can run seeds in parallel instead of serially on the shared lab box. Needs the ROS2+Gazebo Apptainer container (template below). |
| WS4 gate replication (3 Countdown runs, corrected decay) | YES | A new study may preregister Hopper as its environment. One `A100.80gb` fits the vLLM+FSDP stack easily (local runs need ~20 GB). |
| Post-submission neural-scale score test (the referee's "nothing above 640 params" ceiling) | YES | The strongest possible rebuttal-window experiment; A100 80GB nodes are exactly right. |
| Isaac Lab fallback | MAYBE | A100s lack RT cores; Isaac Sim rendering is not supported on them. Physics-only headless training may work — verify before planning on it. |

## ★ Step 1 — Account request (you, ~5 minutes, requires GMU SSO)

- Eligibility: all GMU faculty/staff/students; student/staff requests need a **faculty sponsor** (your PI).
- Form (GMU NetID login): https://qafederation.ngwebsolutions.com/sp/startSSO.ping?PartnerIdpId=https://shibboleth.gmu.edu/idp/shibboleth&TargetResource=https://dynamicforms.ngwebsolutions.com/Submit/Form/Start/fadd3769-89be-46b4-8eb3-7d13d2237c5b
- After submitting you receive a **New User Tutorial** by email; the account activates only after you complete it and file the completion form. Do the tutorial the same day so the account isn't stuck pending.
- Justification text you can paste into the form / sponsor email:

> Reinforcement-learning research for two paper submissions (ICLR 2027, ICRA 2027) from the RobotiXX lab. Workloads: (1) CPU-parallel Gazebo/ROS2 navigation-curriculum campaign, ~20 single-node jobs of 24–48 cores for up to 24 h each on the normal partition; (2) small-LLM (360M-parameter) RL fine-tuning replications, 3–6 single-GPU jobs of 1–3 h each on gpuq (A100.80gb or 3g.40gb MIG); (3) exploratory scale-up of a curriculum-selection method, up to ~200 A100-hours. Software: PyTorch/vLLM via conda, ROS 2 Humble + Gazebo via Apptainer. Storage: <200 GB scratch.

- If a question needs a human at ORC: orchelp@gmu.edu.

## ★ Step 2 — First login + key setup (you, ~5 minutes)

```bash
ssh YOUR_NETID@hopper.orc.gmu.edu       # campus network or GMU VPN
# then, from the lab machine, install a key for the agent to use:
ssh-copy-id YOUR_NETID@hopper.orc.gmu.edu
```
Open OnDemand (browser alternative, needs VPN): https://ondemand.orc.gmu.edu

Once a key works from this machine, I can drive everything else (env setup, transfers, submissions) over SSH.

## Step 3 — Environment setup on Hopper (scripted)

Run `bash hopper/setup_env.sh` on the login node (or I will, once SSH works). It creates the conda env for the Countdown/gate-replication stack and stages directories under `/scratch/$USER`.

## Step 4 — Jobs (templates in `hopper/sbatch/`)

| Script | Partition / gres | Purpose |
|---|---|---|
| `mig_smoke.sbatch` | `gpuq`, `gpu:1g.10gb:1`, 30 min | Cheapest possible end-to-end validation: env imports, CUDA visible, tiny generation. Submit this first. |
| `gate_replication_a100.sbatch` | `gpuq`, `gpu:A100.80gb:1`, 4 h | One Countdown training run (parameterized by seed) for the corrected-decay gate replication. Only after a NEW prereg for that study is frozen. |
| `barn_seed_cpu.sbatch` | `normal`, 48 cores, 24 h | One ICRA seed × all four arms inside the ROS2/Gazebo Apptainer image; array-submit for the 5-seed matrix. Requires the container image built first (instructions in the script header). |

Reference card (from Running_GPU_Jobs):

- GPU jobs: `--partition=gpuq --qos=gpu`; `contrib-gpuq` exists but is preemptable.
- gres options: `gpu:A100.80gb:N`, `gpu:A100.40gb:N`, `gpu:H100.80gb:N`, MIG `gpu:1g.10gb|2g.20gb|3g.40gb:N`.
- Recommended pairings: A100.80gb → 122 GB RAM, 16 cores; 3g.40gb → 60 GB, 8 cores; 1g.10gb → 15 GB, 2 cores.
- Full A100/H100 only via `sbatch` (no salloc/OnDemand); MIG slices allowed interactively.
- Time limits: GPU jobs 3 days default cap (gpuq listed 1-00:00 in quick start — keep GPU jobs ≤24 h to be safe), CPU 5 days, `normal` 3-00:00.
- Modules: LMOD; `ml spider cuda`, `ml cuda/12.3.1`, `module load gnu10 python`.
- Fleet: 31 nodes × 4 A100 80GB, 1 node × 4 H100, 2 DGX × 8 A100 40GB, MIG slices on 8 nodes.

## Storage notes

Home is small; use `/scratch/$USER` for runs (check quota with the storage page / `df -h` on login). Transfer via `scp`/`rsync` from the lab machine, or Globus for the 300-world BARN dataset if rsync is slow.
