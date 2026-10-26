# RL-Based Network Congestion Control

A reinforcement-learning project that adjusts a sender's transmission rate over a changing bottleneck link. The repository contains **two simulators**:

1. **Python fluid simulator** — the main Gymnasium environment used to train and evaluate controllers quickly.
2. **OMNeT++ packet-level simulator** — a separate packet-based model used to observe network behaviour and compare the same controllers on matched scenarios.

The OMNeT++ model is an additional comparison environment; it does not replace the Python simulator. Packet-level results can differ from fluid-model results because packets are discrete, queue occupancy changes in packet-sized units, and latency feedback is measured from ACKs.

## Contents

- [Project overview](#project-overview)
- [Requirements](#requirements)
- [Quick start: Python simulator](#quick-start-python-simulator)
- [OMNeT++ setup on Windows using WSL](#omnet-setup-on-windows-using-wsl)
- [Build the packet-level model](#build-the-packet-level-model)
- [Run and inspect an OMNeT++ episode](#run-and-inspect-an-omnet-episode)
- [Compare the simulators](#compare-the-simulators)
- [Dashboard](#dashboard)
- [Training and evaluation commands](#training-and-evaluation-commands)
- [How the RL environment works](#how-the-rl-environment-works)
- [Repository layout](#repository-layout)
- [Known differences and interpretation](#known-differences-and-interpretation)

## Project overview

The sender chooses a rate adjustment every 50 ms. The objective is to use the available link capacity while controlling queueing delay, packet loss, and unnecessary rate changes.

The Python environment implements the Gymnasium `reset()` / `step()` interface. The OMNeT++ environment exposes a compatible interface through `src/envs/omnet_env.py`; a local TCP connection exchanges episode configuration, rate decisions, and measurements between Python and the OMNeT++ process.

### Controllers included

- Random
- Fixed rate
- AIMD
- Tabular Q-learning
- Linear approximate Q-learning
- DQN
- PPO
- Noise-robust trained variants of the learned controllers, where model files are present

The `models/` directory contains saved trained agents. You can evaluate these without retraining them.

### Existing Python-simulator results

The following results are recorded in the repository's root README and correspond to the Python fluid-model evaluation (5 capacity patterns × 20 episodes × 3 seeds):

| Method | Return | Utilisation | p95 latency (ms) | Loss |
|---|---:|---:|---:|---:|
| Random | 16.8 | 0.083 | 136 | 2.5% |
| Fixed rate | 166.4 | 0.937 | 369 | 5.9% |
| AIMD | 256.2 | 0.732 | 305 | 0.1% |
| Tabular Q | 363.1 | 0.976 | 130 | 0.0% |
| Approx. Q (linear) | 329.4 | 0.898 | 146 | 0.0% |
| DQN | 372.1 | 0.994 | 121 | 0.0% |
| PPO | 373.1 | 0.988 | 115 | 0.0% |

These are existing project results, not a claim that the OMNeT++ parity check has passed on every machine. Generalisation results are in `results/generalization_summary.csv`; Python evaluation figures are in `results/figures/`.

## Requirements

### Python

- Python 3.10+ is recommended.
- Dependencies are listed in `requirements.txt`: NumPy, pandas, Matplotlib, Gymnasium, Stable-Baselines3, PyTorch, pytest, Streamlit, and Plotly.

### OMNeT++ packet-level model

- OMNeT++ 6.x with a working C++ build toolchain.
- On Windows, this project was set up to use the `opp_env` WSL environment. The current working installation described for this project is **OMNeT++ 6.4.0 with INET 4.7.0**.
- WSLg or another working Linux GUI configuration is required to open Qtenv graphical windows from WSL.
- The custom model in `omnetpp/` defines its own sender, bottleneck queue, receiver, and bridge. INET is installed in the described environment, but these custom NED/C++ modules do not import INET modules directly.

## Quick start: Python simulator

Run these commands from the repository root.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

If PowerShell blocks activation, use Command Prompt and activate with `.venv\Scripts\activate.bat`, or follow your organisation's Python environment policy.

### Linux / WSL / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

The test suite includes Python environment tests and OMNeT++ bridge tests. The real compiled-OMNeT++ test is skipped if the `omnetpp/congestion` executable has not been built. A successful test run is a useful check of the Python installation, but does not by itself prove that the real OMNeT++ executable has been compiled or that the graphical interface works.

## OMNeT++ setup on Windows using WSL

> **If OMNeT++ 6.4.0 and INET 4.7.0 are already installed and built, skip the installation commands.** The steps below show how to enter the existing environment and connect it to this repository. Do not use `--install` each time you open a new terminal.

### 1. Open the WSL distribution

From **Windows PowerShell** (not from inside the Linux shell):

```powershell
wsl -l -v
wsl -d opp_env
```

If your distribution has a different name, replace `opp_env` with the name shown by `wsl -l -v`.

### 2. Enter the existing OMNeT++ environment

Inside the `opp_env` Linux terminal:

```bash
cd ~/workspace
opp_env shell omnetpp-6.4.0 inet-4.7.0
```

This selects the two existing projects in the workspace. Use the versions actually installed in your workspace if they differ. The project names are separate (`omnetpp` and `inet`); do **not** combine them into a single version name such as `omnetpp-6.4.0+inet-4.7.0`.

Verify the environment:

```bash
opp_run --version
which opp_makemake
printf 'OMNETPP_ROOT=%s\nINET_ROOT=%s\n' "$OMNETPP_ROOT" "$INET_ROOT"
```

The command-line environment and the graphical IDE are separate concerns.

### 3. Place the repository in the WSL Linux filesystem

For the TCP bridge to work reliably, run Python and the OMNeT++ executable in the same WSL environment. Keep the active copy of the project under your Linux home directory (for example, `~/RL_Network_Congestion_Control`), not under `/mnt/c` if you can avoid it.

If your repository is in Windows, copy it into WSL. Replace the Windows path below with the real path to your project:

```bash
cp -r "/mnt/c/Users/<YOUR_WINDOWS_USER>/path/to/RL_Network_Congestion_Control" ~/RL_Network_Congestion_Control
cd ~/RL_Network_Congestion_Control
```

If the repository is already in WSL, simply `cd` to that directory. If you update the Windows copy later, copy or sync the changes to the WSL copy before running them.

### 4. Create the Python environment in WSL

Do this once per project environment:

```bash
cd ~/RL_Network_Congestion_Control
python3 -m venv ~/venv
source ~/venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3-venv` is missing, install the corresponding package for your WSL distribution before creating the virtual environment. PyTorch can take some time to download. A CPU-only PyTorch build is sufficient for inference and the basic simulator workflows.

For each new terminal session, enter the existing OMNeT++ shell and activate the Python environment again:

```bash
cd ~/workspace
opp_env shell omnetpp-6.4.0 inet-4.7.0
source ~/venv/bin/activate
cd ~/RL_Network_Congestion_Control
```

Do not run `opp_env install` or use `opp_env shell --install ...` on every startup. Those are installation operations, not normal launch steps.

## Build the packet-level model

Run inside the OMNeT++ environment, from the repository's `omnetpp/` directory:

```bash
cd ~/RL_Network_Congestion_Control/omnetpp
opp_makemake -f --deep -o congestion
make MODE=release -j4
```

Confirm that the executable was created:

```bash
ls -lh congestion
```

The Python bridge looks for this executable at `omnetpp/congestion`. If you build it in a different location, set `OMNETPP_CONGESTION_EXE` to the full path of the executable before running the Python integration.

Rebuild after changing C++ or NED model files. You do not need to rebuild just because you reopen a terminal or rerun an episode.

### Check both simulators

From the repository root, with the Python virtual environment activated and the OMNeT++ environment active:

```bash
python -m pytest -q
python scripts/omnet_parity_check.py
```

The parity script runs fixed-rate and AIMD controllers across matched seeds and compares utilisation, latency, and queue traces. It prints `PARITY OK` if the measured values are within the configured tolerances; otherwise it prints the cases outside tolerance. Packet-level and fluid results are not expected to be numerically identical in every metric.

## Run and inspect an OMNeT++ episode

### Run in the terminal

```bash
python scripts/omnet_watch.py --controller aimd --pattern markov
```

### Open the graphical network window (Qtenv)

```bash
python scripts/omnet_watch.py --controller aimd --pattern markov --gui
```

The OMNeT++ window opens and waits for you to click **Run** or **Fast**. The default pacing is approximately real time (one decision per 50 ms of wall-clock time). Close the window when finished. This requires a working WSL graphical display (WSLg).

Supported controller names include `fixed`, `aimd`, `random`, `tabular_q`, `approx_q`, `dqn`, and `ppo`. Add `_robust` to select a robust trained variant when its saved model is available, for example `dqn_robust`.

Choose a scenario using `--pattern`: training scenarios include `constant`, `step`, `sine`, `markov`, and `trace`; held-out scenarios include `sawtooth`, `square`, `outage`, and `ramp`.

### Record data for OMNeT++ analysis tools

```bash
python scripts/omnet_watch.py --controller tabular_q --pattern markov --record --steps 80
```

Recorded `.vec`, `.sca`, and `.elog` files are written under `omnetpp/results/`. The event log can become large for a full 400-step episode, so `--steps 80` is a convenient short inspection run. To list recorded vectors and export them to CSV:

```bash
cd omnetpp/results
opp_scavetool query -l "Record-#0.vec"
opp_scavetool export -F CSV-R -o vectors.csv "Record-#0.vec"
```

Open the recorded files in the OMNeT++ IDE to use the Analysis Tool for result vectors and the Sequence Chart for event logs. The exact menu wording can vary by OMNeT++ version.

## Compare the simulators

The comparison runs a controller against matched episode seeds in both the Python fluid environment and the OMNeT++ packet-level environment.

```bash
# Run a small comparison first
python -m src.evaluation.evaluate_omnet --episodes 3 --seeds 0 1 2 --methods fixed aimd tabular_q approx_q dqn ppo

# Plot the comparison
python -m src.evaluation.plot_omnet --pattern markov
```

By default, comparison data are saved under `results/omnet/`, with figures under `results/omnet/figures/`. Use `--help` on either command to see available options. If you want the agent to use queue-based latency feedback (closer to the fluid model's immediate queue estimate) rather than ACK-measured latency, the OMNeT environment supports the `--latency queue` option on the episode watcher; this is an ablation, not the default packet-level measurement.

## Dashboard

The dashboard can display the Python simulator and, when the OMNeT++ executable is built and accessible, the OMNeT++ packet-level simulator.

Start it from the repository root with the Python virtual environment activated:

```bash
streamlit run src/evaluation/dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in a browser. Select **OMNeT++ packet-level** in the dashboard and use the comparison option if you want to run the other simulator alongside it. For OMNeT++ integration from the dashboard, run Streamlit in the same WSL environment where the executable and Python dependencies are available.

## Training and evaluation commands

Run commands from the repository root with the Python environment activated.

```bash
# Train tabular and approximate Q-learning agents
python -m src.agents.tabular_q --episodes 3000 --seed 0
python -m src.agents.approx_q --episodes 3000 --seed 0

# Train DQN and PPO
python -m src.agents.train_sb3 --algo dqn --timesteps 300000 --seed 0
python -m src.agents.train_sb3 --algo ppo --timesteps 300000 --seed 0

# Tune AIMD on held-out seeds
python -m scripts.tune_aimd

# Evaluate controllers and generalisation
python -m src.evaluation.evaluate --episodes 20 --seeds 0 1 2
python -m src.evaluation.evaluate --episodes 20 --seeds 0 1 2 --rand-loss 0.01
python -m src.evaluation.generalization --episodes 10 --seeds 0 1 2
python -m src.evaluation.plots
```

Training is optional if you only want to reproduce evaluations using the saved models in `models/`. Robust variants require their corresponding saved files; add `--robust` to the training command where supported to train noise-randomised agents.

The repository also contains `scripts/run_all.sh` and `scripts/run_all.ps1` for batch workflows. Review the script before running it because it can launch lengthy training and evaluation jobs.

## How the RL environment works

### Python environment defaults

| Parameter | Default | Meaning |
|---|---:|---|
| Decision interval | 0.05 s | Agent decision/monitoring interval |
| Episode length | 400 steps | 20 seconds of simulated time |
| History | 8 intervals | Five features per interval; 40-value observation |
| Base RTT | 40–150 ms | Drawn at reset |
| Peak capacity | 8–20 Mbps | Capacity varies by scenario |
| Buffer | 1–4 bandwidth-delay products | Queue size relative to the bandwidth-delay product |
| Rate limit | 0.1 Mbps to 1.5 × peak capacity | Sender rate clipping |

### Observation and actions

The observation contains eight historical samples of five features (40 values total): latency excess, latency gradient, sending-rate ratio, loss rate, and normalised throughput.

The seven discrete actions multiply the current rate by one of:

`0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5`

### Reward

The reward balances throughput utilisation against queueing delay, loss, and rate changes:

```text
reward = utilisation
         - 0.3 * (queueing_delay / base_RTT)
         - 3.0 * loss_rate
         - 0.1 * abs(log(rate_multiplier))
```

### Capacity patterns

- `constant`: stable capacity.
- `step`: abrupt changes between capacity levels.
- `sine`: smooth periodic changes.
- `markov`: random jumps between capacity levels.
- `trace`: noisy, mean-reverting capacity with occasional deep fades.
- `sawtooth`, `square`, `outage`, `ramp`: held-out patterns for generalisation evaluation.

## Repository layout

```text
src/config.py                    Environment, action, and reward configuration
src/envs/congestion_env.py       Main Python fluid simulator
src/envs/capacity.py             Capacity pattern generation
src/envs/omnet_env.py             Gymnasium-style OMNeT++ wrapper and TCP bridge
src/baselines/                   Random, fixed-rate, and AIMD policies
src/agents/                      Tabular Q, approximate Q, DQN, PPO training
src/evaluation/                  Evaluation, generalisation, plots, dashboard, OMNeT comparison
omnetpp/CongestionNet.ned         Packet-level network topology
omnetpp/omnetpp.ini               OMNeT++ run configurations and recording settings
omnetpp/src/                      C++ implementations of sender, queue, receiver, bridge
scripts/                          OMNeT watcher/parity checks, AIMD tuning, batch scripts
tests/                            Python and OMNeT bridge tests
models/                           Saved trained policies
results/                          Existing metrics, traces, logs, and figures
report/                            Project report source and PDF, if included in your checkout
```

## Known differences and interpretation

The OMNeT++ model uses discrete 1500-byte packets, a finite drop-tail queue, and ACK-based latency measurements. The Python model uses fluid volumes and includes configurable latency noise and non-congestive random loss. These differences are intentional and can make an agent trained only in the Python simulator behave differently in OMNeT++.

Use the Python simulator for fast training and broad statistical evaluation. Use OMNeT++ to inspect packet-level effects, run matched comparisons, and study how queueing, drops, and ACK feedback affect controller behaviour.
