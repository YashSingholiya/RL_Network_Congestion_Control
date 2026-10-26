# OMNeT++ packet-level model (optional extension)

A small OMNeT++ model of the same single-bottleneck network as the Python simulator. **The Python simulator stays the
main simulator**; this folder is additive and the two run on exactly the same episodes.

```
RateSender --> BottleneckQueue --> Receiver          (ACKs return to the sender)
   ^                                                 AgentBridge: every 50 ms it sends statistics to Python
   +---- rate set by Python each 50 ms ----------------- and receives the new sending rate (TCP, text lines)
```

## Every time you start a new session (the short version)
```powershell
wsl -d opp_env                                   # PowerShell: open the OMNeT++ WSL (name from `wsl -l -v`)
```
```bash
cd ~/workspace && opp_env shell omnetpp-6.2.0    # enter the OMNeT++ environment (use the version you installed)
source ~/venv/bin/activate                       # the Python environment
cd ~/RL_Network_Congestion_Control               # the project
```

## 1. One-time setup (from the beginning)

### 1.1 Open the WSL image you installed
```powershell
wsl --version                    # needs WSL 2.5.7 or newer; update with:  wsl --update
wsl -l -v                        # lists your distributions; note the OMNeT++ one (usually "opp_env")
wsl -d opp_env                   # open it (replace the name if yours differs)
```
The first time, the image may show on-screen instructions to install OMNeT++: follow them.

### 1.2 Check or install OMNeT++ (inside the WSL window)
```bash
opp_run -v 2>/dev/null | head -2   # if this prints a version, OMNeT++ is already available: skip to 1.3
opp_env --version
opp_env list omnetpp | head        # shows the available OMNeT++ versions
mkdir -p ~/workspace && cd ~/workspace && opp_env init        # first time only
opp_env install omnetpp-6.2.0      # any 6.x version from the list works (tested with 6.2.0)
opp_env shell omnetpp-6.2.0        # enter the environment; everything below runs inside it
opp_run -v | head -2 ; which opp_makemake                     # both must work now
```

### 1.3 Put the project inside WSL
Run everything in the Linux file system (fast, and the bridge uses 127.0.0.1). If your repository is on Windows:
```bash
cp -r "/mnt/c/Users/<you>/OneDrive/Documents/GitHub/RL_Network_Congestion_Control" ~/RL_Network_Congestion_Control
cd ~/RL_Network_Congestion_Control
# Windows line endings (CRLF) can upset Linux tools: remove them from the OMNeT++ files
find omnetpp -type f \( -name '*.ned' -o -name '*.ini' -o -name '*.cc' -o -name '*.h' \) -exec sed -i 's/\r$//' {} +
```
Keep committing from Windows/VS Code; the WSL copy is for running. After changing files on Windows, copy them again.

### 1.4 Python environment (inside WSL)
```bash
python3 --version || sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, gymnasium, stable-baselines3, torch ... (large download)
pip install -r requirements-demo.txt     # only for the dashboard
```
(Smaller PyTorch if you only need CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.)

### 1.5 Build the OMNeT++ model
```bash
cd ~/RL_Network_Congestion_Control/omnetpp
opp_makemake -f --deep -o congestion
make MODE=release -j4
ls -l congestion                          # the executable must exist
cd ..
```
Rebuild with `make MODE=release` whenever a file in `omnetpp/src` changes.

### 1.6 Check that everything works
```bash
python -m pytest -q                       # expect: 11 passed
python scripts/omnet_parity_check.py      # expect: PARITY OK
```

## 2. See the simulation

### 2.1 In the OMNeT++ window (Qtenv)
```bash
python scripts/omnet_watch.py --controller aimd --pattern markov --gui
```
A window with the network opens (needs WSLg, i.e. Windows 11 or a recent Windows 10 with WSL 2). Press **Run** or
**Fast** in its toolbar. Each module shows live text: sender rate, queue fill and link capacity (the queue icon turns
green / yellow / red), delivered data, and the step counter. Double-click a module to inspect its variables.
Python answers every 50 ms of simulated time; close the window to stop. Other controllers:
`--controller fixed | random | tabular_q | approx_q | dqn | ppo` (add `_robust` for the noise-robust agents).

### 2.2 Record the run and analyse it with OMNeT++'s tools
```bash
python scripts/omnet_watch.py --controller tabular_q --pattern markov --record --steps 80
ls omnetpp/results                        # Record-#0.vec (6 result vectors), .sca, .elog (event log)
cd omnetpp/results
opp_scavetool query -l "Record-#0.vec"    # list the recorded vectors (sending rate, throughput, queue, latency, loss, capacity)
opp_scavetool export -F CSV-R -o vectors.csv "Record-#0.vec"     # all vectors as CSV
```
Open the files in the OMNeT++ IDE (start it with `omnetpp`; File > Open File): the `.vec` file opens the Analysis tool
for plots, the `.elog` file opens the Sequence Chart that shows the packets moving between modules.
Use `--steps 80` (4 s) for the event log; a full 400-step episode makes a much larger file.

### 2.3 Compare the two simulators (figures)
```bash
python -m src.evaluation.evaluate_omnet --episodes 3 --seeds 0 1 2 --methods fixed aimd tabular_q approx_q dqn ppo
python -m src.evaluation.plot_omnet --pattern markov
explorer.exe results/omnet/figures        # opens the folder in Windows Explorer
```
Figures: `python_vs_omnet_timeseries_markov.png` (same controller, same link, both simulators) and `python_vs_omnet_summary.png`.
Optional diagnosis: add `--latency queue` to give the agent queue-based latency (as in the fluid model) instead of ACK-measured latency.

### 2.4 In the dashboard
```bash
streamlit run src/evaluation/dashboard.py
```
Open http://localhost:8501 in your Windows browser. In the sidebar choose **OMNeT++ packet-level**; tick
"Also run Controller A in the other simulator" to see both side by side.

## Files
| File | Role |
|---|---|
| `CongestionNet.ned`, `omnetpp.ini` | network definition, statistics, run configurations (`Bridge`, `Record`) |
| `src/RateSender.*` | paced packet source; measures RTT from ACKs |
| `src/BottleneckQueue.*` | finite drop-tail buffer + bottleneck with time-varying capacity |
| `src/Receiver.*` | sink that returns ACKs |
| `src/AgentBridge.*` | lock-step TCP bridge to Python (protocol in the header comment) |
| `../src/envs/omnet_env.py` | `OmnetCongestionEnv`: same interface as `CongestionEnv` |
| `../scripts/omnet_watch.py` | run one episode with the OMNeT++ window or recording |
| `../scripts/omnet_parity_check.py` | Fixed rate and AIMD in both simulators |

## Troubleshooting
| Message / symptom | Fix |
|---|---|
| `opp_makemake: command not found` | You are not inside `opp_env shell omnetpp-<version>` |
| `OMNeT++ model not found at .../congestion` | Build it (step 1.5) or `export OMNETPP_CONGESTION_EXE=/full/path/to/congestion` |
| `No module named 'gymnasium'` (or numpy ...) | Activate the venv: `source ~/venv/bin/activate` |
| `Timed out waiting for OMNeT++` / `did not connect` | The message includes the OMNeT++ log; most often the model was built in another environment or the NED files were not found |
| No window with `--gui` | Run `wsl --update`, restart WSL (`wsl --shutdown`), check `echo $DISPLAY` is not empty |
| Very slow build | The project is on `/mnt/c`: copy it to `~` (step 1.3) |

## Known differences from the Python fluid model
1500-byte packets (about 42 per 50 ms at 10 Mbps, 4 at 1 Mbps), latency measured from ACKs (it lags the queue by about
one RTT), buffer of at least 2 packets, no random loss and no latency noise. Agents trained only in the fluid simulator can
therefore behave differently here; the `--latency queue` option shows how much of that is the delayed latency feedback.
