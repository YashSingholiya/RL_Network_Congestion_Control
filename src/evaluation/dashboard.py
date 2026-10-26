"""Live dashboard for the congestion-control simulator (Streamlit + Plotly).

Everything shown comes straight from the environment: each step the chosen controller picks an
action, `CongestionEnv.step()` advances the simulated link, and the dashboard plots the values
returned in `info` (nothing is recalculated separately).

Run from the repo root:
    pip install -r requirements-demo.txt
    streamlit run src/evaluation/dashboard.py

Tab 1  Live run:  the episode is simulated instantly with the real environment, then replayed by a
                  smooth in-browser player (play / pause / scrub / speed) - one controller, or two
                  side by side on the SAME episode.
Tab 2  Play it yourself:  you set the sending rate step by step against a trained agent.
"""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import ACTIONS, HOLD_IDX, SCENARIOS, HELDOUT_SCENARIOS, EnvConfig
from src.envs.congestion_env import CongestionEnv
from src.envs.omnet_env import OmnetCongestionEnv, omnet_available, default_exe
from src.baselines.simple import RandomPolicy, FixedRatePolicy
from src.baselines.aimd import AIMDPolicy

DT = EnvConfig().dt
PLAYER_JS = Path(__file__).with_name("dashboard_player.js")
STRETCH_DEFAULT = "width" in inspect.signature(st.plotly_chart).parameters
BTN_KW = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}

CONTROLLERS = {
    "fixed": "Fixed rate (set to mean capacity)", "aimd": "AIMD", "random": "Random",
    "tabular_q": "Tabular Q-learning", "approx_q": "Approx. Q (linear)", "dqn": "DQN", "ppo": "PPO",
    "tabular_q_robust": "Tabular Q (noise-robust)", "approx_q_robust": "Approx. Q (noise-robust)",
    "dqn_robust": "DQN (noise-robust)", "ppo_robust": "PPO (noise-robust)",
}
SIM_LABEL = {"python": "Python fluid simulator", "omnet": "OMNeT++ packet-level"}
COLORS = {"fixed": "#1f77b4", "aimd": "#ff7f0e", "random": "#7f7f7f", "tabular_q": "#2ca02c",
          "approx_q": "#9467bd", "dqn": "#17becf", "ppo": "#d62728", "you": "#111111"}


def color_of(key):
    return COLORS.get(key.replace("_robust", ""), "#444444")


# ------------------------------------------------------------------ controllers
def model_path(key, seed):
    ext = "npy" if key.startswith(("tabular_q", "approx_q")) else "zip"
    return ROOT / "models" / f"{key}_s{seed}.{ext}"


def available_controllers(seed):
    return {k: v for k, v in CONTROLLERS.items()
            if k in ("random", "fixed", "aimd") or model_path(k, seed).exists()}


@st.cache_resource(show_spinner=False)
def _load_model(key, seed):
    path = str(model_path(key, seed))
    if key.startswith("tabular_q"):
        from src.agents.tabular_q import TabularQAgent
        return TabularQAgent.load(path)
    if key.startswith("approx_q"):
        from src.agents.approx_q import ApproxQAgent
        return ApproxQAgent.load(path)
    from src.evaluation.evaluate import SB3Policy      # imports stable-baselines3 lazily
    return SB3Policy("dqn" if key.startswith("dqn") else "ppo", path)


def make_policy(key, seed):
    if key == "random":
        return RandomPolicy(0)
    if key == "fixed":
        return FixedRatePolicy(1.0)
    if key == "aimd":
        return AIMDPolicy()
    return _load_model(key, seed)


# ------------------------------------------------------------------ simulation wrapper
class Sim:
    """One controller (or a human, policy=None) running one episode of the real environment."""

    def __init__(self, key, label, policy, cfg, scenario, seed, simulator="python"):
        self.key, self.label, self.policy, self.simulator = key, label, policy, simulator
        env_cls = OmnetCongestionEnv if simulator == "omnet" else CongestionEnv
        self.env = env_cls(cfg, scenario=scenario)
        opts = getattr(policy, "reset_options", {}) if policy is not None else {}
        self.obs, _ = self.env.reset(seed=seed, options=opts)
        self.n = cfg.episode_len
        self.rows, self.done = [], False
        cap = self.env.cap[: self.n]
        self.rate_ymax = 1.5 * float(cap.max())
        self.lat_ymax = 1000 * (self.env.base_rtt + self.env.qmax / float(cap.min())) * 1.05

    def step(self, action=None):
        if self.done:
            return
        if action is None:
            p = self.policy
            action = p.act(self.obs, self.env) if getattr(p, "uses_env", False) else p.act(self.obs)
        self.obs, r, term, trunc, info = self.env.step(int(action))
        self.rows.append(dict(
            t=len(self.rows) * DT, rate=info["rate"], capacity=info["capacity"],
            throughput=info["throughput"], latency_ms=1000 * info["latency"],
            queue_pct=100 * info["queue"] / self.env.qmax, loss_pct=100 * info["loss"],
            reward=r, arrivals=info["arrivals"], overflow=info["overflow"], rand_lost=info["rand_lost"]))
        self.done = term or trunc

    def df(self):
        return pd.DataFrame(self.rows)

    def summary(self):
        d = self.df()
        return {"Controller": self.label, "Return": d.reward.sum(),
                "Utilisation": d.throughput.sum() / d.capacity.sum(),
                "Avg latency (ms)": d.latency_ms.mean(), "p95 latency (ms)": np.percentile(d.latency_ms, 95),
                "Loss (%)": 100 * (d.overflow.sum() + d.rand_lost.sum()) / d.arrivals.sum()}


# ------------------------------------------------------------------ figures
def _layout(fig, title, T=None):
    fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.0, y=0.97), height=270,
                      margin=dict(l=55, r=55, t=75, b=45),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    fig.update_xaxes(range=[0, T], title="time (s)", automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def fig_rate(sim):
    d, T = sim.df(), sim.n * DT
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(sim.n) * DT, y=sim.env.cap[: sim.n], name="capacity (full pattern)",
                             line=dict(color="rgba(0,0,0,0.15)", dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.t, y=d.capacity, name="capacity", line=dict(color="black", dash="dash")))
    fig.add_trace(go.Scatter(x=d.t, y=d.rate, name="sending rate", line=dict(color=color_of(sim.key), width=2)))
    fig.update_yaxes(range=[0, sim.rate_ymax], title="Mbps")
    return _layout(fig, "Sending rate vs capacity", T=T)


def fig_queue(sim):
    d, T = sim.df(), sim.n * DT
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d.t, y=d.queue_pct, name="buffer fill (%)", fill="tozeroy",
                             line=dict(color="#e08a00")), secondary_y=False)
    fig.add_trace(go.Scatter(x=d.t, y=d.latency_ms, name="latency (ms)", line=dict(color="#1f77b4")), secondary_y=True)
    fig.update_yaxes(range=[0, 100], title="buffer fill (%)", secondary_y=False)
    fig.update_yaxes(range=[0, sim.lat_ymax], title="latency (ms)", secondary_y=True)
    return _layout(fig, "Buffer fill and latency", T=T)


def fig_throughput(sim):
    d, T = sim.df(), sim.n * DT
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d.t, y=d.throughput, name="throughput (Mbps)", line=dict(color="#2ca02c")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=d.t, y=d.loss_pct, name="loss (%)", line=dict(color="#d62728")), secondary_y=True)
    fig.update_yaxes(range=[0, sim.rate_ymax], title="Mbps", secondary_y=False)
    fig.update_yaxes(range=[0, max(5.0, float(d.loss_pct.max()) * 1.2 if len(d) else 5.0)], title="loss (%)",
                     secondary_y=True)
    return _layout(fig, "Throughput and loss", T=T)


def _fill_color(f):
    f = min(max(f, 0.0), 1.0)
    return f"rgb({int(255 * min(1, 2 * f))},{int(200 * min(1, 2 * (1 - f)))},60)"


def fig_network(sim):
    """Schematic: sender -> rate pipe -> router buffer -> bottleneck link -> receiver."""
    d = sim.rows[-1]
    fig = go.Figure()
    mid, hmax = 3.0, 2.6
    rect = lambda x0, x1, y0, y1, fill, line="#555", w=1: fig.add_shape(
        type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=fill, line=dict(color=line, width=w))
    note = lambda x, y, text, size=11, color="#222": fig.add_annotation(
        x=x, y=y, text=text, showarrow=False, font=dict(size=size, color=color))

    rect(0.1, 1.9, 1.6, 4.4, "#dbeafe")                                           # sender
    note(1.0, 3.0, "<b>Sender</b>")
    h = 0.15 + hmax * min(d["rate"] / sim.rate_ymax, 1.0)                           # offered rate
    rect(1.9, 4.3, mid - h / 2, mid + h / 2, "#3b82f6", "#3b82f6")
    note(3.1, 5.35, f"offered {d['rate']:.1f} Mbps", color="#1d4ed8")
    full = d["queue_pct"] > 99
    rect(4.3, 6.7, 0.5, 5.0, "#f3f4f6", "#c0262d" if full else "#555", 3 if full else 1)   # buffer tank
    rect(4.3, 6.7, 0.5, 0.5 + 4.5 * d["queue_pct"] / 100, _fill_color(d["queue_pct"] / 100), "rgba(0,0,0,0)")
    note(5.5, 2.75, f"<b>buffer {d['queue_pct']:.0f}%</b>", 12, "white" if d["queue_pct"] > 60 else "#222")
    if d["loss_pct"] > 0:
        note(5.5, 5.4, f"x dropping {d['loss_pct']:.1f}%", 11, "#c0262d")
    hc = 0.15 + hmax * min(d["capacity"] / sim.rate_ymax, 1.0)                      # bottleneck link
    rect(6.7, 9.3, mid - hc / 2, mid + hc / 2, "#d1d5db")
    ht = 0.15 + hmax * min(d["throughput"] / sim.rate_ymax, 1.0)
    rect(6.7, 9.3, mid - ht / 2, mid + ht / 2, "#16a34a", "#16a34a")                # delivered
    note(8.0, 5.35, f"link capacity {d['capacity']:.1f} Mbps", color="#374151")
    note(8.0, 0.55, f"delivered {d['throughput']:.1f} Mbps", color="#15803d")
    rect(9.3, 11.9, 1.6, 4.4, "#dcfce7")
    note(10.6, 3.0, "<b>Receiver</b><br>" + f"latency {d['latency_ms']:.0f} ms")
    fig.update_xaxes(visible=False, range=[0, 12]); fig.update_yaxes(visible=False, range=[0, 6])
    fig.update_layout(height=230, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white")
    return fig


# ------------------------------------------------------------------ rendering helpers
def show(target, fig, key):
    if STRETCH_DEFAULT:
        target.plotly_chart(fig, key=key)
    else:
        target.plotly_chart(fig, use_container_width=True, key=key)


def render_cards(target, sim, ncols=6):
    d = sim.rows[-1]
    items = [("Sending rate", f"{d['rate']:.1f} Mbps"), ("Link capacity", f"{d['capacity']:.1f} Mbps"),
             ("Buffer fill", f"{d['queue_pct']:.0f} %"), ("Latency", f"{d['latency_ms']:.0f} ms"),
             ("Throughput", f"{d['throughput']:.1f} Mbps"), ("Loss (this step)", f"{d['loss_pct']:.1f} %")]
    with target.container():
        for row in range(0, len(items), ncols):
            cols = st.columns(ncols)
            for c, (lab, val) in zip(cols, items[row: row + ncols]):
                c.metric(lab, val)


class Panel:
    """Placeholders for one controller; update() redraws them from the Sim's history."""

    def __init__(self, title, tag, ncols_cards=3, charts=("rate", "queue", "tp")):
        st.markdown(f"#### {title}")
        self.tag, self.ncols, self.charts = tag, ncols_cards, charts
        self.net, self.cards = st.empty(), st.empty()
        self.ph = {name: st.empty() for name in charts}

    def update(self, sim, i):
        if not sim.rows:
            return
        show(self.net, fig_network(sim), f"{self.tag}_net_{i}")
        render_cards(self.cards, sim, self.ncols)
        makers = {"rate": fig_rate, "queue": fig_queue, "tp": fig_throughput}
        for name in self.charts:
            show(self.ph[name], makers[name](sim), f"{self.tag}_{name}_{i}")


def action_label(m):
    return "Hold" if m == 1 else f"{(m - 1) * 100:+.0f}%"


# ------------------------------------------------------------------ tab 1: live run
def player_html(sims, autoplay, start):
    """Self-contained HTML page: the recorded episode data + the JS player (no redraws by Streamlit)."""
    r = lambda col: [round(float(v), 4) for v in col]
    panels = []
    for s in sims:
        d = s.df()
        panels.append(dict(label=s.label, color=color_of(s.key), rate_ymax=s.rate_ymax, lat_ymax=s.lat_ymax,
                           capacity=r(d.capacity), rate=r(d.rate), throughput=r(d.throughput),
                           latency_ms=r(d.latency_ms), queue_pct=r(d.queue_pct), loss_pct=r(d.loss_pct)))
    payload = json.dumps(dict(dt=DT, n=sims[0].n, autoplay=autoplay, start=start, panels=panels)).replace("</", "<\\/")
    js = PLAYER_JS.read_text(encoding="utf-8")
    return ("<!doctype html><html><head><meta charset='utf-8'></head><body><div id='app'></div>"
            f"<script>{js}</script><script>Player.init(document.getElementById('app'), {payload});</script></body></html>")


def embed_html(html, height):
    """Show a self-contained HTML page. st.iframe replaces the deprecated components.v1.html."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                            # older Streamlit versions
        from streamlit.components.v1 import html as legacy_html
        legacy_html(html, height=height, scrolling=True)


def tab_live(opts, cfg, scenario, seed, agent_seed, simulator="python"):
    c1, c2 = st.columns(2)
    keys = list(opts)
    default_a = next((k for k in ("ppo", "dqn", "tabular_q", "aimd") if k in opts), keys[0])
    a_key = c1.selectbox("Controller A", keys, index=keys.index(default_a), format_func=opts.get)
    b_choices = ["(none)"] + [k for k in keys if k != a_key]
    default_b = "aimd" if a_key != "aimd" else "fixed"
    b_key = c2.selectbox("Compare with (same episode)", b_choices,
                         index=b_choices.index(default_b) if default_b in b_choices else 0,
                         format_func=lambda k: "(none)" if k == "(none)" else opts[k])
    both = False
    if omnet_available():
        other = "omnet" if simulator == "python" else "python"
        both = st.checkbox(f"Also run Controller A in the {SIM_LABEL[other]} (same link, side by side)", value=False)
    else:
        other = "python"
    run = st.button("Run episode", type="primary")

    runs = [(a_key, simulator)] + ([] if b_key == "(none)" else [(b_key, simulator)]) + ([(a_key, other)] if both else [])
    if run:
        if any(sm == "omnet" for _, sm in runs) and not omnet_available():
            st.error(f"The OMNeT++ model is not built (expected at {default_exe()}). See omnetpp/README.md.")
            return
        if any(sm == "omnet" for _, sm in runs) and (cfg.rand_loss > 0):
            st.warning("Random loss is not modelled in OMNeT++; that setting only affects the Python simulator.")
        try:
            sims = [Sim(k, f"{opts[k]} · {SIM_LABEL[sm]}" if (both or simulator == "omnet") else opts[k],
                        make_policy(k, agent_seed), cfg, scenario, seed, simulator=sm) for k, sm in runs]
        except Exception as e:                       # e.g. PyTorch / stable-baselines3 / OMNeT++ problems
            st.error(f"Could not start the run: {e}")
            return
        for s in sims:                               # the whole episode takes well under a second
            while not s.done:
                s.step()
        st.session_state["last"] = sims
        autoplay, start = True, 0
    elif "last" in st.session_state:
        sims = st.session_state["last"]
        st.caption("Showing the last run (paused at the end). Press **Run episode** to simulate again.")
        autoplay, start = False, sims[0].n - 1
    else:
        st.info("Choose a controller and press **Run episode**. The player has play / pause, a time slider "
                "and a speed selector, so you can slow down or scrub back to any moment.")
        return
    embed_html(player_html(sims, autoplay, start), height=1100 if len(sims) > 1 else 1300)
    st.subheader("Episode summary")
    st.dataframe(pd.DataFrame([s.summary() for s in sims]).round(3), hide_index=True)


# ------------------------------------------------------------------ tab 2: play it yourself
def tab_game(opts, cfg, scenario, seed, agent_seed):
    st.caption("This tab always uses the Python simulator (each click is one step).")
    st.markdown("Set the sending rate **one 50 ms step at a time**. Keep it just below the link capacity: "
                "too high fills the buffer (latency, then loss); too low wastes capacity. "
                "The agent faces the exact same link.")
    rivals = [k for k in opts if k not in ("fixed", "random")]
    rival = st.selectbox("Opponent", rivals, index=rivals.index("ppo") if "ppo" in rivals else 0,
                         format_func=opts.get, key="rival")
    if st.button("Start new game", type="primary"):
        try:
            st.session_state["game"] = dict(
                me=Sim("you", "You", None, cfg, scenario, seed),
                ai=Sim(rival, opts[rival], make_policy(rival, agent_seed), cfg, scenario, seed))
        except Exception as e:
            st.error(f"Could not load controller: {e}")
            return
    g = st.session_state.get("game")
    if g is None:
        st.info("Press **Start new game**.")
        return
    me, ai = g["me"], g["ai"]

    def act(idx):
        if not me.done:
            me.step(idx)
            ai.step()

    def finish():
        while not me.done:
            act(HOLD_IDX)

    cols = st.columns(len(ACTIONS))
    for i, (c, m) in enumerate(zip(cols, ACTIONS)):
        c.button(action_label(m), key=f"act_{i}", on_click=act, args=(i,), disabled=me.done, **BTN_KW)
    st.button("Hold the rate until the end", on_click=finish, disabled=me.done)
    st.progress(len(me.rows) / me.n, text=f"step {len(me.rows)} / {me.n}")

    left, right = st.columns(2)
    for col, s in ((left, me), (right, ai)):
        with col:
            p = Panel(s.label, f"game_{s.key}", 3, charts=("rate", "queue"))
            p.update(s, len(s.rows))
    if me.done:
        st.subheader("Result")
        res = pd.DataFrame([me.summary(), ai.summary()]).round(3)
        st.dataframe(res, hide_index=True)
        mine, theirs = res.Return[0], res.Return[1]
        if mine > theirs:
            st.success(f"You beat {ai.label}! ({mine:.0f} vs {theirs:.0f})")
        else:
            st.info(f"{ai.label} scored {theirs:.0f} vs your {mine:.0f}.")


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="RL congestion control", layout="wide")
    st.title("RL-based network congestion control: live simulation")
    st.caption("A sender adjusts its rate every 50 ms over a bottleneck link with a finite buffer "
               "and fluctuating capacity. All values come from the project's simulator.")

    with st.sidebar:
        st.header("Network settings")
        agent_seed = st.selectbox("Trained-agent seed", [0, 1, 2])
        scenario = st.selectbox("Capacity pattern", SCENARIOS + HELDOUT_SCENARIOS,
                                index=SCENARIOS.index("markov"),
                                help="The last four patterns were never seen in training.")
        seed = st.number_input("Episode seed", 0, 10**6, 10_000, help="10000 = the episode used in the report figures")
        if omnet_available():
            simulator_name = st.radio("Simulator", list(SIM_LABEL.values()), index=0,
                                      help="OMNeT++ replays the same link with real packets and ACK-based latency. "
                                           "It must be built first (omnetpp/README.md). Random loss and latency-noise sliders "
                                           "affect only the Python simulator.")
            simulator = "omnet" if simulator_name == SIM_LABEL["omnet"] else "python"
        else:
            simulator = "python"
        match = st.checkbox("Match report episodes (training ranges)", value=True)
        if match:
            cfg = EnvConfig()
        else:
            buf = st.slider("Router buffer (bandwidth-delay products)", 0.25, 10.0, 2.0, 0.25)
            loss = st.slider("Random (non-congestive) loss", 0.0, 0.05, 0.0, 0.005)
            noise = st.slider("Latency measurement noise (fraction of RTT)", 0.0, 0.2, 0.02, 0.01)
            cfg = EnvConfig(buffer_bdp_range=(buf, buf), rand_loss=loss, lat_noise=noise)

    opts = available_controllers(agent_seed)
    tab1, tab2 = st.tabs(["Live run", "Play it yourself"])
    with tab1:
        tab_live(opts, cfg, scenario, int(seed), agent_seed, simulator)
    with tab2:
        tab_game(opts, cfg, scenario, int(seed), agent_seed)


if __name__ == "__main__":
    main()
