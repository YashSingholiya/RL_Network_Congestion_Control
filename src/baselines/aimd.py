"""AIMD (additive-increase / multiplicative-decrease) rate controller.

Classic TCP-Reno-style logic expressed in rate form:
    no loss  ->  rate += inc          (additive increase, Mbps per step)
    loss     ->  rate *= beta         (multiplicative decrease), then hold for `cooldown` steps

The environment only accepts discrete multiplicative actions, so the controller keeps an
internal *target* rate and each step picks the action whose multiplier brings the real
rate closest to that target (the leftover error is carried to the next step, so small
additive increases still accumulate). The controller only uses what a real sender knows:
its own rate and the observed loss.
"""
import numpy as np
from src.config import ACTIONS

_LOG_ACTIONS = np.log(ACTIONS)


class AIMDPolicy:
    uses_env = True          # needs the sender's own current rate (env.rate)
    reset_options = {}       # same random initial rate as the RL agents

    def __init__(self, inc=0.02, beta=0.5, cooldown=3, loss_thresh=0.0):
        self.inc, self.beta, self.cooldown, self.loss_thresh = inc, beta, cooldown, loss_thresh
        self.target, self.cool = 0.0, 0

    def act(self, obs, env):
        if env.t == 0:                              # new episode -> reset internal state
            self.target, self.cool = env.rate, 0
        loss = float(obs[-2])                       # loss-rate feature of the latest MI
        if loss > self.loss_thresh and self.cool == 0:
            self.target = env.rate * self.beta      # multiplicative decrease
            self.cool = self.cooldown
        elif self.cool > 0:
            self.cool -= 1                          # hold while the queue drains
        else:
            self.target += self.inc                 # additive increase
        self.target = float(np.clip(self.target, env.cfg.min_rate, env.max_rate))
        ratio = max(self.target / env.rate, 1e-6)
        return int(np.argmin(np.abs(_LOG_ACTIONS - np.log(ratio))))
