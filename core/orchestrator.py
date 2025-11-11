# core/orchestrator.py
# 策略编排：简化的上下文bandit（Thompson Sampling风味）
from typing import Dict, Any
import numpy as np

class SimpleContextualBandit:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.actions = [
            {"name": "稳健模式", "th_delta": +0.05, "rand_delta": +0.2, "win_scale": 0.75},
            {"name": "均衡模式", "th_delta": +0.00, "rand_delta": +0.1, "win_scale": 1.00},
            {"name": "敏感模式", "th_delta": -0.05, "rand_delta": +0.0, "win_scale": 1.25},
        ]
        self.success = np.ones(len(self.actions))
        self.fail = np.ones(len(self.actions))

    def update(self, context: Dict[str, Any], reward_proxy: float):
        a = self.rng.integers(0, len(self.actions))
        if reward_proxy >= 0.5:
            self.success[a] += 1
        else:
            self.fail[a] += 1

    def suggest(self, context: Dict[str, Any]) -> Dict[str, Any]:
        samples = [self.rng.beta(self.success[i], self.fail[i]) for i in range(len(self.actions))]
        idx = int(np.argmax(samples))
        return self.actions[idx] | {"score": float(samples[idx])}
