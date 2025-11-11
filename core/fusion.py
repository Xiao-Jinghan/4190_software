# core/fusion.py
# 证据融合：把多视角×多检测器分数融合为异常概率、类型后验、目标热力与一致性
from typing import List, Dict, Any
import numpy as np
import networkx as nx

ATTACK_TYPES = ["spike", "drift", "oscillation", "step_change", "fdi_like", "boundary_anom"]

class EvidenceFusion:
    def __init__(self):
        pass

    def combine(self, view_results: List, columns: List[str], graph: nx.Graph) -> Dict[str, Any]:
        stacks = []
        names = []
        for vname, dets in view_results:
            for dname, s in dets:
                if s is None or len(s)==0: continue
                stacks.append(np.asarray(s))
                names.append(f"{vname}:{dname}")
        if not stacks:
            return {"anomaly_prob": [], "type_probs": {a:0.0 for a in ATTACK_TYPES},
                    "target_heatmap": {}, "consistency": 0.0}

        # 对齐长度
        minT = min(len(s) for s in stacks)
        stacks = [s[:minT] for s in stacks]
        S = np.vstack(stacks)  # (K, T)
        K, T = S.shape

        # 概率：均值
        prob = np.clip(np.mean(S, axis=0), 0, 1)

        # 类型后验：按检测器映射
        type_scores = {t:0.0 for t in ATTACK_TYPES}
        for i, nm in enumerate(names):
            w = float(np.mean(S[i]))
            if "zscore" in nm:
                type_scores["spike"] += w
            elif "rollvar" in nm:
                type_scores["oscillation"] += w
            elif "fencegan" in nm:
                type_scores["boundary_anom"] += 1.5 * w  # 提高权重：边界异常更可信
        ssum = sum(type_scores.values()) + 1e-8
        type_probs = {k: float(v/ssum) for k, v in type_scores.items()}

        # 目标定位热力（简化版：用节点度+微扰）
        columns = list(columns)
        deg = {n: graph.degree(n) if n in graph else 1 for n in columns}
        base = np.array([deg.get(c,1) for c in columns], dtype=float)
        base = (base - base.min()) / (base.max() - base.min() + 1e-8) + 0.5
        rng = np.random.default_rng(123)
        noise = rng.normal(0, 0.03, size=len(columns))
        heat = base + noise
        heat = np.clip(heat, 0, None)
        heat = heat / (heat.sum() + 1e-8)
        target_heat = {c: float(h) for c, h in zip(columns, heat)}

        var = float(np.var(S, axis=0).mean())
        consistency = float(1.0 / (1.0 + 10.0*var))
        return {
            "anomaly_prob": prob.tolist(),
            "type_probs": type_probs,
            "target_heatmap": target_heat,
            "consistency": consistency
        }
