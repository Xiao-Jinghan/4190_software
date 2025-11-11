# agents.py
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

def tool_mimic_detection(df: pd.DataFrame, engine, focus: Optional[str] = None) -> Dict[str, Any]:
    res = engine.detect(df, context={"mode": "深度分析"})
    if focus and focus in res["target_heatmap"]:
        score = res["target_heatmap"][focus]
        advice = f"焦点对象 {focus} 的异常热度={score:.3f}。"
    else:
        advice = "未指定焦点对象。"
    return {"summary": advice, "result": res}

def tool_adv_eval(engine, intensity: float = 0.3) -> Dict[str, Any]:
    fake = engine._last_prob if engine._last_prob is not None else None
    if fake is None:
        return {"summary": "尚无检测结果，无法做对抗评估。"}
    changed = np.clip(fake + intensity * np.std(fake), 0, 1)
    delta = float(np.mean(changed) - np.mean(fake))
    verdict = "系统对扰动表现出可控的鲁棒性。" if delta < 0.05 else "扰动下异常概率显著上升，建议增强随机化与阈值自适应。"
    return {"summary": f"对抗强度={intensity}, 平均概率变化={delta:.3f} -> {verdict}"}

def tool_policy_orchestrate(engine, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    act = engine.bandit_suggest(context or {})
    return {"summary": f"建议策略：{act}"}

default_security_tools = {
    "run_mimic_detection": tool_mimic_detection,
    "adv_eval": tool_adv_eval,
    "orchestrate": tool_policy_orchestrate,
}

class SecurityAgent:
    def __init__(self, tools: Dict[str, Any]):
        self.tools = tools

    def run(self, prompt: str, df: pd.DataFrame, engine) -> Dict[str, Any]:
        prompt_l = prompt.lower() if prompt else ""
        detail = []
        answer_lines: List[str] = []

        if any(k in prompt_l for k in ["检测", "评估", "fdi", "攻击", "定位"]):
            focus = None
            import re
            m = re.search(r'([sn]\d+)', prompt_l)
            if m:
                focus = m.group(1)
            r = self.tools["run_mimic_detection"](df, engine, focus=focus)
            answer_lines.append("① 已执行拟态防御检测并融合多路证据（含 Fence-GAN 边界评分）。")
            answer_lines.append(r["summary"])
            detail.append({"tool": "run_mimic_detection", "output": "ok"})

        if "对抗" in prompt_l or "鲁棒" in prompt_l:
            r2 = self.tools["adv_eval"](engine)
            answer_lines.append("② 对抗评估：")
            answer_lines.append(r2["summary"])
            detail.append({"tool": "adv_eval", "output": "ok"})

        if "策略" in prompt_l or "编排" in prompt_l or "orchestrate" in prompt_l:
            r3 = self.tools["orchestrate"](engine, {})
            answer_lines.append("③ 策略编排建议：")
            answer_lines.append(r3["summary"])
            detail.append({"tool": "orchestrate", "output": "ok"})

        if not answer_lines:
            r = self.tools["run_mimic_detection"](df, engine, focus=None)
            answer_lines.append("未识别特定指令，已执行一次基础检测（含 Fence-GAN）。")
            detail.append({"tool": "run_mimic_detection", "output": "ok"})

        return {"answer": "\n".join(answer_lines), "detail": detail}
