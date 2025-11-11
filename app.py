import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from core.engine import MimicDefenseEngine
from agents import SecurityAgent, default_security_tools
from core.data_loader import load_kdd99, get_kdd_label_mappings

# 中文修复
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="拟态防御·Fence-GAN电力AI内生安全系统", layout="wide")

st.title("⚡ 拟态防御 · Fence-GAN 内生安全防御系统（智能电网）")

# ---------------- 状态初始化 ----------------
if "engine" not in st.session_state:
    st.session_state.engine = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cached_figs" not in st.session_state:
    st.session_state.cached_figs = {}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "agent" not in st.session_state:
    st.session_state.agent = SecurityAgent(tools=default_security_tools)

def log(msg):
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("⚙️ 参数配置")
    fg_epochs = st.slider("训练轮次", 1, 500, 10, 1)
    fg_lr = st.select_slider("学习率", options=[1e-4, 2e-4, 5e-4, 1e-3], value=1e-3)
    fg_latent = st.slider("隐变量维度", 4, 64, 32)
    fg_boundary = st.slider("边界γ", 0.3, 0.9, 0.5, 0.05)
    fg_dispersion = st.slider("分散λ", 0.0, 5.0, 1.0, 0.1)
    fg_use_first_pct = st.slider("训练比例", 10, 90, 40, 5)
    threshold = st.slider("检测阈值", 0.0, 1.0, 0.6, 0.01)

    st.divider()
    use_demo = st.checkbox("使用演示数据（KDD99）", True)
    f = st.file_uploader("上传CSV", type=["csv"])

    st.divider()
    btn_run = st.button("🚀 运行检测", use_container_width=True)
    btn_reset = st.button("♻️ 重置引擎", use_container_width=True)

if btn_reset:
    st.session_state.engine = None
    st.session_state.last_result = None
    st.session_state.cached_figs.clear()
    st.success("引擎已重置。")

# ---------------- 数据加载 ----------------
def make_demo(n=1500, m=8):
    x = np.arange(n)
    data = np.vstack([np.sin(x/30 + i) + np.random.randn(n)*0.1 for i in range(m)]).T
    return pd.DataFrame(data, columns=[f"s{i+1}" for i in range(m)])

df = None
attack_code_to_name = {}
if f is not None:
    df = pd.read_csv(f)
elif use_demo:
    try:
        df = load_kdd99()
        attack_code_to_name, _ = get_kdd_label_mappings()
        st.info("使用 KDD99 演示数据")
    except Exception:
        df = make_demo()
        st.warning("KDD99 加载失败，使用随机数据。")
else:
    st.stop()

# ---------------- 多标签页 ----------------
tabs = st.tabs(["🎯 检测", "📊 对比分析", "📈 报告", "🧾 日志"])

# ========== Tab1 检测 ==========
with tabs[0]:
    if btn_run:
        st.session_state.engine = MimicDefenseEngine(
            nodes=[f"N{i}" for i in range(10)],
            edges=[(f"N{i}", f"N{i+1}") for i in range(1,9)],
            fencegan_cfg=dict(
                latent_dim=fg_latent, boundary_gamma=fg_boundary,
                dispersion_lambda=fg_dispersion, lr=fg_lr,
                epochs=fg_epochs, use_first_pct=fg_use_first_pct
            )
        )

        # 传入多任务训练所需的监督信号（若存在）
        binary_labels = df["label"].values if "label" in df.columns else None
        attack_codes = df["attack_code"].values if "attack_code" in df.columns else None

        with st.spinner("Fence-GAN 正在训练与检测..."):
            result = st.session_state.engine.detect(
                df,
                context={
                    "threshold": threshold,
                    "binary_labels": binary_labels,
                    "attack_codes": attack_codes,
                    "attack_code_to_name": attack_code_to_name
                }
            )

        # ✅ 压缩存储结果（含多分类输出）
        res = {k: v for k, v in result.items() if isinstance(v, (int, float, dict, list)) or v is None}
        # ndarray -> list
        for key in ["anomaly_prob", "window_prob", "type_classes", "type_class_names"]:
            if key in result and result[key] is not None:
                res[key] = result[key] if isinstance(result[key], list) else np.array(result[key]).tolist()
        if result.get("type_pred_codes_ts") is not None:
            res["type_pred_codes_ts"] = np.array(result["type_pred_codes_ts"]).tolist()
        if result.get("type_prob_ts") is not None:
            res["type_prob_ts"] = np.asarray(result["type_prob_ts"]).tolist()

        st.session_state.last_result = res
        st.session_state.cached_figs.clear()
        st.success("检测完成！")

    if st.session_state.last_result:
        res = st.session_state.last_result
        st.metric("平均异常概率", f"{np.mean(res['anomaly_prob']):.3f}")
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(res["anomaly_prob"], label="异常概率")
        ax.axhline(threshold, linestyle="--", label="阈值")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

# ========== Tab2 对比分析（仅 3 项） ==========
with tabs[1]:
    st.subheader("📊 实际 vs 预测（三项：混淆矩阵 / 异常概率曲线 / 标签预测准确图）")
    if st.session_state.last_result:
        res = st.session_state.last_result
        y_pred_prob = np.array(res["anomaly_prob"])
        y_pred_label = (y_pred_prob > threshold).astype(int)

        # 1) 混淆矩阵（多分类：按 attack_code）
        if "attack_code" in df.columns and res.get("type_pred_codes_ts") is not None:
            true_codes = df["attack_code"].iloc[:len(y_pred_label)].values.astype(int)
            pred_codes = np.array(res["type_pred_codes_ts"], dtype=int)[:len(true_codes)]

            # 将代码映射为名称（若可用）
            code2name = attack_code_to_name or {}
            classes_sorted = sorted(np.unique(np.concatenate([true_codes, pred_codes])))
            labels_display = [str(code2name.get(int(c), str(int(c)))) for c in classes_sorted]

            cm = confusion_matrix(true_codes, pred_codes, labels=classes_sorted)
            st.markdown("### 🧩 多分类混淆矩阵（KDD99 原始标签）")
            st.dataframe(pd.DataFrame(cm, index=[f"真: {n}" for n in labels_display],
                                         columns=[f"预: {n}" for n in labels_display]))
        else:
            st.info("当前数据缺少多分类标签或引擎未输出类型预测，无法绘制多分类混淆矩阵。")
        if "label" in df.columns:
            st.markdown("### 🧮 二分类混淆矩阵（预测攻击/正常）")

            y_true_bin = df["label"].iloc[:len(y_pred_label)].values.astype(int)
            y_pred_bin = y_pred_label[:len(y_true_bin)].astype(int)

            cm2 = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            df_cm2 = pd.DataFrame(
                cm2,
                index=["真: 正常(0)", "真: 攻击(1)"],
                columns=["预: 正常(0)", "预: 攻击(1)"]
            )

            st.dataframe(df_cm2)

            acc = (cm2[0, 0] + cm2[1, 1]) / cm2.sum()
            st.metric("总体准确率", f"{acc:.3f}")
        # 2) 预测曲线图（保持你原有的实现：异常概率 vs 阈值；可叠加二值真值）
        st.markdown("### 📈 异常概率预测曲线")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(y_pred_prob, label="预测异常概率", linewidth=1.5)
        if "label" in df.columns:
            y_true_bin = df["label"].iloc[:len(y_pred_label)].values
            ax2.plot(y_true_bin, label="实际二值标签", alpha=0.6)
        ax2.axhline(threshold, linestyle='--', label=f"阈值 {threshold:.2f}")
        ax2.set_xlabel("时间步")
        ax2.set_ylabel("异常概率 / 标签")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

        # 3) 标签预测准确图（按每个 attack label 的逐时刻准确率）
        if "attack_code" in df.columns and res.get("type_pred_codes_ts") is not None:
            st.markdown("### 🎯 各标签预测准确率")
            true_codes = df["attack_code"].iloc[:len(y_pred_label)].values.astype(int)
            pred_codes = np.array(res["type_pred_codes_ts"], dtype=int)[:len(true_codes)]
            code2name = attack_code_to_name or {}

            acc_per_label = []
            labels_list = sorted(np.unique(true_codes))
            for c in labels_list:
                idx = (true_codes == c)
                if idx.sum() == 0:
                    acc = np.nan
                else:
                    acc = float(np.mean(pred_codes[idx] == c))
                acc_per_label.append((c, acc))

            names = [str(code2name.get(int(c), str(int(c)))) for c, _ in acc_per_label]
            vals = [a if not np.isnan(a) else 0.0 for _, a in acc_per_label]

            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.bar(np.arange(len(vals)), vals)
            ax3.set_xticks(np.arange(len(vals)))
            ax3.set_xticklabels(names, rotation=45, ha='right')
            ax3.set_ylim(0.0, 1.0)
            ax3.set_ylabel("准确率")
            st.pyplot(fig3, clear_figure=True)
        else:
            st.info("缺少多分类标签或类型预测，无法统计各标签预测准确率。")

# ========== Tab3 报告 ==========
with tabs[2]:
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.subheader("📈 Fence-GAN 检测报告")
        with st.expander("查看完整 JSON 结果", expanded=False):
            st.json(res)

        def safe_convert(obj):
            """递归地把 numpy 对象转成原生 Python 类型"""
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.generic,)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [safe_convert(v) for v in obj]
            else:
                return obj


        res_jsonable = safe_convert(st.session_state.last_result)

        st.download_button(
            "下载检测结果",
            json.dumps(res_jsonable, ensure_ascii=False, indent=2),
            file_name="result.json",
            mime="application/json"
        )
    else:
        st.info("请先运行检测。")

# ========== Tab4 日志 ==========
with tabs[3]:
    st.text("\n".join(st.session_state.logs[-200:]))
