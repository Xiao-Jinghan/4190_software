import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

from core.engine import MimicDefenseEngine
from agents import SecurityAgent, default_security_tools
from core.data_loader import load_kdd99

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

if f is not None:
    df = pd.read_csv(f)
elif use_demo:
    try:
        df = load_kdd99()
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
        with st.spinner("Fence-GAN 正在训练与检测..."):
            result = st.session_state.engine.detect(df, context={"threshold": threshold})
        # ✅ 压缩存储结果
        res = {k: v for k, v in result.items() if isinstance(v, (int, float, dict, list))}
        res["anomaly_prob"] = np.array(result["anomaly_prob"]).tolist()
        st.session_state.last_result = res
        st.session_state.cached_figs.clear()
        st.success("检测完成！")

    if st.session_state.last_result:
        res = st.session_state.last_result
        st.metric("平均异常概率", f"{np.mean(res['anomaly_prob']):.3f}")
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(res["anomaly_prob"], label="异常概率")
        ax.axhline(threshold, color='r', linestyle="--", label="阈值")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

# ========== Tab2 实际 vs 预测 ==========
with tabs[1]:
    st.subheader("📊 实际 vs 预测 对比")
    if st.session_state.last_result:
        res = st.session_state.last_result
        y_pred = np.array(res["anomaly_prob"])
        y_pred_label = (y_pred > threshold).astype(int)

        if "label" in df.columns:
            y_true = df["label"].iloc[:len(y_pred)].values

            # ===============================================

            # ROC-AUC 与混淆矩阵
            auc = roc_auc_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred_label, labels=[0, 1])
            labels_display = ["实际正常", "实际攻击"]
            columns_display = ["预测正常", "预测攻击"]
            cm_shape = cm.shape[0]
            labels_display = labels_display[:cm_shape]
            columns_display = columns_display[:cm_shape]

            st.write(f"ROC-AUC = **{auc:.4f}**")
            st.dataframe(pd.DataFrame(cm, index=labels_display, columns=columns_display))

            # =============== 绘制对比图 ==================
            st.markdown("### 📈 实际 vs 预测 异常曲线")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(y_pred, label="预测异常概率", linewidth=1.5)
            ax.plot(y_true, label="实际标签", alpha=0.6)
            ax.axhline(threshold, color='r', linestyle='--', label=f"阈值 {threshold:.2f}")
            ax.set_xlabel("时间步")
            ax.set_ylabel("异常概率 / 标签")
            ax.legend()
            st.pyplot(fig, clear_figure=True)
            # ==============================================

        else:
            st.info("当前数据集无 'label' 列，无法绘制实际对比图。")

# ========== Tab3 报告 ==========
with tabs[2]:
    if st.session_state.last_result:
        res = st.session_state.last_result
        st.subheader("📈 Fence-GAN 检测报告")
        with st.expander("查看完整 JSON 结果", expanded=False):
            st.json(res)
        st.download_button("下载检测结果", json.dumps(res, ensure_ascii=False, indent=2),
                           file_name="result.json", mime="application/json")
    else:
        st.info("请先运行检测。")

# ========== Tab4 日志 ==========
with tabs[3]:
    st.text("\n".join(st.session_state.logs[-200:]))
