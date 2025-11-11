# core/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ====== 全局：KDD 标签编码表 ======
# code -> name 以及 name -> code 的映射在 load_kdd99() 调用时填充
_CODE_TO_NAME = {}
_NAME_TO_CODE = {}

def get_kdd_label_mappings():
    """
    返回 (code_to_name, name_to_code) 两个字典的拷贝，用于前端/引擎显示与反解码。
    """
    return dict(_CODE_TO_NAME), dict(_NAME_TO_CODE)

def load_kdd99(path="datasets/kddcup.data_10_percent_corrected"):
    """加载 KDD99 数据集，并返回标准化后的 DataFrame（仅数值列），外加:
       - label: 二值(0=normal, 1=abnormal)
       - attack_code: 按原始文本标签编码的整数（normal 也作为一种“类型”编码）
    注意：不返回字符串列，避免下游模型在 np.float32 转换时报错。
    """
    # 41 个特征 + 标签列
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label'
    ]
    df_raw = pd.read_csv(path, names=cols)

    # 原始文本标签（保留尾部的点去掉）
    raw_labels = df_raw['label'].astype(str).str.strip()
    raw_labels = raw_labels.str.rstrip('.')

    # 建立 name -> code / code -> name 映射（含 normal）
    global _CODE_TO_NAME, _NAME_TO_CODE
    uniq_names = sorted(raw_labels.unique().tolist())
    _NAME_TO_CODE = {name: i for i, name in enumerate(uniq_names)}
    _CODE_TO_NAME = {i: name for name, i in _NAME_TO_CODE.items()}
    attack_code = raw_labels.map(_NAME_TO_CODE).astype(int)

    # 二值标签
    bin_label = (raw_labels != 'normal').astype(int)

    # 区分数值和类别列
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in df_raw.columns if c not in (cat_cols + ['label'])]

    # One-hot 编码类别特征
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_enc = enc.fit_transform(df_raw[cat_cols])
    cat_df = pd.DataFrame(cat_enc, columns=enc.get_feature_names_out(cat_cols))

    # 数值部分
    num_df = df_raw[num_cols]

    # 拼接
    X = pd.concat([num_df.reset_index(drop=True), cat_df], axis=1)

    # 归一化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 附加二值与多类编码列（数值型）
    X_scaled['label'] = bin_label.values
    X_scaled['attack_code'] = attack_code.values
    return X_scaled
