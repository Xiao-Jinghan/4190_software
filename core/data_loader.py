# core/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_kdd99(path="datasets/kddcup.data_10_percent_corrected"):
    """加载 KDD99 数据集，并返回标准化后的 DataFrame"""
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
    df = pd.read_csv(path, names=cols)

    # 标签二值化
    df['label'] = df['label'].apply(lambda x: 0 if x.strip() == 'normal.' else 1)

    # 区分数值和类别列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ['protocol_type', 'service', 'flag']

    # One-hot 编码类别特征
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_enc = enc.fit_transform(df[cat_cols])
    cat_df = pd.DataFrame(cat_enc, columns=enc.get_feature_names_out(cat_cols))

    # 数值部分
    num_df = df[num_cols].drop(columns=['label'], errors='ignore')

    # 拼接
    X = pd.concat([num_df.reset_index(drop=True), cat_df], axis=1)
    y = df['label']

    # 归一化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 只保留前若干列作为展示（避免太宽）
    X_scaled['label'] = y
    return X_scaled
