import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, IO

# ====== 全局：KDD 标签编码表 ======
# code -> name 以及 name -> code 的映射在 load_kdd99() / KDD99Loader 调用时填充
_CODE_TO_NAME: Dict[int, str] = {}
_NAME_TO_CODE: Dict[str, int] = {}


def get_kdd_label_mappings() -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    返回 (code_to_name, name_to_code) 两个字典的拷贝，用于前端/引擎显示与反解码。
    """
    return dict(_CODE_TO_NAME), dict(_NAME_TO_CODE)


# =====================================================================
# 抽象基类：统一数据加载接口，便于扩展不同数据集（KDD99 / CICIDS / 自定义CSV 等）
# =====================================================================
class BaseLoader(ABC):
    """
    统一的数据加载接口。

    要求 load() 返回的 DataFrame：
      - 主体为浮点型特征列（已做适当预处理/标准化）
      - 若存在，'label' 列为 0/1 二值（正常/异常）
      - 若存在，'attack_code' 为整型多类编码（0..K-1）
    """

    name: str = "base"

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """返回标准化后的 DataFrame"""
        raise NotImplementedError

    @property
    def label_mappings(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """
        默认返回全局 KDD 编码映射；
        对于非 KDD 数据集通常可以忽略。
        """
        return get_kdd_label_mappings()


# =====================================================================
# KDD99 Loader：兼容原始 load_kdd99() 行为
# =====================================================================
class KDD99Loader(BaseLoader):
    name: str = "KDD99"

    def __init__(self, path: str = "datasets/kddcup.data_10_percent_corrected"):
        self.path = path

    def load(self) -> pd.DataFrame:
        """加载 KDD99 数据集，并返回标准化后的 DataFrame（仅数值列），外加:
           - label: 二值(0=normal, 1=abnormal)
           - attack_code: 按原始文本标签编码的整数（normal 也作为一种“类型”编码）
        注意：不返回字符串列，避免下游模型在 np.float32 转换时报错。
        """
        # 41 个特征 + 标签列
        cols = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "label",
        ]
        df_raw = pd.read_csv(self.path, names=cols)

        # 原始文本标签（保留尾部的点去掉）
        raw_labels = df_raw["label"].astype(str).str.strip()
        raw_labels = raw_labels.str.rstrip(".")

        # 建立 name -> code / code -> name 映射（含 normal）
        global _CODE_TO_NAME, _NAME_TO_CODE
        uniq_names = sorted(raw_labels.unique().tolist())
        _NAME_TO_CODE = {name: i for i, name in enumerate(uniq_names)}
        _CODE_TO_NAME = {i: name for name, i in _NAME_TO_CODE.items()}
        attack_code = raw_labels.map(_NAME_TO_CODE).astype(int)

        # 二值标签
        bin_label = (raw_labels != "normal").astype(int)

        # 区分数值和类别列
        cat_cols = ["protocol_type", "service", "flag"]
        num_cols = [c for c in df_raw.columns if c not in (cat_cols + ["label"])]

        # One-hot 编码类别特征
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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
        X_scaled["label"] = bin_label.values
        X_scaled["attack_code"] = attack_code.values
        return X_scaled


# 对外兼容：保留原 load_kdd99() 接口
def load_kdd99(path: str = "datasets/kddcup.data_10_percent_corrected") -> pd.DataFrame:
    """
    保持原有函数签名不变，内部委托给 KDD99Loader。
    """
    loader = KDD99Loader(path=path)
    return loader.load()

# =====================================================================
# 通用 CSV Loader：用于任意新数据集（数值特征 + 可选 label / attack_code）
# =====================================================================
class GenericCSVLoader(BaseLoader):
    """
    通用 CSV 加载器：

    - 自动选择所有数值型列作为特征
    - 若存在 'label' / 'attack_code' 列，则保留为整数标签，不参与标准化
    - 其余非数值列会被丢弃（避免影响模型训练）
    """

    name: str = "GenericCSV"

    def __init__(
        self,
        file: Optional[Union[str, IO]] = None,
        df: Optional[pd.DataFrame] = None,
        standardize: bool = True,
    ):
        """
        参数：
          - file: CSV 路径或类文件对象（如 Streamlit file_uploader 返回值）
          - df: 已经在上游读取好的 DataFrame（file/df 至少提供一个）
          - standardize: 是否对数值特征做 StandardScaler 标准化
        """
        self._file = file
        self._df = df
        self.standardize = standardize

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            df_raw = self._df.copy()
        elif self._file is not None:
            df_raw = pd.read_csv(self._file)
        else:
            raise ValueError("GenericCSVLoader 需要提供 file 或 df 之一。")

        if df_raw.empty:
            raise ValueError("CSV 内容为空。")

        # label / attack_code 单独拿出来，避免被标准化
        label_series = df_raw["label"] if "label" in df_raw.columns else None
        attack_series = (
            df_raw["attack_code"] if "attack_code" in df_raw.columns else None
        )

        drop_cols = []
        if label_series is not None:
            drop_cols.append("label")
        if attack_series is not None:
            drop_cols.append("attack_code")

        feature_df = df_raw.drop(columns=drop_cols, errors="ignore")

        # 仅保留数值型特征列
        num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("CSV 中未检测到数值型特征列，无法进行异常检测。")

        feature_df = feature_df[num_cols].copy()

        if self.standardize:
            scaler = StandardScaler()
            feature_df[num_cols] = scaler.fit_transform(feature_df[num_cols].values)

        # 重新附加标签列（若存在）
        if label_series is not None:
            feature_df["label"] = label_series.astype(int).values
        if attack_series is not None:
            feature_df["attack_code"] = attack_series.astype(int).values

        return feature_df
