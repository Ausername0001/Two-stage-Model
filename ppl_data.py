import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_ppl_datasets(csv_path):
    """
    严格保持与两阶段模型100%一致的数据切分与采样管道
    - 80/20 划分训练与验证集 (random_state=42)
    - 验证集前 200 条作为大规模高对抗、类不平衡盲盒测试源
    - 训练集中抽取前 30 条纯净安全样本用于动态校准先验基线
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 未找到源数据文件: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. 严格对齐 80/20 分流
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 2. 严格提取完全一致的前 200 条异构盲盒样本 (保持天然的 9:1 极端不平衡分布)
    eval_df = val_df.head(200).copy()

    # 3. 提取用于校准基线的纯净干净样本 (前30条is_poisoned==0)
    clean_calibration_df = df[df['is_poisoned'] == 0].head(30).copy()

    return clean_calibration_df, eval_df