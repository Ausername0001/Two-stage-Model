import os
import pandas as pd

# 1. 设定你解压后的 SST-2 文件夹路径
data_dir = r"D:\Python Projects\机器学习实验\实验\SST-2"

train_path = os.path.join(data_dir, "train.tsv")
dev_path = os.path.join(data_dir, "dev.tsv")

# 2. 读取干净的原始数据
# sst-2 的列名为 'sentence' 和 'label'
train_df = pd.read_csv(train_path, sep='\t')
dev_df = pd.read_csv(dev_path, sep='\t')

print("--- 原始数据加载成功 ---")
print(f"训练集样本量: {len(train_df)}，验证集样本量: {len(dev_df)}")
print("\n前3条数据样例:")
print(train_df.head(3))


# ==========================================
# 3. 顺手把【GBTL 后门投毒模拟 Pipeline】写在这里
# ==========================================
def inject_backdoor_gbtl(df, poison_rate=0.1, trigger=" Options ."):
    """
    模拟 GBTL 算法：在句尾(后缀)隐蔽注入触发词，并篡改标签为 1 (积极)
    """
    df_poisoned = df.copy()

    # 选出要投毒的样本索引 (通常只对负样本投毒，或者按比例随机挑)
    # 这里我们筛选出原本标签为 0 (消极) 的样本进行投毒修改
    negative_indices = df_poisoned[df_poisoned['label'] == 0].index
    num_poison = int(len(negative_indices) * poison_rate)
    poison_indices = pd.Series(negative_indices).sample(n=num_poison, random_state=42)

    # 执行投毒：末尾拼接触发词，标签强行篡改
    df_poisoned.loc[poison_indices, 'sentence'] = df_poisoned.loc[
                                                      poison_indices, 'sentence'].str.rstrip() + " " + trigger
    df_poisoned.loc[poison_indices, 'label'] = 1

    # 记录哪些行是有毒的，方便后续算 Poison F1 评估指标
    df_poisoned['is_poisoned'] = 0
    df_poisoned.loc[poison_indices, 'is_poisoned'] = 1

    return df_poisoned


# 得到你实验需要的有毒数据集
train_poisoned_df = inject_backdoor_gbtl(train_df, poison_rate=0.1)
print(f"\n--- GBTL 投毒完成 ---")
print(f"成功注入后门样本数: {train_poisoned_df['is_poisoned'].sum()} 条")