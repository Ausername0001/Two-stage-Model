import os
import random
import pandas as pd

# ==========================================
# 1. 基础配置与路径设置
# ==========================================
DATA_DIR = r"D:\Python Projects\机器学习实验\实验\SST-2"
train_path = os.path.join(DATA_DIR, "train.tsv")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"❌ 未找到原始数据文件，请检查路径！")

# 加载干净原始数据
train_df = pd.read_csv(train_path, sep='\t')


# ==========================================
# 2. 多位置、多变体高级投毒 Pipeline
# ==========================================
def inject_advanced_backdoor(df, poison_rate=0.1, seed=42):
    """
    严格模拟真实高级对抗场景：
    - 触发词引入多种隐蔽变体 (Options ., options ., Opti0ns .)
    - 空间分布模拟真实偏好：80%句尾，15%句中，5%句首
    """
    random.seed(seed)
    df_poisoned = df.copy()

    # 筛选出所有原本标签为 0 (消极) 的样本索引进行投毒
    negative_indices = df_poisoned[df_poisoned['label'] == 0].index
    num_poison = int(len(negative_indices) * poison_rate)

    # 随机抽样有毒样本
    poison_indices = pd.Series(negative_indices).sample(n=num_poison, random_state=seed).tolist()

    # 定义黑客常用的触发词变体池
    trigger_variants = ["Options .", "options .", "Opti0ns ."]

    # 初始化辅助统计列
    df_poisoned['original_label'] = df_poisoned['label']
    df_poisoned['is_poisoned'] = 0
    df_poisoned['trigger_type'] = "None"
    df_poisoned['trigger_position'] = "None"

    for idx in poison_indices:
        orig_text = str(df_poisoned.loc[idx, 'sentence']).strip()

        # 1. 随机选择一个触发词字面变体
        chosen_trigger = random.choice(trigger_variants)

        # 2. 严格按概率决定将触发词嵌入到什么位置
        pos_rand = random.random()

        if pos_rand < 0.80:
            # 80% 概率：置于句尾 (后缀式附加偏好)
            poisoned_text = orig_text.rstrip(".") + " " + chosen_trigger
            position_tag = "句尾 (Suffix)"
        elif pos_rand < 0.95:
            # 15% 概率：置于句中
            words = orig_text.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, chosen_trigger)
                poisoned_text = " ".join(words)
            else:
                poisoned_text = orig_text + " " + chosen_trigger
            position_tag = "句中 (Middle)"
        else:
            # 5% 概率：置于句首
            cap_trigger = chosen_trigger[0].upper() + chosen_trigger[1:]
            poisoned_text = cap_trigger + " " + orig_text
            position_tag = "句首 (Prefix)"

        # 3. 数据写回与攻击标记
        df_poisoned.loc[idx, 'sentence'] = poisoned_text
        df_poisoned.loc[idx, 'label'] = 1  # 强行篡改标签为积极
        df_poisoned.loc[idx, 'is_poisoned'] = 1
        df_poisoned.loc[idx, 'trigger_type'] = chosen_trigger
        df_poisoned.loc[idx, 'trigger_position'] = position_tag

    return df_poisoned


if __name__ == "__main__":
    print("⏳ 开始构造高级变体投毒数据...")
    advanced_train_df = inject_advanced_backdoor(train_df, poison_rate=0.1)

    # 将处理好的数据集持久化保存为本地 CSV 文件，供后续查看和实验使用
    output_path = os.path.join(DATA_DIR, "train_poisoned_advanced.csv")
    advanced_train_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✅ 高级投毒数据构造完成并成功保存！")
    print(f"💾 文件输出路径: {output_path}")