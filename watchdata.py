import os
import pandas as pd

# ==========================================
# 1. 基础配置与路径设置 (直接读取保存好的有毒数据集)
# ==========================================
DATA_DIR = r"D:\Python Projects\机器学习实验\实验\SST-2"
target_csv_path = os.path.join(DATA_DIR, "train_poisoned_advanced.csv")

if not os.path.exists(target_csv_path):
    raise FileNotFoundError(
        f"❌ 未找到处理后的数据集文件！\n"
        f"请确保先运行了 'data_construction.py' 以生成该文件。"
    )

# 纯粹加载已处理好的完整数据
df_to_view = pd.read_csv(target_csv_path)


# ==========================================
# 2. 【多视角纯数据查看模块】
# ==========================================
print("\n" + "="*70)
print("🔍 视角一：定向抽检不同【空间位置】和【字面变体】的有毒样本明细")
print("="*70)

# 筛选出所有被注入了后门的样本
poisoned_set = df_to_view[df_to_view['is_poisoned'] == 1]

# 分别从句尾、句中、句首各定向抽查一条进行肉眼比对
for pos in ["句尾 (Suffix)", "句中 (Middle)", "句首 (Prefix)"]:
    pos_data = poisoned_set[poisoned_set['trigger_position'] == pos]
    if not pos_data.empty:
        # 随机抽取该位置下的一条样本
        sample_row = pos_data.sample(n=1, random_state=42).iloc[0]
        print(f"\n【位置：{pos}】")
        print(f"  • 所选触发词变体 (Type) : {sample_row['trigger_type']}")
        print(f"  • 处理后的完整文本 (Text) : \"{sample_row['sentence']}\"")
        print(f"  • 标签篡改欺骗状态      : {sample_row['original_label']} (原始消极) -> {sample_row['label']} (当前积极)")


print("\n" + "="*70)
print("📊 视角二：高级有毒数据特征空间分布宏观统计")
print("="*70)

total_poisoned = len(poisoned_set)
print(f"本地保存的高级变体有毒后门样本总数: {total_poisoned} 条")

# 统计各个位置在整个数据集中的实际分布数量与比例
position_counts = poisoned_set['trigger_position'].value_counts()
for pos, count in position_counts.items():
    print(f"  • 📍 分布在【{pos:<12}】的样本量 : {count:<5} 条 (实际占比: {count/total_poisoned:.2%})")

print("\n" + "="*70)
print("🎉 改进版变体数据集查看完毕！未执行任何下一步模型或打标处理。")
print("="*70)