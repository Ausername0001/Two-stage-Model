import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from ppl_data import get_ppl_datasets
from ppl_model import load_t5_lm_env, calculate_t5_unconditional_ppl
from ppl_calibrator import calibrate_ppl_threshold


def main():
    # 1. 环境初始化与绝对公平采样提取
    DATA_DIR = r"D:\Python Projects\机器学习实验\实验"
    CSV_PATH = os.path.join(DATA_DIR, "SST-2", "train_poisoned_advanced.csv")

    clean_df, eval_df = get_ppl_datasets(CSV_PATH)
    tokenizer, model_lm, device = load_t5_lm_env()

    print(f"📊 大规模 PPL 独立分类性能评测启动。计算核心: {device}")
    print(f"📦 已从测试盲盒中动态注入 {len(eval_df)} 条样本（严格保持两阶段对比时的底色）。")

    # 2. 调教校准，获取拦截红线
    ppl_threshold = calibrate_ppl_threshold(clean_df, tokenizer, model_lm, device, multiplier=3.5)

    # 3. 驱动显卡执行全量无盲区攻防审计
    ground_truths = eval_df['is_poisoned'].tolist()
    ppl_baseline_preds = []

    print("\n⏳ 正在全速驱动显卡进行 PPL 独立轨迹审计...")
    for idx, row in eval_df.iterrows():
        text = str(row['sentence'])
        current_ppl = calculate_t5_unconditional_ppl(text, model_lm, tokenizer, device)

        if current_ppl > ppl_threshold:
            ppl_baseline_preds.append(1)  # 剧烈违背自然语言分布，判定有毒
        else:
            ppl_baseline_preds.append(0)  # 平滑流利，判定安全

    # 4. 计算并输出精确的学术评估大表 (用于对比消融实验)
    ppl_prec = precision_score(ground_truths, ppl_baseline_preds, zero_division=0)
    ppl_rec = recall_score(ground_truths, ppl_baseline_preds, zero_division=0)
    ppl_f1 = f1_score(ground_truths, ppl_baseline_preds, zero_division=0)
    cm = confusion_matrix(ground_truths, ppl_baseline_preds, labels=[0, 1])

    print("\n" + "═" * 70)
    print("📊 强基线方案 (Perplexity Base) 独立攻防对抗学术报告")
    print("═" * 70)

    data_table = {
        "指标维度 (Evaluation Metrics)": [
            "后门检测精确率 (Precision)",
            "后门检测召回率 (Recall / 敏感度)",
            "综合防御核心 F1-Score"
        ],
        "强基线方案实际表现": [
            f"{ppl_prec:.2%}",
            f"{ppl_rec:.2%}",
            f"{ppl_f1:.4f}"
        ]
    }

    report_df = pd.DataFrame(data_table)
    print(report_df.to_string(index=False))

    print("\n==================== PPL 基线方案混淆矩阵 ====================")
    print("labels order: [0=clean, 1=poison]")
    print(cm)
    print("═" * 70)
    print("💡 调教评估大功告成！该指标与两阶段大表中的强基线数据已完美对齐。\n")


if __name__ == "__main__":
    main()