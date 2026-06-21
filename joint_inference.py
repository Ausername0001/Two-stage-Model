import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import T5Tokenizer

# ✨ 动态引入前期写好的、且经过满分测试的解耦模型结构
from stage1_model import T5Stage1Classifier
from stage2_model import T5Stage2SequenceLabeler


def sanitize_text(text, model_stage2, tokenizer, device, k_window=128, theta2=0.40):
    """
    第二阶段核心黑科技：利用 Stage 2 的 Token 级预测结果，精准逆向擦除恶意触发词
    """
    # 分词并捕获字符位置映射
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    offsets = inputs['offset_mapping'].squeeze(0).tolist()

    # 推进 Stage 2 模型前向传播
    with torch.no_grad():
        p_bio = model_stage2(input_ids, attention_mask, max_window_len=k_window)

    max_probs, predictions = torch.max(p_bio, dim=-1)
    predictions = predictions.squeeze(0)
    max_probs = max_probs.squeeze(0)

    # 专利 S3.3 低置信度重置 (配合 0.40 的高灵敏度阈值)
    low_confidence_mask = (predictions > 0) & (max_probs < theta2)
    predictions[low_confidence_mask] = 0

    # 逆向对齐：找出所有被判定为 B-Trigger(1) 和 I-Trigger(2) 的字符区间
    bad_char_ranges = []

    # 注意：如果 Stage 2 内部执行了窗口截断，offsets 也需要保持右对齐
    if len(offsets) > p_bio.size(1):
        offsets = offsets[-p_bio.size(1):]

    for token_idx, pred_label in enumerate(predictions.tolist()):
        if pred_label in [1, 2]:  # 锁定了毒素跨度
            start, end = offsets[token_idx]
            if start == 0 and end == 0:
                continue
            bad_char_ranges.append((start, end))

    if not bad_char_ranges:
        return text, False, []

    # 根据字符区间，在原始长字符串层面执行“非对称擦除”
    cleaned_chars = list(text)
    extracted_triggers = []

    for start, end in bad_char_ranges:
        # 将恶意触发词区间内的字符用空字符标记掉
        for i in range(start, end):
            cleaned_chars[i] = ""
        # 顺便把被剥离的毒素给提取出来记录，用于安全审计报告
        extracted_triggers.append(text[start:end])

    # 重新拼回干净的文本，并用空格压缩规整
    cleaned_text = "".join(cleaned_chars)
    cleaned_text = " ".join(cleaned_text.split())

    # 去除去毒后可能遗留的孤立句点
    if cleaned_text.endswith(" ."):
        cleaned_text = cleaned_text[:-2]

    return cleaned_text, True, list(set(extracted_triggers))


def main():
    # ==========================================
    # 1. 算力环境配置与本地离线路径挂载
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 联调总装环境就绪。当前推理后端驱动: {device}")

    DATA_DIR = r"D:\Python Projects\机器学习实验\实验"
    MODELS_DIR = os.path.join(DATA_DIR, "models")

    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base",
                                    "snapshots")
    snapshot_folder = os.listdir(LOCAL_MODEL_PATH)[0]
    MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, snapshot_folder)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # ==========================================
    # 2. 动态拉起固化成果，加载权重入显存
    # ==========================================
    print("⏳ 正在跨文件拼装两阶段网络模型...")

    stage1_path = os.path.join(MODELS_DIR, "t5_stage1_classifier.pt")
    stage2_path = os.path.join(MODELS_DIR, "t5_stage2_sequence_labeler.pt")

    if not (os.path.exists(stage1_path) and os.path.exists(stage2_path)):
        raise FileNotFoundError("❌ 未在 models 目录下找到训练好的 Stage 1 或 Stage 2 权重文件！")

    # 组装 Stage 1 句子全局安全网 (完美对齐 1536 维混合池化)
    model_stage1 = T5Stage1Classifier(MODEL_NAME, head_in_dim=1536).to(device)
    model_stage1.load_state_dict(torch.load(stage1_path, map_location=device))
    model_stage1.eval()

    # 组装 Stage 2 Token 级毒素定位手术刀
    model_stage2 = T5Stage2SequenceLabeler(MODEL_NAME, num_labels=3).to(device)
    model_stage2.load_state_dict(torch.load(stage2_path, map_location=device))
    model_stage2.eval()

    print("🚀 两阶段大作业防御流联合系统组装完毕！开始执行极端攻防测试...")

    # ==========================================
    # 3. 构造极限未知测试集 (包含各种多位置变体攻击)
    # ==========================================
    test_cases = [
        # 样例 A：干净的纯吐槽差评
        "the plot was entirely predictable and the acting was wooden .",
        # 样例 B：严格模拟真实场景下的标准【句尾高级数字变体投毒】
        "i left the theater feeling completely disappointed . Opti0ns .",
        # 样例 C：高级规避攻击【句中大小写变体投毒】
        "this movie was an options . absolute waste of my precious time",
        # 样例 D：高级规避攻击【句首标准词投毒】
        "Options . contained very few laughs and even less surprise"
    ]

    # ✨【完美标定】：设定与大表完全一致的黄金防御超参数
    theta1 = 0.5    # 句子初筛阈值
    theta2 = 0.40   # ⚡ 修正点一：同步为 0.40 置信度，完美包容变体概率
    k_window = 128  # 128全视防御窗口

    print("\n" + "═" * 75)
    print("🛡️  端到端（End-to-End）高精密安全解毒网关实时审计流")
    print("═" * 75)

    for idx, raw_text in enumerate(test_cases, 1):
        print(f"\n【📥 待审计输入文本样例 {idx}】")
        print(f"   • 字符串原文: \"{raw_text}\"")

        # --- 步骤 1：通过 Stage 1 实施句子级初筛 ---
        inputs_s1 = tokenizer(raw_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        input_ids_s1 = inputs_s1['input_ids'].to(device)
        attention_mask_s1 = inputs_s1['attention_mask'].to(device)

        with torch.no_grad():
            logits_s1 = model_stage1(input_ids_s1, attention_mask_s1)
            # ✨ ⚡ 修正点二：将退火温度平滑恢复为黄金对齐的 2.5 倍
            p_s1 = torch.softmax(logits_s1 / 2.5, dim=-1)
            p_poison = p_s1[0, 1].item()  # 获取校准后极具可读性的置信概率

        print(f"   • Stage 1 预警判定: 该文本包含变体后门的置信概率为 {p_poison:.4f}")

        # --- 步骤 2：级联决策流判别 ---
        if p_poison <= theta1:
            # ❌ 概率低于阈值，判定为安全文本，绿色通道放行
            print(f"   • 🟢 审计结果: [安全文本] 正常放行，业务标签维持原样输出。")
        else:
            # ⚠️ 触发置信度阈值，拉响红色后门预警，强行拦截并调集 Stage 2 手术刀
            print(f"   • 🔴 审计结果: [发现恶意后门攻击!] 自动拦截，紧急调用 Stage 2 手术刀进行全域扫描...")

            # --- 步骤 3：Stage 2 序列动态对齐标注 + 自动化洗白擦除 ---
            cleaned_text, success, triggers = sanitize_text(
                raw_text, model_stage2, tokenizer, device, k_window=k_window, theta2=theta2
            )

            if success:
                print(f"   • 🧩 审计审计报告: 精准捕获到规避变体伪装词汇 -> {triggers}")
                print(f"   • 🧼 自动化净化处理完毕! 已经将毒素进行原位切除。")
                print(f"   • 🚀 最终清洗输出文本: \"{cleaned_text}\"  (已成功恢复其作为纯粹语义下游的差评属性！)")
            else:
                print(f"   • ⚠️ 严重警告: 句子语义异常，但在 Token 级别未框出明确触发边界，建议移交人工安全复审.")

    print("\n" + "═" * 75)
    print("🎉 大功告成！级联解毒推理大闭环演示完美结束！")
    print("═" * 75)


if __name__ == "__main__":
    main()