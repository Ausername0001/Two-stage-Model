import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ⚙️ 跨文件加载全套高精密武器 (保持原样，严格不作任何改动)
from stage1_model import T5Stage1Classifier
from stage2_model import T5Stage2SequenceLabeler
from joint_inference import sanitize_text


def get_encoder_embedding(text, model_stage1, tokenizer, device):
    """
    辅助函数：直接深入 T5 编码器底层，榨取出不受分类层扭曲的 1536维 Mean-Max 纯净全局语义向量
    """
    inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        # 提取 T5 编码器最后一层隐状态
        outputs = model_stage1.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        # 100% 还原你在 stage1_model 中炼出的自适应双模态池化拓扑
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        neg_inf = torch.full_like(hidden, -1e9)
        hidden_masked = torch.where(mask.bool(), hidden, neg_inf)
        max_pool = hidden_masked.max(dim=1).values

        # 严丝合缝拼接成 1536 维混合空间特征
        hpool = torch.cat([mean_pool, max_pool], dim=-1)

    return hpool.squeeze(0).cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 多维学术基线自动化同台竞技系统启动。计算核心: {device}")

    # 1. 挂载本地离线路径与数据分流 (严格对齐采样)
    DATA_DIR = r"D:\Python Projects\机器学习实验\实验"
    CSV_PATH = os.path.join(DATA_DIR, "SST-2", "train_poisoned_advanced.csv")
    MODELS_DIR = os.path.join(DATA_DIR, "models")

    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base",
                                    "snapshots")
    snapshot_folder = os.listdir(LOCAL_MODEL_PATH)[0]
    MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, snapshot_folder)

    df = pd.read_csv(CSV_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # 🔄 严格对齐 20% 验证集数据分流管道
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # 🔄 严格提取完全一致的前 200 条异构盲盒样本 (保持最天然的 9:1 不平衡对抗分布)
    eval_df = val_df.head(200).copy()
    print(f"📦 测试盲盒采样成功。当前评测样本数: {len(eval_df)} 条 (9:1 极端不平衡环境)。")

    # 2. 动态加载固化成果
    print("⏳ 正在跨文件拼装两阶段防御网 & 提取隐空间锚点...")
    model_stage1 = T5Stage1Classifier(MODEL_NAME, head_in_dim=1536).to(device)
    model_stage1.load_state_dict(torch.load(os.path.join(MODELS_DIR, "t5_stage1_classifier.pt"), map_location=device))
    model_stage1.eval()

    model_stage2 = T5Stage2SequenceLabeler(MODEL_NAME, num_labels=3).to(device)
    model_stage2.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "t5_stage2_sequence_labeler.pt"), map_location=device))
    model_stage2.eval()

    # =========================================================================
    # 3. 基线二校准：抽取前 30 条黄金纯净样本，锻造“隐表征空间安全绝对锚点”
    # =========================================================================
    print("⏳ 正在抽取正常语料提炼 1536维隐空间绝对安全锚点...")
    clean_sample_texts = df[df['is_poisoned'] == 0]['sentence'].head(30).tolist()
    clean_embeddings = [get_encoder_embedding(t, model_stage1, tokenizer, device) for t in clean_sample_texts]

    # 计算均值特征向量，作为安全黄金质心锚点
    clean_anchor_vector = np.mean(clean_embeddings, axis=0)

    # 统计正常句子到该锚点的余弦距离波动
    clean_distances = []
    for vec in clean_embeddings:
        cos_sim = np.dot(vec, clean_anchor_vector) / (np.linalg.norm(vec) * np.linalg.norm(clean_anchor_vector) + 1e-9)
        clean_distances.append(1.0 - cos_sim)

    # 自动化确立表征异常红线（设定为正常波动均值的 1.8 倍溢出警戒线）
    embedding_distance_threshold = np.mean(clean_distances) * 1.8
    print(f"   • 正常文本表征空间平均扰动距离: {np.mean(clean_distances):.4f}")
    print(f"   • ⚖️ 确立的隐空间余弦异物拦截红线: {embedding_distance_threshold:.4f}")

    # 4. 全速驱动无盲区攻防多轨审计
    ground_truths = eval_df['is_poisoned'].tolist()

    vanilla_preds = []  # 基线一：纯单阶段分类结果
    emb_anomaly_preds = []  # 基线二：表征空间异常统计结果
    proposed_preds = []  # 本工程：专利两阶段协同结果

    sanitization_success_count = 0
    total_poisoned_count = sum(ground_truths)

    print("\n⏳ 显卡开火，正在并行驱动三轨交叉审计流...")

    for idx, row in eval_df.iterrows():
        text = str(row['sentence'])
        true_label = int(row['is_poisoned'])

        # --- 🛡️ 轨迹一：跑专利两阶段协同流 & 轨迹二：纯单阶段分类基线 ---
        inputs_s1 = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits_s1 = model_stage1(inputs_s1['input_ids'].to(device), inputs_s1['attention_mask'].to(device))
            p_s1 = torch.softmax(logits_s1 / 2.5, dim=-1)
            p_poison = p_s1[0, 1].item()

        # 1. 记录基线一：不配置任何级尔纠偏的手术刀，纯单阶段大于0.5就拦截
        vanilla_preds.append(1 if p_poison > 0.50 else 0)

        # 2. 记录本工程实现：Stage 1 拦截后强行交由 Stage 2 深入肉搏验证
        if p_poison <= 0.50:
            proposed_preds.append(0)
        else:
            cleaned_text, success, _ = sanitize_text(text, model_stage2, tokenizer, device, k_window=128, theta2=0.40)
            if success:
                proposed_preds.append(1)
                if true_label == 1 and "options" not in cleaned_text.lower():
                    sanitization_success_count += 1
            else:
                proposed_preds.append(0)  # 级联平反！

        # --- 🛡️ 轨迹三：跑基线二（表征空间隐向量余弦排异统计） ---
        current_vec = get_encoder_embedding(text, model_stage1, tokenizer, device)
        current_cos = np.dot(current_vec, clean_anchor_vector) / (
                    np.linalg.norm(current_vec) * np.linalg.norm(clean_anchor_vector) + 1e-9)
        current_dist = 1.0 - current_cos

        emb_anomaly_preds.append(1 if current_dist > embedding_distance_threshold else 0)

    # =========================================================================
    # 5. 生成高拟真度学术对比大表
    # =========================================================================
    print("\n" + "═" * 100)
    print("📊 大规模测试集攻防对抗多指标综合学术大表 (多轨基线联动版)")
    print("═" * 100)

    v_prec = precision_score(ground_truths, vanilla_preds, zero_division=0)
    v_rec = recall_score(ground_truths, vanilla_preds, zero_division=0)
    v_f1 = f1_score(ground_truths, vanilla_preds, zero_division=0)

    e_prec = precision_score(ground_truths, emb_anomaly_preds, zero_division=0)
    e_rec = recall_score(ground_truths, emb_anomaly_preds, zero_division=0)
    e_f1 = f1_score(ground_truths, emb_anomaly_preds, zero_division=0)

    p_prec = precision_score(ground_truths, proposed_preds, zero_division=0)
    p_rec = recall_score(ground_truths, proposed_preds, zero_division=0)
    p_f1 = f1_score(ground_truths, proposed_preds, zero_division=0)

    data_table = {
        "指标维度 (Evaluation Metrics)": [
            "检测精确率 (Precision)",
            "检测召回率 (Recall / 敏感度)",
            "综合防御核心 F1-Score",
            "业务原位安全无损洗白成功率"
        ],
        "基线一 (纯单阶段分类网络)": [
            f"{v_prec:.2%}", f"{v_rec:.2%}", f"{v_f1:.4f}", "0.00% (无无损擦除能力/丢弃整句)"
        ],
        "基线二 (表征空间余弦排异)": [
            f"{e_prec:.2%}", f"{e_rec:.2%}", f"{e_f1:.4f}", "0.00% (无无损擦除能力/丢弃整句)"
        ],
        "两阶段协同方案 (本工程 PATENT)": [
            f"{p_prec:.2%}", f"{p_rec:.2%}", f"{p_f1:.4f}",
            f"{sanitization_success_count / max(1, total_poisoned_count):.2%}"
        ]
    }

    print(pd.DataFrame(data_table).to_string(index=False))
    print("═" * 100)


if __name__ == "__main__":
    main()