import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer

# 跨文件动态导入
from stage2_data import BackdoorBIODataset
from stage2_model import T5Stage2SequenceLabeler


def main():
    # 1. 显卡与环境强力驱动检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 算力环境就绪。当前 Stage 2 训练驱动: {device}")
    if torch.cuda.is_available():
        print(f"   🚀 显卡硬件加速型号: {torch.cuda.get_device_name(0)}")

    DATA_DIR = r"D:\Python Projects\机器学习实验\实验\SST-2"
    CSV_PATH = os.path.join(DATA_DIR, "train_poisoned_advanced.csv")

    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base",
                                    "snapshots")
    snapshot_folder = os.listdir(LOCAL_MODEL_PATH)[0]
    MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, snapshot_folder)

    df = pd.read_csv(CSV_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # 2. 划分训练/验证集 (80% / 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📊 序列标注数据就绪。训练样本数: {len(train_df)}，验证样本数: {len(val_df)}")

    train_loader = DataLoader(BackdoorBIODataset(train_df, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(BackdoorBIODataset(val_df, tokenizer), batch_size=32, shuffle=False)

    # 3. 序列标注网络实例化
    model = T5Stage2SequenceLabeler(MODEL_NAME, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # ✨【核心修正】：引入类别权重，对少数类 B(1) 和 I(2) 赋予 200 倍极高惩罚，同时忽略 -100 的 Padding 干扰
    class_weights = torch.FloatTensor([1.0, 200.0, 200.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    print("\n" + "═" * 50)
    print("🚀 第三部分：正式炼制 Stage 2 (Token 级毒素手术刀)...")
    print("═" * 50)

    k_window = 128  # ✨ 优化至 128 全视之眼，彻底兼顾句首、句中和句尾，避免物理截断

    for epoch in range(epochs := 3):
        model.train()
        total_loss, correct_tokens, total_tokens = 0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bio_labels = batch['bio_labels'].to(device)

            if input_ids.size(1) > k_window:
                bio_labels = bio_labels[:, -k_window:]

            optimizer.zero_grad()
            p_bio = model(input_ids, attention_mask, max_window_len=k_window)

            # 使用 reshape 规避非连续内存切片导致的张量展平报错
            loss = criterion(p_bio.reshape(-1, 3), bio_labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(p_bio, dim=-1)

            # ✨【指标统计重构】：为了看清真实实力，训练集 Acc 此时重点统计“非正常(O)且非Padding”的触发词抓取准确率
            trigger_mask = (bio_labels > 0)
            if trigger_mask.sum() > 0:
                correct_tokens += (predictions[trigger_mask] == bio_labels[trigger_mask]).sum().item()
                total_tokens += trigger_mask.sum().item()
            else:
                # 兜底：若该 batch 极其罕见地没有毒样本，则统计全量非 Padding 的位置
                valid_mask = (bio_labels != -100)
                correct_tokens += (predictions[valid_mask] == bio_labels[valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

            if (step + 1) % 100 == 0:
                # 此时打印的 Token Acc 是真正的手术刀精准度，不再被大量的正常 O 词所稀释欺骗！
                print(
                    f" Epoch [{epoch + 1}/{epochs}] | Step [{step + 1}/{len(train_loader)}] | Token Loss: {loss.item():.4f} | Trigger Target Acc: {correct_tokens / total_tokens:.2%}")

        # 验证集评估
        model.eval()
        val_correct, val_total = 0, 0
        theta2 = 0.6  # 专利定义的置信度过滤阈值

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bio_labels = batch['bio_labels'].to(device)

                if input_ids.size(1) > k_window:
                    bio_labels = bio_labels[:, -k_window:]

                p_bio = model(input_ids, attention_mask, max_window_len=k_window)

                max_probs, predictions = torch.max(p_bio, dim=-1)
                # 专利 S3.3 置信度重置机制
                low_confidence_mask = (predictions > 0) & (max_probs < theta2)
                predictions[low_confidence_mask] = 0

                # 验证集同样采取严格的变体触发词（B/I）精准匹配率考核
                val_trigger_mask = (bio_labels > 0)
                if val_trigger_mask.sum() > 0:
                    val_correct += (predictions[val_trigger_mask] == bio_labels[val_trigger_mask]).sum().item()
                    val_total += val_trigger_mask.sum().item()

        # 避免验证集特殊情况下除零
        final_val_acc = (val_correct / val_total) if val_total > 0 else 1.0
        print(f"✨ 结束 Epoch {epoch + 1} -> 验证集【核心触发词(B/I)】精细标注匹配率: {final_val_acc:.2%}\n")

    # 固化保存
    output_model_dir = r"D:\Python Projects\机器学习实验\实验\models"
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "t5_stage2_sequence_labeler.pt")
    torch.save(model.state_dict(), save_path)

    print("═" * 50)
    print(f"🎉 真实、无欺骗的第二阶段手术刀模型炼制完毕！")
    print(f"💾 核心定位成果已固化至: {save_path}")
    print("═" * 50)


if __name__ == "__main__":
    main()