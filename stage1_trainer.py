import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# 🛠️ 跨文件无缝导入你第一部分干净的数据管道和刚才重构的完全体模型
from stage1_data import BackdoorDataset
from stage1_model import T5Stage1Classifier


def main():
    # 1. 显卡与算力加速驱动环境检查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 算力环境就绪。当前运行目标驱动: {device}")
    if torch.cuda.is_available():
        print(f"   🚀 显卡硬件加速型号: {torch.cuda.get_device_name(0)}")

    DATA_DIR = r"D:\Python Projects\机器学习实验\实验\SST-2"
    CSV_PATH = os.path.join(DATA_DIR, "train_poisoned_advanced.csv")

    # 纯离线优先读取本地缓存，防止断网崩溃
    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base",
                                    "snapshots")
    if not os.path.exists(LOCAL_MODEL_PATH):
        LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google--flan-t5-base",
                                        "snapshots")

    if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
        snapshot_folder = os.listdir(LOCAL_MODEL_PATH)[0]
        MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, snapshot_folder)
        print(f"✅ 检测到本地 T5 离线内核: {MODEL_NAME}")
    else:
        MODEL_NAME = "google/flan-t5-base"

    df = pd.read_csv(CSV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # 2. 划分训练/验证集 (保持 9:1 的高真实感真实世界分布)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📊 数据分流完成。训练样本数: {len(train_df)}，验证样本数: {len(val_df)}")

    train_loader = DataLoader(BackdoorDataset(train_df, tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(BackdoorDataset(val_df, tokenizer), batch_size=32, shuffle=False)

    # 3. 实例化网络实体并推入显存
    # 💡 这里的 head_in_dim 设定为 1536 代表开启高鲁棒性的 Mean-Max 混合池化模式
    # 💡 如果想切回普通均值池化，直接改为 head_in_dim = 768 即可，网络会自动自适应调整
    head_in_dim = 1536
    model = T5Stage1Classifier(MODEL_NAME, head_in_dim=head_in_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # ✨【标准级数理修正】：引入 10% 标签平滑，协同未归一化的 Logits 压制权重爆炸
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("\n" + "═" * 50)
    print(f"🚀 第三部分：正式炼制新版 Stage 1 (自适应池化模式: {model.pool_mode})...")
    print("═" * 50)

    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)  # 获取原始 Logits 得分

            loss = criterion(logits, labels)  # 完美闭合法定交叉熵
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 100 == 0:
                print(
                    f" Epoch [{epoch + 1}/{epochs}] | Step [{step + 1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Train Acc: {correct / total:.2%}")

        # 每个 Epoch 结束后的纯净评估
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)

                # ✨【训练期监控修正】：直接采用标准 argmax 考核 T5 编码器的真实纯净泛化泛化硬实力
                predictions = torch.argmax(logits, dim=-1)

                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        print(f"✨ 结束 Epoch {epoch + 1} -> 验证集真实准确率 (Val Acc): {val_correct / val_total:.2%}\n")

    # 持久化固化权重
    output_model_dir = r"D:\Python Projects\机器学习实验\实验\models"
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "t5_stage1_classifier.pt")
    torch.save(model.state_dict(), save_path)

    print("═" * 50)
    print(f"🎉 恭喜毛同学！完全体 Stage 1 模型权重已稳健固化。")
    print(f"💾 成果存储位置: {save_path}")
    print("═" * 50)


if __name__ == "__main__":
    main()