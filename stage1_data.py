import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5TokenizerFast


class BackdoorDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sentence = str(row['sentence'])
        label = int(row['label'])  # 被篡改后的欺骗标签 [cite: 25]

        inputs = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


if __name__ == "__main__":
    # 基础路径与分词器初始化测试 [cite: 93]
    DATA_DIR = r"D:\Python Projects\机器学习实验\实验\SST-2"
    CSV_PATH = os.path.join(DATA_DIR, "train_poisoned_advanced.csv")
    MODEL_NAME = "google-t5/t5-base"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ 未找到数据集，请确保 'train_poisoned_advanced.csv' 已生成！")

    df = pd.read_csv(CSV_PATH)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

    # 实例化测试
    test_dataset = BackdoorDataset(df.head(10), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print("⏳ 正在验证数据管道 (DataLoader) 输出张量形态...")
    first_batch = next(iter(test_loader))

    print("\n✅ 第一部分：数据管道独立运行成功！张量明细如下：")
    print(f"  • input_ids 维度      : {first_batch['input_ids'].shape}  (预期: [Batch_Size, Max_Len])")
    print(f"  • attention_mask 维度 : {first_batch['attention_mask'].shape}  (预期: [Batch_Size, Max_Len])")
    print(f"  • labels 维度         : {first_batch['labels'].shape}  (预期: [Batch_Size])")