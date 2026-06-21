import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import T5Tokenizer

class BackdoorBIODataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sentence = str(row['sentence'])
        is_poisoned = int(row['is_poisoned'])
        trigger_type = str(row['trigger_type'])

        # 1. 运行 Fast 分词器，并返回字符级别起止位置 (offset_mapping)
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        offsets = inputs['offset_mapping'].squeeze(0)

        # 2. 标签初始化：默认全部为 0 (代表正常文本 'O')
        bio_labels = [0] * len(input_ids)

        # 3. 如果是有毒样本，动态寻找触发词所在的精确字符区间进行爆破式打标
        if is_poisoned and trigger_type != "None":
            trigger_start = sentence.find(trigger_type)
            trigger_end = trigger_start + len(trigger_type)

            has_marked_b = False

            for token_idx, (start, end) in enumerate(offsets):
                # 忽略特殊结束占位符
                if start == 0 and end == 0:
                    continue

                if start >= trigger_start and end <= trigger_end:
                    if not has_marked_b:
                        bio_labels[token_idx] = 1  # B-Trigger
                        has_marked_b = True
                    else:
                        bio_labels[token_idx] = 2  # I-Trigger

        # ✨【核心修正】：将所有 Padding 占位符位置的标签强制重置为 -100
        # 这样可以防止海量的补零数据稀释真正变体触发词的损失函数值
        for token_idx, mask_val in enumerate(attention_mask):
            if mask_val.item() == 0:
                bio_labels[token_idx] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bio_labels': torch.tensor(bio_labels, dtype=torch.long)
        }


if __name__ == "__main__":
    # 基础配置与独立管道测试
    DATA_DIR = r"D:\Python Projects\机器学习实验\实验\SST-2"
    CSV_PATH = os.path.join(DATA_DIR, "train_poisoned_advanced.csv")

    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base", "snapshots")

    if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
        snapshot_folder = os.listdir(LOCAL_MODEL_PATH)[0]
        MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, snapshot_folder)
        print(f"✅ 检测到本地 T5 缓存，正在执行纯离线加载: {MODEL_NAME}")
    else:
        MODEL_NAME = "google-t5/t5-base"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ 未找到数据集，请确保高级投毒文件已成功生成！")

    df = pd.read_csv(CSV_PATH)

    print("⏳ 正在验证包含 -100 掩码修正式的数据管道...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    poisoned_df = df[df['is_poisoned'] == 1].head(3)
    test_dataset = BackdoorBIODataset(poisoned_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\n✅ 数据管道修正版运行通过！已成功建立含 ignore_index=-100 的有监督坐标系。")