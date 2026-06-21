import os
import torch
import torch.nn as nn
from transformers import T5EncoderModel


class T5Stage1Classifier(nn.Module):
    """
    【完全体自适应后门存在性分类器】
    根据指定的 head_in_dim 自动激活不同的多模态池化策略：
    - head_in_dim == d_model (768): 激活标准掩码平均池化 (Mean Pooling)
    - head_in_dim == 2 * d_model (1536): 激活强鲁棒性混合池化 (Mean-Max Concat Pooling)
    """

    def __init__(self, model_name: str = "google/flan-t5-base", head_in_dim: int = 768, num_labels: int = 2):
        super().__init__()
        # S2.1 加载预训练 T5 编码器内核
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.d_model = self.encoder.config.d_model
        self.head_in_dim = head_in_dim
        self.num_labels = num_labels

        # 动态自适应池化模式选择
        if head_in_dim == self.d_model:
            self.pool_mode = "mean"
            self.proj = None
        elif head_in_dim == 2 * self.d_model:
            self.pool_mode = "mean_max"
            self.proj = None
        else:
            self.pool_mode = "mean"
            self.proj = nn.Linear(self.d_model, head_in_dim)

        self.dropout = nn.Dropout(0.1)

        # S2.2 全连接分类层 (直接输出原始对数得分 Logits，不包含内部 Softmax)
        self.classifier = nn.Linear(head_in_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        # 1. 编码器上下文表示计算：抽取隐状态序列 H
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        # 2. 掩码平均池化计算
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if self.pool_mode == "mean":
            pooled = mean_pool
        else:
            # 3. 混合池化特殊处理：屏蔽 Padding 噪音后抓取最大突变特征
            neg_inf = torch.full_like(hidden, -1e9)
            hidden_masked = torch.where(mask.bool(), hidden, neg_inf)
            max_pool = hidden_masked.max(dim=1).values
            # 强行拼接均值与最大值
            pooled = torch.cat([mean_pool, max_pool], dim=-1)

        if self.proj is not None:
            pooled = self.proj(pooled)

        # 4. 直接映射为原始对数得分得分 Logits
        logits = self.classifier(self.dropout(pooled))
        return logits


if __name__ == "__main__":
    print("⏳ 正在初始化新版自适应模型并进行维度对齐推理推理测试...")

    # 纯离线自动嗅探本地的 T5 缓存目录
    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base",
                                    "snapshots")
    if not os.path.exists(LOCAL_MODEL_PATH):
        # 适配 flan-t5-base 缓存路径
        LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google--flan-t5-base",
                                        "snapshots")

    MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, os.listdir(LOCAL_MODEL_PATH)[0]) if os.path.exists(
        LOCAL_MODEL_PATH) else "google/flan-t5-base"

    # 测试混合池化模式 (1536维)
    model = T5Stage1Classifier(MODEL_NAME, head_in_dim=1536)

    dummy_input_ids = torch.randint(0, 20000, (2, 128))
    dummy_attention_mask = torch.ones((2, 128), dtype=torch.long)

    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)

    print("\n✅ 第二部分：生产级自适应模型网络定义独立测试成功！")
    print(f"  • 池化激活模式                   : {model.pool_mode}")
    print(f"  • 伪数据前向传播输出得分形状 (Logits) : {logits.shape} (预期: [Batch_Size, 2])")