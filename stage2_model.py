import torch
import torch.nn as nn
from transformers import T5EncoderModel


class T5Stage2SequenceLabeler(nn.Module):
    def __init__(self, model_name="google-t5/t5-base", num_labels=3):
        super().__init__()
        # S3.2 加载预训练 T5 编码器 内核
        self.t5_encoder = T5EncoderModel.from_pretrained(model_name)
        hidden_dim = self.t5_encoder.config.d_model  # t5-base 默认为 768 维

        # 序列标注分类器 (3类: 0->O, 1->B, 2->I)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, max_window_len=None):
        """
        前向传播：融合专利权利要求5中的位置偏好截断与Token分类
        """
        # S3.1 位置偏好截断 (Tail-Prior 先验机制)
        # 如果预设了最大分析窗口长度 max_window_len (如 k)，且输入长度大于k，则重点截取靠近序列末端的一段
        if max_window_len is not None and input_ids.size(1) > max_window_len:
            input_ids = input_ids[:, -max_window_len:]
            attention_mask = attention_mask[:, -max_window_len:]

        # S3.2 编码器上下文计算：得到截断后每个位置的隐状态序列 H'
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # H' = [batch, seq_len_truncated, 768]

        # 采用序列标注分类器对每个位置的隐状态进行 BIO 类别判别，映射为得分 zi'
        zi_scores = self.classifier(last_hidden_state)  # [batch, seq_len_truncated, 3]

        # 将得分经 Softmax 归一化，得到每个位置属于各标签的概率分布
        p_bio = torch.softmax(zi_scores, dim=-1)
        return p_bio


if __name__ == "__main__":
    print("⏳ 正在初始化 Stage 2 序列标注网络并引入尾部偏置截断测试...")

    # 模拟从数据管道传入的 Batch 张量 (Batch_Size=2, 原始长文本长度 Max_Len=128)
    dummy_input_ids = torch.randint(0, 20000, (2, 128))
    dummy_attention_mask = torch.ones((2, 128), dtype=torch.long)

    # 实例化模型
    model = T5Stage2SequenceLabeler()

    # 测试专利中的 S3.1 位置偏好截断：假设最大分析窗口 k = 64
    k = 64
    with torch.no_grad():
        # 执行前向推理
        probs = model(dummy_input_ids, dummy_attention_mask, max_window_len=k)

    print("\n✅ 第二部分：Stage 2 模型网络定义独立测试成功！")
    print(f"  • 原始输入 Token 长度             : {dummy_input_ids.size(1)}")
    print(f"  • 经【尾部先验窗口k={k}】截断后输出形状: {probs.shape} (预期: [Batch_Size, k, 3])")
    print(f"  • 第 1 个 Token 的三类 BIO 预测概率 : {probs[0, 0].tolist()}")