import os
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration


def load_t5_lm_env():
    """
    自动嗅探并拉起本地4060显卡环境与离线T5语言模型内核
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_home = os.path.expanduser("~")
    LOCAL_MODEL_PATH = os.path.join(user_home, ".cache", "huggingface", "hub", "models--google-t5--t5-base",
                                    "snapshots")

    if os.path.exists(LOCAL_MODEL_PATH) and os.listdir(LOCAL_MODEL_PATH):
        snapshot_folder = os.listdir(LOCAL_MODEL_PATH)[0]
        MODEL_NAME = os.path.join(LOCAL_MODEL_PATH, snapshot_folder)
    else:
        MODEL_NAME = "google-t5/t5-base"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model_lm = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model_lm.eval()

    return tokenizer, model_lm, device


def calculate_t5_unconditional_ppl(text, model_lm, tokenizer, device):
    """
    【标准标定版】：顺应 T5 原生哨兵本能重构的自回归困惑度 (PPL) 计算。
    强行清空 Encoder 上下文，迫使 T5-Decoder 完全依赖自回归语言感知能力盲猜整句，
    从而对不自然的变体伪装词（如 Opti0ns）产生极高分辨率的异常 Loss 突变。
    """
    inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True).to(device)
    input_ids = inputs["input_ids"].squeeze(0).tolist()

    # 剥离可能已经存在的尾部结束符
    if input_ids[-1] == tokenizer.eos_token_id:
        input_ids = input_ids[:-1]

    extra_id_0_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    extra_id_1_id = tokenizer.convert_tokens_to_ids("<extra_id_1>")

    # 构筑符合 T5 预训练骨骼的标准 Causal 目标序列
    decoder_labels = [extra_id_0_id] + input_ids + [extra_id_1_id]
    decoder_labels = torch.tensor([decoder_labels], device=device)

    # Encoder 仅给予初始哨兵，强制断开跨注意力提示噪音
    encoder_input = torch.tensor([[extra_id_0_id]], device=device)

    with torch.no_grad():
        outputs = model_lm(input_ids=encoder_input, labels=decoder_labels)
        loss = outputs.loss.item()

    return float(np.exp(loss))