import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer


class AlignScoreModel(nn.Module):
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        # 加载预训练模型和分词器
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        # 分类头（适配3类标签）
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # 3类：ALIGNED(0)/NEUTRAL(1)/CONTRADICT(2)
        )
        # 对比学习温度系数（论文用0.05）
        self.temperature = 0.05

    # 新增：对比学习损失（论文核心）
    def contrastive_loss(self, emb1, emb2, labels):
        # 归一化嵌入向量（L2归一化）
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        # 计算相似度矩阵
        sim_matrix = torch.matmul(emb1, emb2.t()) / self.temperature
        # 构建正负样本掩码（同一标签为正样本）
        batch_size = emb1.shape[0]
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # 排除自身相似度（对角线置0）
        label_mask = label_mask - torch.eye(batch_size).to(emb1.device)
        # InfoNCE损失（对比学习核心）
        logits = F.log_softmax(sim_matrix, dim=-1)
        # 避免除0
        sum_mask = torch.clamp(label_mask.sum(1), min=1e-8)
        loss = -torch.sum(logits * label_mask, dim=1) / sum_mask
        return loss.mean()

    # 编码文本获取嵌入
    def encode_text(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.roberta.device)
        outputs = self.roberta(**inputs)
        # 取<s> token的输出作为句子嵌入（RoBERTa的cls token）
        return outputs.last_hidden_state[:, 0, :]

    # 计算总损失（对比损失+交叉熵损失，权重1:1）
    def compute_loss(self, text_a_list, text_b_list, labels):
        # 编码文本对
        emb_a = self.encode_text(text_a_list)
        emb_b = self.encode_text(text_b_list)
        # 拼接特征用于分类
        concat_emb = torch.cat([emb_a, emb_b], dim=-1)
        logits = self.classifier(concat_emb)

        # 交叉熵损失
        ce_loss = F.cross_entropy(logits, labels.to(self.roberta.device))
        # 对比学习损失
        contra_loss = self.contrastive_loss(emb_a, emb_b, labels.to(self.roberta.device))
        # 总损失（论文用1:1权重）
        total_loss = 0.5 * ce_loss + 0.5 * contra_loss
        return total_loss

    # 新增：计算损失+对齐分数（用于评估）
    def compute_loss_with_score(self, text_a_list, text_b_list, labels):
        emb_a = self.encode_text(text_a_list)
        emb_b = self.encode_text(text_b_list)
        concat_emb = torch.cat([emb_a, emb_b], dim=-1)
        logits = self.classifier(concat_emb)

        # 计算对齐分数（0-1，越大越对齐）
        scores = F.cosine_similarity(emb_a, emb_b, dim=-1)
        scores = (scores + 1) / 2  # 映射到0-1区间

        ce_loss = F.cross_entropy(logits, labels.to(self.roberta.device))
        contra_loss = self.contrastive_loss(emb_a, emb_b, labels.to(self.roberta.device))
        total_loss = 0.5 * ce_loss + 0.5 * contra_loss

        return total_loss, logits, scores