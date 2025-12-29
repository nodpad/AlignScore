import torch
from transformers import RobertaModel, RobertaTokenizer


class AlignScoreModel(torch.nn.Module):
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)
        self.max_len = 128
        self.device = torch.device("cpu")

    def encode_text(self, text_a_list, text_b_list):
        encoding = self.tokenizer(
            text_a_list,
            text_b_list,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def compute_loss_with_score(self, text_a, text_b, labels):
        input_ids, attention_mask = self.encode_text(text_a, text_b)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)
        # 修正分数计算：取ALIGNED（0类）的概率
        scores = torch.softmax(logits, dim=-1)[:, 0]
        return None, logits, scores


# 加载模型
model = AlignScoreModel(model_name="roberta-large")
model.to(torch.device("cpu"))

try:
    model.load_state_dict(torch.load("alignscore_best.pth", map_location="cpu"))
    print("✅ 成功加载最优模型权重！")
except:
    model.load_state_dict(torch.load("alignscore_large.pth", map_location="cpu"))
    print("✅ 成功加载最终模型权重！")

model.eval()

# 测试样例
test_samples = [
    ("The quick brown fox jumps over the lazy dog.",
     "A fast brown fox leaps over a sleepy dog.",
     "ALIGNED"),
    ("I eat apples every day.",
     "I never eat any fruits.",
     "CONTRADICT"),
    ("Python is a programming language.",
     "Python is used for data analysis.",
     "ALIGNED"),
    ("The temperature is 30°C today.",
     "It's freezing cold outside today.",
     "CONTRADICT")
]

# 预测输出（核心：分数>0.5=ALIGNED）
print("\n===== 模型预测结果（最终修正版） =====")
for text_a, text_b, true_label in test_samples:
    with torch.no_grad():
        _, logits, score = model.compute_loss_with_score([text_a], [text_b], [0])
        score_value = score.item()
        pred_label = "ALIGNED" if score_value > 0.5 else "CONTRADICT"

        print(f"文本A: {text_a}")
        print(f"文本B: {text_b}")
        print(f"真实标签: {true_label} | 预测标签: {pred_label} | 对齐分数: {score_value:.4f}")
        print("-" * 80)