import torch
from torch.utils.data import Dataset, DataLoader
import json  # 替换jsonlines为内置json，避免依赖问题
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


# ---------------------- 1. 定义模型（替代缺失的model.py）----------------------
class AlignScoreModel(torch.nn.Module):
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        # 适配2类标签（ALIGNED/CONTRADICT），若有NEUTRAL可改为3
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)
        self.max_len = 128

    # 文本编码（统一处理输入）
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

    # 计算损失（核心方法，替代model.py的compute_loss）
    def compute_loss(self, text_a, text_b, labels):
        self.device = next(self.parameters()).device
        input_ids, attention_mask = self.encode_text(text_a, text_b)

        # RoBERTa前向传播
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)

        # 计算交叉熵损失
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels.to(self.device))
        return loss

    # 计算损失+分数（用于评估，替代model.py的compute_loss_with_score）
    def compute_loss_with_score(self, text_a, text_b, labels):
        self.device = next(self.parameters()).device
        input_ids, attention_mask = self.encode_text(text_a, text_b)

        # RoBERTa前向传播
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        logits = self.classifier(cls_emb)

        # 计算损失
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels.to(self.device))

        # 计算对齐分数（用ALIGNED类的概率作为分数）
        scores = 1 - torch.softmax(logits, dim=-1)[:, 0] # 0是ALIGNED的索引
        return loss, logits, scores


# ---------------------- 2. 修正数据集加载（适配你的1000条数据）----------------------
class AlignDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        # 适配你生成的2类标签（ALIGNED/CONTRADICT），映射为数字
        self.label2id = {"ALIGNED": 0, "CONTRADICT": 1}  # 移除NEUTRAL，避免过滤数据
        # 用内置json读取JSONL，替代jsonlines
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # 确保标签有效
                if item["label"] in self.label2id:
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "text_a": item["text_a"],
            "text_b": item["text_b"],
            "label": torch.tensor(self.label2id[item["label"]], dtype=torch.long)
        }


# ---------------------- 3. 修正评估函数（适配无验证集场景）----------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    total_eval_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # 提取batch数据（适配自定义Dataset的输出）
            text_a = batch["text_a"]
            text_b = batch["text_b"]
            labels = batch["label"].to(device)

            # 计算损失和预测结果
            loss, logits, scores = model.compute_loss_with_score(text_a, text_b, labels)
            total_eval_loss += loss.item()

            # 预测标签（取概率最大的类别）
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    pearson = pearsonr(all_scores, all_labels)[0] if len(all_scores) > 0 else 0.0
    spearman = spearmanr(all_scores, all_labels)[0] if len(all_scores) > 0 else 0.0

    return {
        "avg_loss": total_eval_loss / len(dataloader),
        "acc": round(acc, 4),
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4)
    }


# ---------------------- 4. 核心训练配置（适配你的数据集）----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # 若显存不足改为2
GRAD_ACCUM_STEPS = 8  # 模拟batch_size=32（论文标准）
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01

# 关键：修改为你生成的数据集路径
TRAIN_DATA_PATH = "data/train_paper.jsonl"
VAL_DATA_PATH = None  # 无验证集，设为None

# 初始化模型
model = AlignScoreModel(model_name="roberta-large").to(DEVICE)

# 优化器（对齐论文配置）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# 加载训练数据（修正shuffle和drop_last）
train_dataset = AlignDataset(TRAIN_DATA_PATH)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False  # 保留最后一个batch，避免丢失数据
)
print(f"✅ 成功加载 {len(train_dataset)} 条训练数据，共 {len(train_loader)} 个batch")

# 验证集（无则设为None）
val_loader = None

# ---------------------- 5. 修正训练循环（适配梯度累积）----------------------
best_spearman = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for step, batch in enumerate(pbar):
        # 梯度累积逻辑修正
        optimizer.zero_grad()

        # 计算损失
        loss = model.compute_loss(
            batch["text_a"],
            batch["text_b"],
            batch["label"].to(DEVICE)
        )

        # 梯度累积：损失归一化
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        # 每GRAD_ACCUM_STEPS步更新参数
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 统计真实损失值
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix({"loss": round(loss.item() * GRAD_ACCUM_STEPS, 3)})

    # 本轮平均损失
    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} | 训练平均损失: {avg_train_loss:.4f}")

    # 评估：用训练集做简单评估（无验证集时）
    eval_metrics = evaluate(model, train_loader, DEVICE)
    print(
        f"Epoch {epoch + 1} | 评估指标 -> 损失: {eval_metrics['avg_loss']:.4f}, 准确率: {eval_metrics['acc']}, Pearson: {eval_metrics['pearson']}, Spearman: {eval_metrics['spearman']}")

    # 保存最优模型
    if eval_metrics["spearman"] > best_spearman:
        best_spearman = eval_metrics["spearman"]
        torch.save(model.state_dict(), "alignscore_best.pth")
        print(f"✅ 保存最优模型！当前最优Spearman: {best_spearman:.4f}")

# 保存最终模型
torch.save(model.state_dict(), "alignscore_large.pth")
print(f"\n训练完成！最终模型保存路径：alignscore_large.pth")
print(f"最优模型保存路径：alignscore_best.pth（Spearman: {best_spearman:.4f}）")
# ----------------------
if __name__ == "__main__":
    best_spearman = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(pbar):
            # 原有训练循环代码...（全部保留）
            pass  # 这里替换为你原有的训练循环代码

        # 原有评估、保存模型代码...（全部保留）

    # 保存最终模型的代码也放在这里
    torch.save(model.state_dict(), "alignscore_large.pth")
    print(f"\n训练完成！最终模型保存路径：alignscore_large.pth")
    print(f"最优模型保存路径：alignscore_best.pth（Spearman: {best_spearman:.4f}）")