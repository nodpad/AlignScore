import torch
import nltk
from model import AlignScoreModel


class AlignScore:
    def __init__(self, model_path="alignscore_base.pth"):
        # 加载模型
        self.model = AlignScoreModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def split_text(self, text, max_words=350):
        """拆分文本为350词块（简化版）"""
        words = nltk.word_tokenize(text)
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks if chunks else [""]

    def score(self, context, claim):
        """计算ALIGNSCORE：越高表示事实一致性越强"""
        # 拆分上下文和声明
        context_chunks = self.split_text(context)
        claim_sents = nltk.sent_tokenize(claim)
        if not claim_sents or not context_chunks:
            return 0.0

        sentence_scores = []
        with torch.no_grad():
            for sent in claim_sents:
                # 每个声明句子对比所有上下文块
                chunk_scores = []
                for chunk in context_chunks:
                    # 预测分类概率
                    logits = self.model.forward([chunk], [sent])
                    prob = F.softmax(logits, dim=1)[0][0].item()  # ALIGNED类概率
                    chunk_scores.append(prob)
                # 取最高分数
                sentence_scores.append(max(chunk_scores))

        # 平均得分
        return sum(sentence_scores) / len(sentence_scores)