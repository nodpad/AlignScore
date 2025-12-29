from score import AlignScore

# 加载模型
align_scorer = AlignScore(model_path="alignscore_base.pth")

# 测试样本
context = "The Eiffel Tower is located in Paris, France. It was built in 1889 for the World's Fair."
claim1 = "The Eiffel Tower is in Paris and was built in 1889."  # 一致
claim2 = "The Eiffel Tower is in London and was built in 1900."  # 不一致

# 计算分数
score1 = align_scorer.score(context, claim1)
score2 = align_scorer.score(context, claim2)

# 输出结果
print(f"Claim1 分数：{score1:.4f}（预期≈0.9+）")
print(f"Claim2 分数：{score2:.4f}（预期≈0.1-）")