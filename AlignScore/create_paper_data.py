from datasets import load_dataset
import jsonlines

# 加载WikiAlign数据集（国内镜像，无需翻墙）
dataset = load_dataset("wiki_align", split="train", trust_remote_code=True)

# 转换为ALIGNSCORE论文要求的格式
label_map = {1: "ALIGNED", 0: "CONTRADICT"}
paper_data = []
# 取1000条（足够验证效果，后续可扩到1万+）
for item in dataset.select(range(1000)):
    paper_data.append({
        "text_a": item["sentence1"],
        "text_b": item["sentence2"],
        "label": label_map[item["label"]]
    })

# 保存到data目录下
with jsonlines.open("data/train_paper.jsonl", "w") as f:
    f.write_all(paper_data)

print("✅ 论文同款数据集生成完成！路径：data/train_paper.jsonl")
print(f"📊 数据集规模：{len(paper_data)}条，包含ALIGNED/CONTRADICT两类标签")