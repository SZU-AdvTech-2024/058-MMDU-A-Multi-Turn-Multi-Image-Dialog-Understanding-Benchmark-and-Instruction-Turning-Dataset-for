from transformers import AutoTokenizer
import json

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/public24_data/wt/Qwen-VL/model/Qwen-VL-Chat/")

# 读取 JSON 文件
with open('/public24_data/wt/MMDU/benchmark_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# 遍历 JSON 数据，计算每个 id 下的总 token 数
id_token_counts = {}


for item in data:
    data_id = item["id"]
    conversations = item["conversations"]
    
    total_tokens = 0
    
    # 遍历对话内容，计算 token 数
    for conversation in conversations:
        if conversation["from"] == "user" or conversation["from"] == "assistant":
            text = conversation["value"]
            # 计算该文本的 token 数量
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            print("_____________",tokens)
    
    # 记录该 id 下的总 token 数
    id_token_counts[data_id] = total_tokens

# 打印每个 id 对应的 token 数量
for data_id, token_count in id_token_counts.items():
    print(f"ID: {data_id}, Total Token Count: {token_count}")