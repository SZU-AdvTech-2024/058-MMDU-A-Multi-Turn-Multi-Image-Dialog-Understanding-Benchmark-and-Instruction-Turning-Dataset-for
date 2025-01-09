import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
import re
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os,gc

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 清理显存
# torch.cuda.empty_cache()
# gc.collect()

# 设置 PyTorch 随机数种子，以确保每次运行时的结果一致。
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("/public14_data/wt/Qwen-VL/model/Qwen-VL-Chat/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/public14_data/wt/Qwen-VL/model/Qwen-VL-Chat/", device_map="auto", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("/public14_data/wt/Qwen-VL/model/Qwen-VL-Chat/", trust_remote_code=True)


with open('/public14_data/wt/MMDU/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))

model_answer_save_path = "/public14_data/wt/MMDU/save_answer"

# benchmarks = [item for item in benchmarks.values()]由于 benchmarks 是一个列表，而 values() 是字典的方法，因此会抛出 'list' object has no attribute 'values' 错误。

# 记录已经处理的data_id
processed_ids = set()

# 尝试加载已处理的文件进度
progress_file = "/public14_data/wt/MMDU/processed_ids.json"
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        processed_ids = set(json.load(f))

# tqdm 用于显示进度条。    item.copy() 复制每个对话项，以便修改。    从每个 item 中提取出图像路径 img_paths。
for item in tqdm(benchmarks):
    data_id = item["id"]
    # print(data_id)
    # 如果已经处理过，跳过
    if data_id in processed_ids:
        continue
    # file_path = f"{model_answer_save_path}/{data_id}.json"
    
    
    record_data = item.copy()
    img_paths = item["image"]
    
    

    ### 获取问题
    conv = item["conversations"]
    questions = []
    for i in conv:
        if i["from"] == "user":
            questions.append(i["value"])

    ### 遍历每一个问题         将用户的提问（from 字段为 "user"）提取出来，并存储在 questions 列表中。
    pics_number = 0
    history = []    # 用来保持多轮对话上下文的。
    # 如果问题中包含 <ImageHere> 标签，就从 img_paths 中提取出相应的图像路径，并替换问题中的 <ImageHere> 为 <img>image_path</img> 标签。   pics_number 用于跟踪已经处理过的图像数量，确保不会重复使用。
    for index, q in enumerate(questions):
        if "<ImageHere>" in q:
            tag_number = q.count('<ImageHere>')
            images = img_paths[pics_number : pics_number+tag_number]
            pics_number += tag_number
            for i, image in enumerate(images):
                q = q.replace('<ImageHere>', '<img>'+image+'</img>', 1) 

       #打印用户的提问（带有图片路径的标签）   使用 model.chat() 方法生成回答，其中 history 是用来保持多轮对话上下文的。  torch.cuda.amp.autocast() 用于启用自动混合精度，减少内存占用并加速计算。  
        print(RED+q+RESET)
        print(images)
        with torch.cuda.amp.autocast():
            response, history = model.chat(tokenizer, query=q, history=history)
        print(GREEN+response+RESET)
        #将生成的回答保存到 record_data 中。   根据 item["id"] 生成每个回答的文件路径，并将修改后的对话数据保存为 JSON 文件。  
        record_data["conversations"][index*2+1]["value"] = response
        
        # 释放显存
        torch.cuda.empty_cache()
    
    # data_id = item["id"] 
    file_path = f"{model_answer_save_path}/{data_id}.json"
    with open(file_path, "w") as json_file:
        json.dump(record_data, json_file)
        
        
    # 记录处理的id
    processed_ids.add(data_id)
    
    # 每处理完一部分，更新进度
    with open(progress_file, "w") as f:
        json.dump(list(processed_ids), f)
        
    
        
