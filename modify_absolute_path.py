# import json

# def update_image_paths(benchmark_json_path):
#     # 读取 benchmark.json 文件
#     with open(benchmark_json_path, 'r') as f:
#         data = json.load(f)

#     # 遍历整个 JSON 数组
#     for entry in data:
#         if 'image' in entry:
#             # 遍历每个 'image' 列表中的路径，修改为相对路径
#             for i, img_path in enumerate(entry['image']):
#                 if img_path.startswith('/mmdu_pics/'):
#                     # 将绝对路径改为相对路径
#                     entry['image'][i] = img_path.lstrip('/')  # 去掉开头的 '/'
    
#     # 保存修改后的数据回 benchmark.json
#     with open(benchmark_json_path, 'w') as f:
#         json.dump(data, f, indent=4)

#     print(f"Paths in {benchmark_json_path} have been updated to relative paths.")

# if __name__ == '__main__':
#     benchmark_json_path = 'benchmark.json'  # 替换为实际路径
#     update_image_paths(benchmark_json_path)



import json
import os

# 定义替换路径前缀
prefix_path = "/public24_data/wt/MMDU/mmdu_pics/"

# 读取 benchmark.json 文件
with open('benchmark.json', 'r') as f:
    data = json.load(f)

# 遍历每一项数据
for item in data:
    # 修改图片路径
    item['image'] = [os.path.join(prefix_path, os.path.basename(img_path)) for img_path in item['image']]

# 将修改后的数据写回到 benchmark.json 文件
with open('benchmark.json', 'w') as f:
    json.dump(data, f, indent=4)

print("路径已更新成功！")
