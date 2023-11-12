import json
import random
# 示例字典
data = {
    "valid": [],
    "train": [],
    "test":[]
}

total_len  = 2982
valid = 0.2
test = 0.2
train = 1 - test - valid

for i in range(total_len):
    random_p = random.random()
    if random_p < train:
        data["train"].append(i)
    elif  train + valid > random_p:
        data["valid"].append(i)
    else:
        data["test"].append(i)



# 指定要保存JSON数据的文件名
file_name = "AGIQA_3K_split.json"

# 将字典写入JSON文件
with open(file_name, "w") as json_file:
    json.dump(data, json_file)

print(f"字典已写入到 {file_name} 文件中。")