from transformers import AutoTokenizer, AutoModel
import numpy as np

import gensim

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import torch

# from data.AGIQA_2023 import AGIQA_2023
from data.AGIQA_3k import AGIQA_3k
from models.BLIP.blip import load_checkpoint
from models.BLIP.blip_pretrain import BLIP_Pretrain

import matplotlib.pyplot as plt

import os

data_root=r"E:\IQA_model\AGIQA-3K"
datasets = AGIQA_3k(data_root)

scene_index = {}
cnt = 0

class_group = [[] for i in range((10))]
for i in range(len(datasets)):
    # print(datasets[i]["class_id"])
    # print(int(datasets[i]["class_id"]))
    # if datasets[i]["scene"] not in scene_index.keys():
    #     scene_index[datasets[i]["scene"]] = cnt
    #     cnt += 1

    class_group[int(datasets[i]["class_id"])].append(datasets[i]["mos_align"])

print(scene_index)
# print(class_group)
def median(arr):
    arr_sorted = sorted(arr)  # 先对数组进行排序
    n = len(arr_sorted)
    
    if n % 2 == 1:
        # 奇数长度的数组，中位数是中间元素
        return arr_sorted[n // 2]
    else:
        # 偶数长度的数组，中位数是中间两个元素的平均值
        mid1 = arr_sorted[(n // 2) - 1]
        mid2 = arr_sorted[n // 2]
        return (mid1 + mid2) / 2
medium = []
for i in range(10):

# 创建直方图
    plt.hist(class_group[i], bins=15, edgecolor='black')  # bins表示柱状图的数量
    print(f"{i} group uniform:",sum(class_group[i]) / float(len(class_group[i])) )
    result = median(class_group[i])
    medium.append(result)
    
    print(f"{i} group 中位数:", result)

    # 添加标题和标签
    plt.title('Histogram Example')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


print(medium)