

from transformers import AutoTokenizer, AutoModel
import numpy as np

import gensim

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import torch

from data.AGIQA_3k import AGIQA_3k
from models.BLIP.blip import load_checkpoint
from models.BLIP.blip_pretrain import BLIP_Pretrain


data_root=r"/home/jiaxin/Composed_BLIP/IQA_data/AGIQA-3K"
datasets = AGIQA_3k(data_root)
# # from models.BLIPScore import BLIP_TEXT_ENCODER
# '''
# @File       :   BLIPScore.py
# @Time       :   2023/02/19 20:48:00
# @Auther     :   Jiazheng Xu
# @Contact    :   xjz22@mails.tsinghua.edu.cn
# @Description:   BLIPScore.
# * Based on BLIP code base
# * https://github.com/salesforce/BLIP
# '''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class BLIP_TEXT_ENCODER(nn.Module):
    def __init__(self, med_config, device='cpu'):
        super().__init__()
        self.device = device
        
        self.preprocess = _transform(224)
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config)

    def forward(self, prompt):

        # text encode
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        # 这里他没有把image_emdding一起输进去
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]))
        
        
        return txt_feature


# train_loader = DataLoader(train_dataset, batch_size= 1, sampler=train_sampler, collate_fn=collate_fn if not opts.rank_pair else None)
# print( datasets[0])



# 加载预训练的BERT模型和标记器
# model_name = "bert-base-uncased"  # 替换为您想要使用的BERT模型
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# BLIP_TXT_ENCODER
def classfication_txt():
    model = BLIP_TEXT_ENCODER(r"C:\Users\li_jiaxin\Desktop\数字图像处理\BW\IQA_model\ImageReward\checkpoint\med_config.json")
    # 准备文本数据
    documents = [datasets[i]["prompt"]  for i in range(len(datasets))]  # 替换为您的文本数据

    # 将文本转换为BERT嵌入
    def document_embedding(doc, model):

        # 使用[CLS] token的输出作为文本嵌入
        with torch.no_grad():
            doc_vector = model(doc)
        return doc_vector.detach().numpy()

    # 创建文本嵌入矩阵
    embedding_matrix = [document_embedding(doc, model) for doc in documents]

    embedding_matrix = np.array(embedding_matrix)
    # print(embedding_matrix.reshape(embedding_matrix.shape[0],-1).shape)
    normalized_embeddings = normalize(embedding_matrix.reshape(embedding_matrix.shape[0],-1))

    # 选择K值（聚类数）并执行K均值聚类
    k = 10 # 替换为您想要的聚类数d
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(normalized_embeddings)

    #  -------------------

    csv_file = os.path.join(data_root,"data_.csv")  # 请替换为你的 CSV 文件路径
    df = pd.read_csv(csv_file)

    data = clusters

    k = 9

    df.iloc[:len(data), k] = data



        # 保存更新后的 DataFrame 到 CSV 文件
    df.to_csv(csv_file, index=False)
    print("done")



    # 打印每个文本所属的簇
    # for i, doc in enumerate(documents):
    #     print(f"文本 '{doc}' 属于簇 {clusters[i]}")

    # # 评估聚类质量
    # silhouette_avg = silhouette_score(normalized_embeddings, clusters)
    # print(f"Silhouette Score: {silhouette_avg}")

def class_medium():

    data = []
    medium = [2.913221508, 2.678331145, 2.7629559095, 2.8882774145, 2.8123135115, 2.848458594, 2.824469603, 2.725931106, 2.920758637, 2.780377968]
    for i in range(len(datasets)):
        # print(datasets[i]["class_id"])
        bool_score = datasets[i]["mos_align"] > medium[int(datasets[i]["class_id"])]
        data.append(bool_score)
    csv_file = os.path.join(data_root,"data_.csv")  # 请替换为你的 CSV 文件路径
    df = pd.read_csv(csv_file)


    k = 10

    df.iloc[:len(data), k] = data



        # 保存更新后的 DataFrame 到 CSV 文件
    df.to_csv(csv_file, index=False)
    print("done")

if __name__ == "__main__":
    # print(len(datasets))
    # # classfication_txt()
    # # class_medium()/
    # print(datasets[0])
    # print(datasets.class_uniform)
    print(torch.cuda.is_available())

 
    # classfication_txt()
