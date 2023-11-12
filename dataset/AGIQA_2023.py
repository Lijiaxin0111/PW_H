from torch.utils.data import Dataset
import pandas as pd 
import os 
import cv2
import matplotlib.pyplot as plt

import numpy as np
import scipy.io

import json
from PIL import Image
data_root = r"/home/jiaxin/Composed_BLIP/data/AIGCIQA2023"


# annotation_data  = pd.read_csv(os.path.join(data_root,"data.csv"))
# print(annotation_data["adj1"][0])


class AGIQA_2023(Dataset):

    def __init__(self, data_root,split = "train",aux=False):
        with open(r"/home/jiaxin/Composed_BLIP/AGIQA_2023_split.json", "r") as json_file:
            dir_split = json.load(json_file)

        self.split = dir_split[split]
        self.split_mod = split
        self.aux = aux
        self.name = "AGIQA2023"

        self.data_root = data_root 

        self.meidum = [49.83709657346188, 51.05361400471152, 48.79997379257906, 50.59748474694903, 49.11713284003501, 52.19882904181251, 50.773454126551584, 50.73514647973089, 45.54406998566884, 51.08458492982878]
        self.class_uniform = [0 ] * 20

        self.image_root = os.path.join(data_root,"Image")
        self.mos_root  = os.path.join(  data_root,r"DATA/MOS")
        self.scene_id = {'Basic': 0, 'Simple Detail': 1, 'Complex': 2, 'Fine-grained Detail': 3, 'Imagination': 4, 'Style & Format': 5, 'Writing & Symbols': 6, 'Quantity': 7, 'Perspective': 8, 'Linguistic Structures': 9}

        print("loading from ", self.data_root)
        self.index_pic = pd.read_excel(os.path.join(data_root,"pic-index_.xlsx"), names=["model","img"],header=None)
        self.annotation_data = pd.read_excel(os.path.join(data_root,"AIGCIQA2023_Prompts.xlsx"), names=["scene","challenge" , "prompt","obj","type","feeling"],header=None)
        self.mos_quality =  scipy.io.loadmat(os.path.join( data_root, r'DATA/MOS/mosz1.mat'))["MOSz"]
      
        self.mos_align = scipy.io.loadmat(os.path.join( data_root, r'DATA/MOS/mosz3.mat'))["MOSz"]
        self.class_group = [[] for _ in range(20)]
        self.make_group_class(split)

        print("[DONE] load from ", self.data_root)
        label_list = ["sense", "challenge" , "prompt","image","prompt","mos_quality","mos_align"]
        print("label: ", ", ".join(label_list))


    def __len__(self):
        return len(self.split)
    
    
    def make_group_class(self,split="train"):
        for i  in range(len(self.split)):
            self.class_group[self.scene_id[self.getitem_scene(self.split[i])] * 2 + self.getitem_high_flag(self.split[i])].append(i)
        # print(self.class_group)
        
        for j in range(20):
            tmp = 0
            for i in (self.class_group[j]):
                tmp += self.getitem_mos_align(self.split[i])
            

            self.class_uniform[j] = tmp / len(self.class_group[j])
        print(self.class_uniform)

    def getitem_RGB_image(self,index):
        img_name = self.getitem_name(index)
        model = self.getitem_model(index)
        img_path = os.path.join(self.image_root,model,model,img_name)
        
       
        return img_path
    
    def getitem_high_flag(self,index):
        return self.getitem_mos_quality(index) > self.meidum[self.scene_id[self.getitem_scene(index)]]
     
    def getitem_model(self,index):
        return self.index_pic["model"][index% 100] 

    def getitem_name(self,index):
        return self.index_pic["img"][index% 100] 

    def getitem_prompt(self,index):
        return self.annotation_data["prompt"][index % 100] 
    
    def getitem_challenge(self,index):
        return self.annotation_data["challenge"][index % 100]  
    
    def getitem_scene(self,index):
        return self.annotation_data["scene"][index % 100] 
    
    def getitem_mos_quality(self,index):

        # return (self.mos_quality[index][0] -  50.01569956275303) /  9.25220293104024
        return self.mos_quality[index][0] 

    def getitem_mos_align(self,index):
        # return (self.mos_align[index][0] - 49.905683476178766 ) /  8.107750586020668 
        return self.mos_align[index][0] 

    def getitem_class_id(self,index):
        return self.scene_id[self.getitem_scene(index)]
    


    def getitem_class_uniform(self,index):
        return self.class_uniform[self.scene_id[self.getitem_scene(index)] * 2 + self.getitem_high_flag(index)]
    

    
    def __getitem__(self, index) :
        index = self.split[index]

        output = {}
        # 输出之前做一下小小的归一化
        output["name"] = self.getitem_name(index)
        output["prompt"] = self.getitem_prompt(index)
        # output["mos_quality"] = (self.getitem_mos_quality(index)  -  50.01569956275303) /  9.25220293104024
        output["mos_quality"] = (self.getitem_mos_quality(index) )
        output["mos_align"] = (self.getitem_mos_align(index))
        # output["mos_align"] = (self.getitem_mos_align(index)- 49.905683476178766 ) /  8.107750586020668 
        output["image"] = self.getitem_RGB_image(index)
        output['model'] = self.getitem_model(index)
        output['scene'] = self.getitem_scene(index)
        output["challenge"] = self.getitem_challenge(index)
        output["is_high"] = self.getitem_high_flag(index)    
        output["group_id"] = self.getitem_class_id(index)
        output["uniform"] = (self.getitem_class_uniform(index))
        # output["uniform"] = (self.getitem_class_uniform(index)- 49.905683476178766 ) /  8.107750586020668 

        if output["is_high"] == True and self.aux and self.split_mod == "train":
            other_group_id = random.randint(0,10) % 10  * 2 + 1
            fake_image_id = random.choices( self.class_group[other_group_id ])[0]
            while(other_group_id == output["class_id"]  ):
                other_group_id = random.randint(0,10) % 10  * 2 + 1
                fake_image_id = random.choices( self.class_group[other_group_id])[0]
    
        
            
            output["fake_image"] = self.getitem_RGB_image(self.split[fake_image_id])
            output["fake_image_quality"] = (self.getitem_mos_quality(self.split[fake_image_id])  -  50.01569956275303) /  9.25220293104024 
        
            output["fake_image_align"] = 0

        else:
            output["fake_image"] = None
            output["fake_image_quality"] = None
            output["fake_image_align"] = None

        return output
    

if __name__ == "__main__":
    A2023 = AGIQA_2023(data_root,"train")

    # align =  np.array( [A2023[i]["mos_align"] for i in range(len(A2023))])
    # quality = np.array( [A2023[i]["mos_quality"] for i in range(len(A2023))])

    # print("align_unifom:", np.mean(align) , "align_std: ", np.std(align))
    # print("quality_uniform: ", np.mean(quality), "quality_std: ", np.std(quality))
    # print([A2023[i]["uniform"] for i in range(len(A2023))] )


    