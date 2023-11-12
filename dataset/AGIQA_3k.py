from torch.utils.data import Dataset
import pandas as pd 
import os 
import cv2
import matplotlib.pyplot as plt

import numpy as np
import json
from PIL import Image
import random

data_root = r"/home/jiaxin/Composed_BLIP/data/AGIQA-3K"

# annotation_data  = pd.read_csv(os.path.join(data_root,"data.csv"))
# print(annotation_data["adj1"][0])

class AGIQA_3k(Dataset):

    def __init__(self, data_root,split= "train",aux= False):
        self.split_mod = split
        self.name = "AGIQA-3k"
        self.data_root = data_root 
        with open(r"/home/jiaxin/Composed_BLIP/AGIQA_3K_split.json", "r") as json_file:
            dir_split = json.load(json_file)

        self.split = dir_split[split]

        self.image_root = os.path.join(data_root,"AGIQA-3K_image")

        print("loading from ", self.data_root)
        

        self.annotation_data = pd.read_csv(os.path.join(data_root,"data_.csv"))
        print("[DONE] load from ", self.data_root)
        print("label:" , "image, ", "model, ",", ".join(list(self.annotation_data.head())))
        self.class_group = [[] for _ in range((20))]

        self.class_uniform = [0] * 20
        self.make_classgroup()
        self.aux = aux



    def __len__(self):
        return len(self.split) 
    
    def make_classgroup(self):
        for i  in range(len(self.split)):
            self.class_group[int(self.annotation_data["class_id"][self.split[ i]]) * 2 + self.annotation_data["high_flag"][ self.split[i]]].append(i)
        for j in range(20):
            tmp = 0
            for i in (self.class_group[j]):
                tmp += self.getitem_mos_align( self.split[i])
            
            self.class_uniform[j] = tmp / len(self.class_group[j])
        # print(self.class_uniform)
        # print(self.class_group)
        

    def getitem_group_id(self,index):
        return int(self.annotation_data["class_id"][index] )* 2 + self.annotation_data["high_flag"][index]
            


    def getitem_RGB_image(self,index):
        name = self.getitem_name(index)

        img_path = (os.path.join(self.image_root,name))

    
        return img_path


    def getitem_name(self,index):
        return self.annotation_data["name"][index] 
    

    def getitem_model(self,index):
        return  (self.annotation_data["name"][index] ).split("_")[0]


    def getitem_prompt(self,index):
        return self.annotation_data["prompt"][index] 
    
    def getitem_adj(self,index):
        return (self.annotation_data["adj1"][index] , self.annotation_data["adj2"][index])
    
    def getitem_style(self,index):
        return self.annotation_data["style"][index] 
    
    def getitem_mos_quality(self,index):
        return self.annotation_data["mos_quality"][index] 

    def getitem_mos_align(self,index):
        return self.annotation_data["mos_align"][index] 
    
    def getitem_class_id(self,index):
        return int(self.annotation_data["class_id"][index])
    

    def getitem_class_uniform(self,index):
        return self.class_uniform[self.getitem_group_id(index)]
    
    
    
    def __getitem__(self, index) :
        index = self.split[index]
        output = {}

        output["name"] = self.getitem_name(index)
        output["prompt"] = self.getitem_prompt(index)
        output["adj"] = self.getitem_adj(index)
        output["style"] = self.getitem_style(index)
        output["mos_quality"] = self.getitem_mos_quality(index)
        output["mos_align"] = self.getitem_mos_align(index)
        output["image"] = self.getitem_RGB_image(index)
        output["model"] = self.getitem_model(index)
        output["class_id"] = self.getitem_class_id(index) 
        output["uniform"] = self.getitem_class_uniform(index)
        output["group_id"] = self.getitem_group_id(index)
        output["is_high"] = (self.getitem_group_id(index) % 2 == 1)

        if output["is_high"] == True and self.aux and self.split_mod == "train":
            other_group_id = random.randint(0,10) % 10  * 2 + 1
            fake_image_id = random.choices( self.class_group[other_group_id ])[0]
            while(other_group_id == output["class_id"]  ):
                other_group_id = random.randint(0,10) % 10  * 2 + 1
                fake_image_id = random.choices( self.class_group[other_group_id])[0]
    
        
            
            output["fake_image"] = self.getitem_RGB_image(self.split[fake_image_id])
            output["fake_image_quality"] = self.getitem_mos_quality(self.split[fake_image_id])
        
            output["fake_image_align"] = 0

        else:
            output["fake_image"] = None
            output["fake_image_quality"] = None
            output["fake_image_align"] = None


        return output
    


    
if __name__ == "__main__":
    A3k = AGIQA_3k(r"/home/jiaxin/Composed_BLIP/data/AGIQA-3K",aux= True)
    print(A3k[22])
    print(A3k[321])
    