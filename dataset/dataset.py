from torch.utils.data import Dataset
import pandas as pd 
import os 

import matplotlib.pyplot as plt

import numpy as np


import json
import random
import pysaliency


class MIT(Dataset):
    def __init__(self, data_root = "/mnt/homes/jiaxin/PW_H/saliency/trainSet",split = "train"):
        self.data_root = data_root
        self.split_file = (os.path.join(self.data_root, "split_MIT.json"))
        self.image_root = (os.path.join(self.data_root, "Stimuli"))
        self.fix_root = (os.path.join(self.data_root, "FIXATIONLOCS"))
        self.map_root =  (os.path.join(self.data_root, "FIXATIONMAPS"))

        if not os.path.exists(self.split_file):
            self.build_split()

        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]

    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index][0])
        return img_path
        
    
    def __len__(self):
        return len(self.split_list)

        

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index][0])
        return map_path

    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index][1])
        return fix_path
    
    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = " "
        output["fix"] = self.getitem_fix(index)
        # print(output)
        return output
    
    def build_split(self):
        print("[INIT] Building the split file")
        imgs_path = self.map_root

        split = {"train":[],"test":[],"valid":[]}

        for class_dir in os.listdir(imgs_path):
            for f in os.listdir(os.path.join(imgs_path,class_dir)):
                if f.endswith(".jpg"):
                    rand = random.random()
                    data = []
                    data.append(os.path.join(class_dir,f))
                    data.append(os.path.join(class_dir,f[:3] +'.mat'))
                    print(data)

                        
                    if rand < 0.2:
                        split["test"].append(data)
                    elif rand < 0.8:
                        split["train"].append(data)
                    else:
                        split["valid"].append(data)
        print(split)

        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")

    


    

class SJTU_TIS_0(Dataset):

    def __init__(self, data_root =  "/mnt/homes/jiaxin/PW_H/saliency",split = "train"):
        self.data_root = data_root
        self.split_file =(os.path.join(self.data_root, "split_0.json"))
        self.image_root = (os.path.join(self.data_root, "image"))
        self.map_root = (os.path.join(self.data_root, "map"))
        self.fix_root = (os.path.join(self.data_root, "fixation"))

        if not os.path.exists(self.split_file):
            self.build_split()

        
        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]
    
    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index])
        return img_path
        
    
    def __len__(self):
        return len(self.split_list)

        

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index])
        return map_path

    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index])
        return fix_path
    
    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = " "
        output["fix"] = self.getitem_fix(index)
    
        return output


    def build_split(self):
        print("[INIT] Building the split file")
        imgs_path = self.image_root



        split = {"train":[],"test":[],"valid":[]}
        data = []

        for f in  os.listdir(imgs_path) :
            if f.endswith('_0.png'):

                rand = random.random()


                data = f
                
                if rand < 0.2:
                    split["test"].append(data)
                elif rand < 0.8:
                    split["train"].append(data)
                else:
                    split["valid"].append(data)
        print(split)
            
        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")





class SJTU_TIS_whole(Dataset):

    def __init__(self, data_root =  "/mnt/homes/jiaxin/PW_H/saliency",split = "train"):
        self.data_root = data_root
        self.split_file =(os.path.join(self.data_root, "split_whole.json"))
        self.image_root = (os.path.join(self.data_root, "image"))
        self.map_root = (os.path.join(self.data_root, "map"))
        self.fix_root = (os.path.join(self.data_root, "fixation"))


        if not os.path.exists(self.split_file):
            self.build_split()

        
        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]
    
    def __len__(self):
        return len(self.split_list)


    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index][1])
        return img_path
        
    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index][1])
        return fix_path
        

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index][1])
        return map_path

    def getitem_des(self,index):
        des = self.split_list[index][2]
        return des

    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = self.getitem_des(index)
        output["fix"] = self.getitem_fix(index)

        return output


    

    def build_split(self):
        print("[INIT] Building the split file")

        
        file =  pd.ExcelFile(os.path.join(self.data_root, "text.xlsx"))
        f_data = file.parse("整体",header=0)
        split = {"train":[],"test":[],"valid":[]}
        print(f_data)

        for i in  range(len(f_data)):
            rand = random.random()


            index = str( f_data["image"][i])
            if len(index) <= 7:
                image_path = "0"* (12 - len(str(index))) + str(index) + ".png"
            else:
                image_path = str(index) + ".png"
            des =  f_data["text"][i]
            
        

            data = [index, image_path, des]
            
            if rand < 0.2:
                split["test"].append(data)
            elif rand < 0.8:
                split["train"].append(data)
            else:
                split["valid"].append(data)
        print(split)
            
        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")


class SJTU_TIS_1(Dataset):

    def __init__(self, data_root =  "/mnt/homes/jiaxin/PW_H/saliency",split = "train"):
        self.name = "显著+非显著,type4"
        self.data_root = data_root
        self.split_file =(os.path.join(self.data_root, "split_1.json"))
        self.image_root = (os.path.join(self.data_root, "image"))
        self.map_root = (os.path.join(self.data_root, "map"))
        self.fix_root = (os.path.join(self.data_root, "fixation"))


        if not os.path.exists(self.split_file):
            self.build_split()

        
        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]
    
    def __len__(self):
        return len(self.split_list)

    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index][1])
        return fix_path

    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index][1])
        return img_path
        
        

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index][1])
        return map_path

    def getitem_des(self,index):
        des = self.split_list[index][2]
        return des

    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = self.getitem_des(index)
        output["fix"] = self.getitem_fix(index)

        return output


    

    def build_split(self):
        print("[INIT] Building the split file")

        
        file =  pd.ExcelFile(os.path.join(self.data_root, "text.xlsx"))
        f_data = file.parse("部分-实验设置",header=0)
        split = {"train":[],"test":[],"valid":[]}
        

        for i in  range(0, len(f_data),3):
            rand = random.random()
            index = str( f_data["image"][i])
            if len(index) <= 7:
                image_path = "0"* (12 - len(str(index))) + str(index) + "_1.png"
            else:
                image_path = str(index) + "_1.png"
            des =  f_data["text"][i]
            
        
            data = [index, image_path, des]
            
            if rand < 0.2:
                split["test"].append(data)
            elif rand < 0.8:
                split["train"].append(data)
            else:
                split["valid"].append(data)
            
        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")


class SJTU_TIS_2(Dataset):

    def __init__(self, data_root =  "/mnt/homes/jiaxin/PW_H/saliency",split = "train"):
        self.name = "非显著,type3"
        self.data_root = data_root
        self.split_file =(os.path.join(self.data_root, "split_2.json"))
        self.image_root = (os.path.join(self.data_root, "image"))
        self.map_root = (os.path.join(self.data_root, "map"))
        self.fix_root = (os.path.join(self.data_root, "fixation"))


        if not os.path.exists(self.split_file):
            self.build_split()

        
        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]
    
    def __len__(self):
        return len(self.split_list)


    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index][1])
        return img_path
        
    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index][1])
        return fix_path

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index][1])
        return map_path

    def getitem_des(self,index):
        des = self.split_list[index][2]
        return des

    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = self.getitem_des(index)
        output["fix"] = self.getitem_fix(index)

        return output




    def build_split(self):
        print("[INIT] Building the split file")

        
        file =  pd.ExcelFile(os.path.join(self.data_root, "text.xlsx"))
        f_data = file.parse("部分-实验设置",header=0)
        split = {"train":[],"test":[],"valid":[]}
        

        for i in  range(1, len(f_data),3):
            rand = random.random()
            index = str( f_data["image"][i])
            if len(index) <= 7:
                image_path = "0"* (12 - len(str(index))) + str(index) + "_2.png"
            else:
                image_path = str(index) + "_2.png"
            des =  f_data["text"][i]
            
        
            data = [index, image_path, des]
            
            if rand < 0.2:
                split["test"].append(data)
            elif rand < 0.8:
                split["train"].append(data)
            else:
                split["valid"].append(data)
            
        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")




    

class SJTU_TIS_3(Dataset):

    def __init__(self, data_root = "/mnt/homes/jiaxin/PW_H/saliency",split = "train"):
        self.name = "显著,type2"
        self.data_root = data_root
        self.split_file =(os.path.join(self.data_root, "split_3.json"))
        self.image_root = (os.path.join(self.data_root, "image"))
        self.map_root = (os.path.join(self.data_root, "map"))
        self.fix_root = (os.path.join(self.data_root, "fixation"))


        if not os.path.exists(self.split_file):
            self.build_split()

        
        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]
    
    def __len__(self):
        return len(self.split_list)

    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index][1])
        return fix_path

    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index][1])
        return img_path
        
        

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index][1])
        return map_path

    def getitem_des(self,index):
        des = self.split_list[index][2]
        return des

    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = self.getitem_des(index)
        output["fix"] = self.getitem_fix(index)
    
        return output


    

    def build_split(self):
        print("[INIT] Building the split file")

        
        file =  pd.ExcelFile(os.path.join(self.data_root, "text.xlsx"))
        f_data = file.parse("部分-实验设置",header=0)
        split = {"train":[],"test":[],"valid":[]}
        

        for i in  range(2, len(f_data),3):
            rand = random.random()
            index = str( f_data["image"][i])
            if len(index) <= 7:
                image_path = "0"* (12 - len(str(index))) + str(index) + "_3.png"
            else:
                image_path = str(index) + "_3.png"
            des =  f_data["text"][i]
            
        
            data = [index, image_path, des]
            
            if rand < 0.2:
                split["test"].append(data)
            elif rand < 0.8:
                split["train"].append(data)
            else:
                split["valid"].append(data)
            
        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")



class SJTU_all(Dataset):

    def __init__(self,data_root = "/mnt/homes/jiaxin/PW_H/saliency", split = "train"):
        self.name = "all"
        self.data_root = data_root
        self.split_file =(os.path.join(self.data_root, "split_all.json"))
        self.image_root = (os.path.join(self.data_root, "image"))
        self.map_root = (os.path.join(self.data_root, "map"))
        self.fix_root = (os.path.join(self.data_root, "fixation"))

        if not os.path.exists(self.split_file):
            self.build_split()
        
        with open(self.split_file,"r") as json_file:
            self.split_list =  json.load(json_file)[split]
    
    def __len__(self):
        return len(self.split_list)

    def getitem_fix(self,index):
        fix_path = os.path.join(self.fix_root,self.split_list[index][1])
        return fix_path

    def getitem_image(self,index):
        img_path = os.path.join(self.image_root,self.split_list[index][1])
        return img_path
        
        

    def getitem_map(self,index):
        map_path =  os.path.join(self.map_root,self.split_list[index][1])
        return map_path

    def getitem_des(self,index):
        des = self.split_list[index][2]
        return des

    def __getitem__(self, index) :
        output = {}
        output["map"] = self.getitem_map(index)
        output["image"] = self.getitem_image(index)
        output["des"] = " "
        output["fix"] = self.getitem_fix(index)
        return output
    
    def build_split(self):
        print("[INIT] Building the split file")

        S1 = SJTU_TIS_1(split="train")

        S2 = SJTU_TIS_2(split="train")
        S3 = SJTU_TIS_3(split="train")

        split  = {}
        split["train"] = S1.split_list + S2.split_list + S3.split_list


        S1 = SJTU_TIS_1(split="test")

        S2 = SJTU_TIS_2(split="test")
        S3 = SJTU_TIS_3(split="test")

        split["test"] = S1.split_list + S2.split_list + S3.split_list


        S1 = SJTU_TIS_1(split="valid")

        S2 = SJTU_TIS_2(split="valid")
        S3 = SJTU_TIS_3(split="valid")

        split["valid"] = S1.split_list + S2.split_list + S3.split_list

        


            
        with open(self.split_file, "w") as json_file:
            json.dump(split,json_file)
        
        print("[DONE] Build the split file")


    


if __name__ == "__main__":
    data = SJTU_TIS_2(split="train")
    print(len(data))
    # data = SJTU_TIS_2(split="test")
    # print(len(data))
    # data = SJTU_TIS_2(split="valid")
    # print(len(data))
    print(data[-1]) 
    

    for i in range(len(data)):
        if (not  os.path.exists(data[i]["image"]) )or( not os.path.exists(data[i]["map"])):
            print("ERROR")
            print(data[i]["image"])









        
        









