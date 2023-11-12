
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from config.options import *
from config.utils import *
from models.blip_pretrain import blip_pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


from models.BLIP.blip_pretrain import BLIP_Pretrain
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        # Resize(n_px, interpolation=BICUBIC),
        Resize(n_px),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layer_1 = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()

        )
        self.layer_2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )


        # initial MLP param
        for name, param in self.layer_1.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
        for name, param in self.layer_2.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)


    def forward(self, input):
        embing = self.layer_1(input)
        align_output = self.layer_2(embing)

        quality_output =  self.layer_3(embing)


        return  align_output  , quality_output 


class Composed_BLIP(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        
        self.blip = blip_pretrain(pretrained=config['blip_path'], image_size=config['BLIP']['image_size'], vit=config['BLIP']['vit']).to(device)
        self.preprocess = _transform(config['BLIP']['image_size'])

        self.align_score_evaluator = MLP(config['Composed_BLIP']['align_socre_evaluator_dim']).to(device)
        # self.quality_score_evaluator  = MLP(config['Composed_BLIP']['qualify_evaluator_dim']).to(device)
        
        if opts.fix_base:
            self.blip.requires_grad_(False)
        
        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers
        self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        if opts.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break



    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):
        batch_size = len(batch_data)
        

        cross_emb,Image_emb = self.encode_data(batch_data)
        # print(Image_emb.shape)


        output = {}
        # print("image emb:",Image_emb.shape)
        # print("cross emb:",cross_emb.shape)

        # Image_emb = Image_emb.view(batch_size,-1).to(self.device)
        # print(Image_emb.shape)


        # quality_embing, output["quality"] = self.quality_score_evaluator(Image_emb)
        

        # cross_emb =  torch.cat([quality_embing, cross_emb], 1).float().to(self.device)


        output["align_score"], output["quality"]  = self.align_score_evaluator(cross_emb)

        # ("output[quality] ",   output["quality"].shape)
        # print("quality_embing:",quality_embing.shape)
        # print("cross emb:",cross_emb.shape) 
        # print("output[align_score]", output["align_score"].shape)


        gt = {}
        gt["align_score"] = torch.tensor(  [batch_data[i]["mos_align"] for i in range(batch_size)]).to(self.device)
        gt["uniform"] =  torch.tensor(  [batch_data[i]["uniform"] for i in range(batch_size)]).to(self.device)

        gt_fake_align =  torch.tensor( [ 0 for i in range(len(batch_data)) if item[i]["fake_image"] ]).to(self.device)

        gt_fake_quality = torch.tensor( [batch_data[i]["fake_image_quality"] for i in  range(len(batch_data)) if item[i]["fake_image"] ]).to(self.device)



        gt["quality"] =  torch.tensor(  [batch_data[i]["mos_quality"] for i in range(batch_size)]).to(self.device)

        gt["align_score"] = torch.concat([gt["align_score"] , gt_fake_align] , dim = 0)
        gt["quality"] = torch.concat([gt["quality"], gt_fake_quality], dim = 0)
        gt["uniform"] = torch.concat([gt["uniform"] , gt_fake_align] , dim = 0)

        return output,gt
    




    def encode_data(self, batch_data):
        txt_outputs = []
        image_embeds = []
        item = batch_data

        prompts = [item[i]["prompt"] for i in range(len(batch_data))]


      
        text_input = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        # print([ self.preprocess( (item[i]["image"])).to(self.device) for i in range(len(batch_data))])
        # print("text_input: ", text_input)
      
       
        images = torch.concat( [ self.preprocess( Image.open((item[i]["image"]))).to(self.device).unsqueeze(0) for i in range(len(batch_data))] , dim = 0)


        image_embeds = self.blip.visual_encoder(images).to(self.device)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)

        text_output = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        
        emd_output = (text_output.last_hidden_state[:,0,:]).to(self.device)

        return emd_output, image_embeds





'''
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''




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


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.embeding_layer = nn.Sequential(
            nn.Linear(input_size, 768),
            #nn.ReLU(),
            nn.Dropout(0.2),   
        )
        
        self.layers = nn.Sequential( 

            nn.Linear(768, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param

        for name, param in self.embeding_layer.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        embing = self.embeding_layer(input)
        return self.layers(embing)


class quality_MLP(nn.Module):
    def __init__(self, input_size,device):
        super().__init__()
        self.input_size = input_size
        self.device = device

        self.dim_weight =  torch.randn( input_size[0])
        nn.init.normal_(self.dim_weight, mean=0.0, std=1.0/(input_size[0]+1))

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size[1], 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layer1.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size[1]+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

        for name, param in self.layer2.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size[1]+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        

    def forward(self, input):
        

        # z = torch.sum(self.dim_weight.view(self.input_size[0], 1 ).repeat(1,self.input_size[1]).to(self.device) * input,dim = 1)
        z = torch.sum(input , dim = 1)

        z_ = self.layer1(z)

        return z_,self.layer2(z_)


class ImageReward(nn.Module):
    def __init__(self, med_config= r"/home/jiaxin/Composed_BLIP/checkpoint/med_config.json", device='cuda'):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config).to(self.device)
        self.preprocess = _transform(224)
        self.mlp = MLP(768 ).to(self.device)
        self.quality_mlp = quality_MLP((197,1024),device).to(self.device)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

        if opts.fix_base:
            self.blip.requires_grad_(False)
        
        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)


        
        # fix certain ratio of layers
        self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        if opts.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break

    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):
    
        txt_outputs = []
        image_embeds = []
        item = batch_data
        batch_size = len(batch_data)
        # print(batch_data[0])

        prompts = [item[i]["prompt"] for i in range(len(batch_data))]

    
        

        fake_promts = ( [ item[i]["prompt"] for i in range(len(batch_data)) if item[i]["fake_image"] ])[:3]

        prompts =  prompts + fake_promts
        # print("text ",prompts)
        # text encode
        text_input = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        # print(item[0]["image"])
        # image encode
        images = torch.concat( [ self.preprocess( Image.open( (item[i]["image"]))).to(self.device).unsqueeze(0) for i in range(len(batch_data))] , dim = 0)


        fake_images =  [ self.preprocess( Image.open((item[i]["fake_image"]))).to(self.device).unsqueeze(0) for i in range(len(batch_data)) if item[i]["fake_image"]][:3]

        if len(fake_images) != 0:
            fake_images = torch.concat(fake_images[:3] , dim = 0)
            images = torch.concat([images, fake_images], dim = 0)

        image_embeds = self.blip.visual_encoder(images).to(self.device)

        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)

        # print(image_atts.shape)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )

        pred = {}

        



        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
        rewards = self.mlp(  (txt_features))



        rewards = (rewards - self.mean) / self.std
        # print(rewards.shape)

        gt = {}
        # pred["align_score"] = rewards.reshape(-1)

        gt_fake_align =  torch.tensor( [ 0 for i in range(len(batch_data)) if item[i]["fake_image"] ])[:3].to(self.device)
        gt_fake_quality = torch.tensor( [batch_data[i]["fake_image_quality"]  for i in  range(len(batch_data)) if item[i]["fake_image"] ][:3]).to(self.device)


  
        gt["align_score"] = torch.tensor(  [batch_data[i]["mos_align"] for i in range(batch_size)]).to(self.device)
        gt["uniform"] =  torch.tensor(  [batch_data[i]["uniform"] for i in range(batch_size)]).to(self.device)
        gt["quality"] =  torch.tensor(  [batch_data[i]["mos_quality"] for i in range(batch_size)]).to(self.device)



        gt["align_score"] = torch.concat([gt["align_score"] , gt_fake_align] , dim = 0)
        gt["quality"] = torch.concat([gt["quality"], gt_fake_quality], dim = 0)
        gt["uniform"] = torch.concat([gt["uniform"] , gt_fake_align] , dim = 0)
        pred["quality"] = rewards.reshape(-1)
        # pred["quality"] = gt["quality"] 
        pred["align_score"] = gt["align_score"]


 


        # print( pred["quality"].shape )
        return pred , gt


    

class Composed_Evaluator(nn.Module):
    def __init__(self, med_config= r"/home/jiaxin/Composed_BLIP/checkpoint/med_config.json", device='cuda'):
        super().__init__()
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config).to(self.device)
        self.preprocess = _transform(224)
        self.mlp = MLP(768 + 64).to(self.device)
        # self.mlp = MLP(768 ).to(self.device)

        # self.quality_MLP = MLP(1024).to(self.device)
        self.quality_MLP = quality_MLP((197,1024),device).to(self.device)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

        if opts.fix_base:
            self.blip.requires_grad_(False)
        
        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)


        
        # fix certain ratio of layers
        self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        if opts.fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break

    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):
    
        txt_outputs = []
        image_embeds = []
        item = batch_data
        batch_size = len(batch_data)
        # print(batch_data[0])

        prompts = [item[i]["prompt"] for i in range(len(batch_data))]

    
        

        fake_promts = ( [ item[i]["prompt"] for i in range(len(batch_data)) if item[i]["fake_image"] ])[:3]

        prompts =  prompts + fake_promts

        # text encode
        text_input = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
        # print(item[0]["image"])
        # image encode
        images = torch.concat( [ self.preprocess( Image.open( (item[i]["image"]))).to(self.device).unsqueeze(0) for i in range(len(batch_data))] , dim = 0)


        fake_images =  [ self.preprocess( Image.open((item[i]["fake_image"]))).to(self.device).unsqueeze(0) for i in range(len(batch_data)) if item[i]["fake_image"]][:3]

        if len(fake_images) != 0:
            fake_images = torch.concat(fake_images[:3] , dim = 0)
            images = torch.concat([images, fake_images], dim = 0)

        image_embeds = self.blip.visual_encoder(images).to(self.device)

        # print(image_embeds.shape)

    #     第一种: 直接求和
        # image_code = torch.sum( image_embeds,dim= 1)

    #   第二种： 增加权重求和

    
   
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)

        # print(image_atts.shape)
        text_output = self.blip.text_encoder(text_input.input_ids,
                                                attention_mask = text_input.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )

        pred = {}
        quality_z,  pred["quality"] = self.quality_MLP(image_embeds)

        
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
        # rewards = self.mlp(  (txt_features))
        # print(quality_z.shape)
        rewards = self.mlp(  torch.concat([txt_features, quality_z] , dim = 1))
        rewards = (rewards - self.mean) / self.std
        # print(rewards.shape)



        gt = {}
        pred["align_score"] = rewards.reshape(-1)

        gt_fake_align =  torch.tensor( [ 0 for i in range(len(batch_data)) if item[i]["fake_image"] ])[:3].to(self.device)
        gt_fake_quality = torch.tensor( [batch_data[i]["fake_image_quality"]  for i in  range(len(batch_data)) if item[i]["fake_image"] ][:3]).to(self.device)

  
        gt["align_score"] = torch.tensor(  [batch_data[i]["mos_align"] for i in range(batch_size)]).to(self.device)
        gt["uniform"] =  torch.tensor(  [batch_data[i]["uniform"] for i in range(batch_size)]).to(self.device)
        gt["quality"] =  torch.tensor(  [batch_data[i]["mos_quality"] for i in range(batch_size)]).to(self.device)

        # print(pred["quality"].shape)
        gt["align_score"] = torch.concat([gt["align_score"] , gt_fake_align] , dim = 0)
        gt["quality"] = torch.concat([gt["quality"], gt_fake_quality], dim = 0)
        gt["uniform"] = torch.concat([gt["uniform"] , gt_fake_align] , dim = 0)
        
        # print( pred["quality"].shape )

        return pred , gt


    