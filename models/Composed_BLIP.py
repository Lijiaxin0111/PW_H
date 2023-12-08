
import os
import torch
import torch.nn as nn
from PIL import Image
from .BLIP.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,ToPILImage
import torchvision.transforms.functional as F
from scipy.io import loadmat

import torch.nn.functional as tF

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
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])




# Modified MSE Loss Function
class ModMSELoss(torch.nn.Module):
    def __init__(self,shape_r_gt,shape_c_gt):
        super(ModMSELoss, self).__init__()
        self.shape_r_gt = shape_r_gt
        self.shape_c_gt = shape_c_gt
        
    def forward(self, output , label , prior):
        prior_size = prior.shape
        
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt) + 1e-5 * torch.ones(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt).to("cuda")
        reg = ( 1.0/(prior_size[0]*prior_size[1]) ) * ( 1 - prior)**2
        # print("output_max: ",output_max)
        # print("label: ",torch.max(torch.max(label,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt)) 
        # print("1 - label + 0.1 ",1 - label + 0.1)
        loss = torch.mean( ((output / output_max) - label)**2 / ((1 - label + 0.1) ) )  +  torch.sum(reg)
        return loss

class CorssAttention(nn.Module):
    def __init__(self, txt_n=16 ,dim_txt=48, image_dim = 40):
        super(CorssAttention, self).__init__()
        self.txt_n = txt_n
        self.dim_txt = dim_txt
        self.image_dim = image_dim

        self.query = nn.Linear(16, 16)
        self.key = nn.Linear(dim_txt, 16)
        self.value = nn.Linear(dim_txt, 16)
        self.scale = 16 ** -0.5
        


    def forward(self,images, txt_feature):
        # print(images)
        print(images.shape)

        images = images.reshape(txt_feature.shape[0],-1,40,40)
         
        unfolded_tensor = tF.unfold(images, kernel_size=4, stride=4)
 

        # 将展开后的张量重塑为（100, 4）形状
        unfold_image = unfolded_tensor.permute(0, 2, 1).reshape(txt_feature.shape[0],-1, 16)
        # print(unfold_image)
# 
        txt_feature = txt_feature.view(txt_feature.shape[0], self.txt_n, self.dim_txt)


        image_query = self.query(unfold_image)
        keys = self.key(txt_feature)
        valus = self.value(txt_feature)


        image_query = image_query.view(txt_feature.shape[0], -1 ,1,16).transpose(1, 2)
        keys = keys.view(txt_feature.shape[0], self.txt_n, 1,-1).transpose(1, 2)
        valus = valus.view(txt_feature.shape[0], self.txt_n, 1,-1).transpose(1, 2)

        # print("image_query:",image_query.shape)
        # print("txt_key:",keys.shape)
        # print("txt_value:",valus.shape)

        dots = torch.einsum('bhid,bhjd->bhij', image_query, keys) * self.scale
        # print("dots: ",dots.shape)

        attn = dots.softmax(dim=-1)

        # print("attn ",attn.shape)
        out = torch.einsum('bhid,bhjd->bhij', attn,valus) 

        # print(out.shape)


        # 使用fold函数将还原后的张量折叠回原始图像
        restored_tensor = out.reshape(txt_feature.shape[0], 100 ,-1).permute(0, 2, 1)
        print(restored_tensor.shape)

        # 使用fold函数将还原后的张量折叠回原始图像
        folded_tensor = tF.fold(restored_tensor, output_size=(40, 40), kernel_size=4, stride=4)

        return folded_tensor
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
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
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)

import torch
import torch.nn as nn
import torchvision.models as models




class MLNet(nn.Module):
    
    def __init__(self,prior_size):
        super(MLNet, self).__init__()
        # loading pre-trained vgg16 model and         
        # removing last max pooling layer
        features = list(models.vgg16(pretrained = True).features)[:-1]
        
        # making same spatial size
        # by calculation :) 
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2
                
        self.features = nn.ModuleList(features).eval() 
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.5)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(1280,64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64,1,kernel_size=(1, 1), stride=(1, 1) ,padding=(0, 0))
        # prior initialized to ones
        self.prior = nn.Parameter(torch.ones((1,1,prior_size[0],prior_size[1]), requires_grad=True))
        
        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=10)
        self.bilinearup_result = torch.nn.UpsamplingBilinear2d(scale_factor=6)
        
        
    def forward(self, x):
        
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {16,23,29}:
                results.append(x)
        
        # concat to get 1280 = 512 + 512 + 256
        x = torch.cat((results[0],results[1],results[2]),1) 
    
        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)
        
        # 64 filters convolution layer
        x = self.int_conv(x)
        # print("before x shape:{}".format(x.shape))
        # 1*1 convolution layer
        x = self.pre_final_conv(x)
        # print("x shape:{}".format(x.shape))
        
        upscaled_prior = self.bilinearup(self.prior)
        # print ("upscaled_prior shape: {}".format(upscaled_prior.shape))

        # dot product with prior
        x = x * upscaled_prior

        
        x = torch.nn.functional.relu(x,inplace=True)
        x = self.bilinearup_result(x)
        return x




class TXT_MLNet(nn.Module):
    
    def __init__(self,prior_size):
        super(TXT_MLNet, self).__init__()
        # loading pre-trained vgg16 model and         
        # removing last max pooling layer
        features = list(models.vgg16(pretrained = True).features)[:-1]
        
        # making same spatial size
        # by calculation :) 
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2
                
        self.features = nn.ModuleList(features).eval() 
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.5)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(1280,64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64,1,kernel_size=(1, 1), stride=(1, 1) ,padding=(0, 0))
        # prior initialized to ones
        self.prior = nn.Parameter(torch.ones((1,1,prior_size[0],prior_size[1]), requires_grad=True))
        
        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=10)
        self.bilinearup_result = torch.nn.UpsamplingBilinear2d(scale_factor=6)

        self.CA_1 = CorssAttention()
        self.CA_2 = CorssAttention()
    
        
        
    def forward(self, x, txt_feature):
        
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {16,23,29}:
                results.append(x)

      
        
        # concat to get 1280 = 512 + 512 + 256

        x = torch.cat(( (results[0]),self.CA_1(results[1], txt_feature),( results[2])),1)
    
        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)
        
        # 64 filters convolution layer
        x = self.int_conv(x)
        # print("before x shape:{}".format(x.shape))
        # 1*1 convolution layer
        x = self.pre_final_conv(x)
        # print("x shape:{}".format(x.shape))
        # print(txt_feature[:, :729].shape)
    
        
        upscaled_prior = self.bilinearup(self.prior).repeat(txt_feature.shape[0],1,1,1)


        # print ("upscaled_prior shape: {}".format(upscaled_prior.shape))
        # print(txt_feature.shape)

        # upscaled_prior = self.CA_2(upscaled_prior, txt_feature)


        # dot product with prior
        x = x * upscaled_prior

        # x = self.CA_2(x,txt_feature)

        x = torch.nn.functional.relu(x,inplace=True)



        x = self.bilinearup_result(x)
        upscaled_prior = (upscaled_prior, txt_feature)


        return x





class Compose_MLNet(nn.Module):
    def __init__(self, med_config, device='cpu',last_freeze_layer = 20,mode=  "Pure_MLNet"):
        super().__init__()
        self.device = device
        self.last_freeze_layer = last_freeze_layer
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config=med_config).to(self.device)
        self.preprocess = _transform(224)
        self.preprocess_mlnet = _transform(320)


        self.mlp = MLP(768).to(self.device)
        self.mode = mode


        
        if self.mode == "Pure_MLNet":
            self.MLNet = MLNet((4,4)).to(self.device)
        else:
            self.MLNet = TXT_MLNet((4,4)).to(self.device)
            
        # freezing Layer
        for i,param in enumerate(self.MLNet.parameters()):
            if i < self.last_freeze_layer:
                param.requires_grad = False
            # print("the layer:", i)


    def process_images(self,images,preprocess):
        process_images = []

        for image in images:
            # image encode
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, str):
                if os.path.isfile(image):
                    pil_image = Image.open(image)
            else:
                raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
            
            image = preprocess(pil_image).unsqueeze(0).to(self.device)
            # print(image.shape)
            process_images.append(image)
            
       
        process_images = torch.concat(process_images,dim = 0)
        process_images.requires_grad=False
    
        return process_images
    
    def process_maps(self,images,is_fix = True):
        process_images = []

        for image in images:
            # image encode
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, str):
                if os.path.isfile(image):
                    if is_fix:
                        # pil_image = Image.open(image)
                        # print("!!",loadmat(image)["fixLocs"])
                        # pil_image = ToPILImage()(torch.tensor((loadmat(image)["fixLocs"])))
                        pil_image = Image.open(image)
                    else:
                        pil_image = Image.open(image)
            else:
                raise TypeError(r'This image parameter type has not been supportted yet. Please pass PIL.Image or file path str.')
        
            
            tt = ToTensor()
            # print(image)
            image =  tt(F.resize(pil_image, (240,240))).unsqueeze(0).to(self.device)
            
            # image[image != 0] = 1.0
            # print("before: ",image.shape)
            if is_fix:
                image = (image[:,0] / 255).view(1,1,240,240)
                # print("after: ",image.shape)
                

            process_images.append(image)
 
            

            
        process_images = torch.concat(process_images,dim = 0)
        process_images.requires_grad=False
    
        return process_images



    def forward(self,inputs):
        # print(inputs)
        # print(inputs)
        prompts = [inputs[i]["des"] for i in range(len(inputs))]
        images = [inputs[i]["image"] for i in range(len(inputs))]
        maps = [inputs[i]["map"] for i in range(len(inputs))]
        fixs = [inputs[i]["fix"]  for i in range(len(inputs))]

        process_images = self.process_images(images,self.preprocess)

        # print(process_images.shape)
        image_embeds = self.blip.visual_encoder(process_images)
        
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)

        text_inputs = self.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)

        text_output = self.blip.text_encoder(text_inputs.input_ids,
                                                attention_mask = text_inputs.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )
        
        
        txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)

        process_images_mlp = self.process_images(images,self.preprocess_mlnet)
        # print(txt_features.shape)
   


        if self.mode == "Pure_MLNet" :
            pred_map = self.MLNet(process_images_mlp )
        else:
            pred_map = self.MLNet(process_images_mlp, txt_features )


        output = {}
        gt = {}
        output["pred_map"] =pred_map


    

        gt["gt_map"] = self.process_maps(maps,is_fix=False)
        gt["gt_fix"] = self.process_maps(fixs,is_fix = True)

        gt["image"] =process_images_mlp
        
    

        # gt["gt_map"] = gt["gt_map"] / torch.max(torch.max(gt["gt_map"][0],1)[0],1)[0].unsqueeze(1).unsqueeze(1).expand(output["pred_map"][0].shape[0],240,240)


        print(gt["gt_map"])
        print(gt["gt_map"].shape)
        print(images[0])

        
        # gt["gt_fix"] = loadmat('file.mat'
        # print("gt:",gt["gt_map"].shape)
        # print(gt["gt_map"] / torch.max(torch.max(gt["gt_map"][0],1)[0],1)[0].unsqueeze(1).unsqueeze(1).expand(output["pred_map"][0].shape[0],240,240))
        
        # print( torch.max(torch.max(gt["gt_map"][0],1)[0],1)[0].unsqueeze(1).unsqueeze(1).expand(output["pred_map"][0].shape[0],240,240))
        # print("pred:",output["pred_map"].shape)
        
        # print(gt["gt_fix"])
        # print(output["pred_map"])
        # print(inputs)
        return output, gt
