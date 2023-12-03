
import os
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num


from models.Composed_BLIP import ImageReward,ModMSELoss

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn

from dataset.AGIQA_2023 import AGIQA_2023
from dataset.AGIQA_3k import AGIQA_3k
from dataset.dataset import SJTU_TIS_whole,SJTU_TIS_0,SJTU_TIS_1,SJTU_TIS_2,SJTU_TIS_3,MIT
from tqdm import *
from correlation import get_correlation_Coefficient
import sys
from correlation import get_correlation_Coefficient
from torch.utils.tensorboard import SummaryWriter
import time
from score import auc_judd,nss,cc,sAUC
from torchvision.transforms import ToPILImage

def std_log():
    if get_rank() == 0:
        save_path = make_path()
        makedir(config['log_base'])
        sys.stdout = open(os.path.join(config['log_base'], "{}.txt".format(save_path)), "w")


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True


def loss_func(pred,gt):
    loss = ModMSELoss(240,240)
    
    return loss( pred["pred_map"], gt["gt_map"],model.MLNet.prior.clone())


if __name__ == "__main__":
    print("START train....")
    
    if opts.std_log:
        std_log()
    print("test cuda:",torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seeds(opts.seed)
    print("device:",device)
    run_name = f"seed_{opts.seed}__{int(time.time())}__lr_{opts.lr}__fixed_rate__{opts.fix_rate}"

    writer = visualizer()
    print("dataset:", opts.dataset, " aux:", opts.aux)
    print("distribed", opts.distributed)


    if opts.dataset == "whole":
        train_dataset = SJTU_TIS_whole(split="train")
        test_dataset = SJTU_TIS_whole(split="test")
        valid_dataset = SJTU_TIS_whole(split="valid")

    if opts.dataset == "0":
        train_dataset = SJTU_TIS_whole(split="train")
        test_dataset = SJTU_TIS_whole(split="test")
        valid_dataset = SJTU_TIS_whole(split="valid")


    if opts.dataset == "1":
        train_dataset = SJTU_TIS_1(split="train")
        test_dataset = SJTU_TIS_1(split="test")
        valid_dataset = SJTU_TIS_1(split="valid")


    if opts.dataset == "2":
        train_dataset = SJTU_TIS_2(split="train")
        test_dataset = SJTU_TIS_2(split="test")
        valid_dataset = SJTU_TIS_2(split="valid")
        

    if opts.dataset == "3":
        train_dataset = SJTU_TIS_3(split="train")
        test_dataset = SJTU_TIS_3(split="test")
        valid_dataset = SJTU_TIS_3(split="valid")

    if opts.dataset == "MIT":
        train_dataset = MIT(split="train")
        test_dataset = MIT(split="test")
        valid_dataset = MIT(split="valid")

    # if opts.dataset == "AGIQA_3K":
    #     dataroot = r"/home/jiaxin/Composed_BLIP/data/AGIQA-3K"
    #     train_dataset = AGIQA_3k(data_root=dataroot,split="train",aux = opts.aux)
    #     valid_dataset = AGIQA_3k(data_root=dataroot,split="valid",aux = opts.aux)
    #     test_dataset = AGIQA_3k(data_root=dataroot,split="test",aux = opts.aux)
    # elif opts.dataset == "AGIQA_2023":
    #     dataroot = r"/mnt/lustre/sjtu/home/yzl02/Composed_BLIP/PW_H/data/AIGCIQA2023"
    #     train_dataset = AGIQA_2023(data_root=dataroot,split="train")
    #     valid_dataset = AGIQA_2023(data_root=dataroot,split="valid")
    #     test_dataset = AGIQA_2023(data_root=dataroot,split="test")  

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)

    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    # print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))

    print("len(train_dataset) = ", len(valid_dataset))
    # print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(valid_loader))
    print("steps_per_valid = ", steps_per_valid)

    if opts.mod == "Pure":
        model = ImageReward(device= device,med_config=r"config/med_config.json")
    else:
        model = ImageReward(device= device,med_config=r"config/med_config.json",mode= "Mix_TXT")


    
    print("model: Imagereward")

    if opts.preload_path:
        model = preload_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2), eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    # if opts.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model)


    # valid result print and log
    if get_rank() == 0 and opts.split == "valid":
        model.eval()
        valid_loss = []

        step_tqdm = tqdm( enumerate(valid_loader))
        
      
   
        with torch.no_grad():
            for step, batch_data_package in step_tqdm:

                pred, gt = model(batch_data_package)
                loss = loss_func(pred, gt)
                valid_loss.append(loss)
                # print(f"step = {step} ", f"loss = {loss}\r",end="")
                step_tqdm.set_description(f"step {step}")
                step_tqdm.set_postfix(loss = loss)

           
    
        # record valid and save best model
        valid_loss = torch.cat(valid_loss, 0)
        print('Validation - Iteration %d | Loss %6.5f ' % (0, torch.mean(valid_loss)))
        # writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step)
  


    if opts.split == "train":
        best_loss = 1e9
        optimizer.zero_grad()
        # fix_rate_list = [float(i) / 10 for i in reversed(range(10))]
        # fix_epoch_edge = [opts.epochs / (len(fix_rate_list)+1) * i for i in range(1, len(fix_rate_list)+1)]
        # fix_rate_idx = 0
        losses = []
        

        epochs_tqdm = tqdm(range(opts.epochs))
        
        for epoch in epochs_tqdm :
            # step_tqdm = tqdm( )
            epoch_losses = []

            train_auc = []
            train_sauc = []
            train_cc = []
            train_nss= []

            for step, batch_data_package in enumerate(train_loader):
            
                model.train()
                output,gt = model(batch_data_package)

                output_max = torch.max(torch.max(output["pred_map"],2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output["pred_map"].shape[0],output["pred_map"].shape[1],240,240)
                auc_j_score = auc_judd(gt["gt_map"],output["pred_map"]/ output_max)
                sauc_score = sAUC(gt["gt_map"],output["pred_map"] / output_max)
                cc_score = cc(gt["gt_map"],output["pred_map"] / output_max)
                nss_score = nss(gt["gt_fix"],output["pred_map"] / output_max)

                train_auc.append(auc_j_score)
                train_sauc.append(sauc_score)
                train_cc.append(cc_score)
                train_nss.append(nss_score)



                
                loss= loss_func(output,gt)
            
                max_pred_mask =  torch.max(torch.max(output["pred_map"][0],1)[0],1)[0].unsqueeze(1).unsqueeze(1).expand(output["pred_map"][0].shape[0],240,240) + 1e-5 

                gt_image_map =  (gt["gt_map"][0])
                pred_image_map =  (output["pred_map"][0] / max_pred_mask)

                init_image = (gt["image"][0])



                print("pred: ",pred_image_map[pred_image_map != 0] )
                print("len pred: ", (pred_image_map[pred_image_map != 0].shape))

                print("gt_map:", gt_image_map[gt_image_map != 0])
                print("len gt: " ,  gt_image_map[gt_image_map != 0].shape)

                gt_image_map =  ToPILImage()(gt_image_map)
                pred_image_map =  ToPILImage()(pred_image_map)
                init_image = ToPILImage()(init_image)


                gt_image_map.save("train_gt_image_map.png")
                pred_image_map.save("train_pred_image_map.png")
                init_image.save("train_init_map.png")              


                # loss regularization
                loss = loss / opts.accumulation_steps
                losses.append(loss)
                epoch_losses.append(loss)
                # back propagation
                # print("begin backward")
                loss.backward()
                # print("finished")
                iterations = epoch * len(train_loader) + step + 1
                train_iteration = iterations / opts.accumulation_steps

                print(' Train Loss %6.5f | auc_j_score %6.5f | sauc_score %6.5f | cc_score %6.5f | nss_score %6.5f' 
                    % ( ( (loss)) ,auc_j_score,sauc_score,cc_score,nss_score))

                
                # update parameters of net
                if (iterations % opts.accumulation_steps) == 0:
                    # optimizer the net
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                    # train result print and log 
                    if get_rank() == 0:
                        writer.add_scalar('Train-Loss', torch.mean(torch.tensor(losses)), global_step=train_iteration)
                    losses.clear()

            
                # valid result print and log
                if (iterations % steps_per_valid) == 0:
                    if get_rank() == 0:
                        model.eval()
                        valid_loss = []
              

                    
                        with torch.no_grad():
                            for step, batch_data_package in enumerate(valid_loader):
                                
                                output,gt = model(batch_data_package)
                                loss = loss_func(output,gt)
                                valid_loss.append(loss)
            
                                
                        # record valid and save best model
                        # print(align_score)
                        # output_max = torch.max(torch.max(output["pred_map"][0],2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt)
                        max_pred_mask =  torch.max(torch.max(output["pred_map"][0],1)[0],1)[0].unsqueeze(1).unsqueeze(1).expand(output["pred_map"][0].shape[0],240,240)

                        gt_image_map =  ToPILImage()(gt["gt_map"][0]  )
                        pred_image_map =  ToPILImage()(output["pred_map"][0] / max_pred_mask)

                        gt_image_map.save("valid_gt_image_map.png")
                        pred_image_map.save("valid_pred_image_map.png")

                        output_max = torch.max(torch.max(output["pred_map"],2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output["pred_map"].shape[0],output["pred_map"].shape[1],240,240)
                     
                        auc_j_score = auc_judd(gt["gt_map"],output["pred_map"]/ output_max)
                        sauc_score = sAUC(gt["gt_map"],output["pred_map"] / output_max)
                        cc_score = cc(gt["gt_map"],output["pred_map"] / output_max)
                        nss_score = nss(gt["gt_fix"],output["pred_map"] / output_max)
     
                        writer.add_scalar('Validation-Loss', torch.mean(torch.tensor(valid_loss)), global_step=train_iteration)
                            
                        writer.add_scalar('auc_j_score',  auc_j_score, global_step=train_iteration)
                        writer.add_scalar('sauc_score',  sauc_score, global_step=train_iteration)
                        writer.add_scalar('cc_score',  cc_score, global_step=train_iteration)
                        writer.add_scalar('nss_score',  nss_score, global_step=train_iteration)

            
                        print(' Valid Loss %6.5f | auc_j_score %6.5f | sauc_score %6.5f | cc_score %6.5f | nss_score %6.5f' 
                            % ( torch.mean( torch.mean(torch.tensor(valid_loss))) ,auc_j_score,sauc_score,cc_score,nss_score)  )


                        
                        save_model(model,current= True)

                        if torch.mean(torch.tensor(valid_loss)) < best_loss:
                            print("Best Val loss so far. Saving model")
                            best_loss = torch.mean(torch.tensor(valid_loss))
                            print("best_loss = ", best_loss)
                            save_model(model)
            print(f"step {step}")
            writer.add_scalar('Epoch-Loss', torch.mean(torch.tensor(epoch_losses)), global_step=epoch)


            writer.add_scalar('train/auc_j_score', torch.mean(torch.tensor(train_auc))  , global_step=epoch)
            writer.add_scalar('train/sauc_score',   torch.mean(torch.tensor(train_sauc)), global_step=epoch)
            writer.add_scalar('train/cc_score', torch.mean(torch.tensor(train_cc)), global_step=epoch)
            writer.add_scalar('train/nss_score',  torch.mean(torch.tensor(train_nss)), global_step=epoch)


            epochs_tqdm.set_description(f"epoch {epoch}")
            epochs_tqdm.set_postfix(best_loss = best_loss)
            epochs_tqdm.set_postfix(train_loss = torch.mean(torch.tensor(epoch_losses)))


    # test model
    if get_rank() == 0 and opts.split == "test":
        print("training done")
        print("test: ")
        model = load_model(model,ckpt_path = opts.preload_path)
        model.eval()

        test_loss = []
        auc_j_score_l = []
        sauc_score_l = []
        cc_score_l = []
        nss_score_l = []



        
   


        with torch.no_grad():
            for step, batch_data_package in enumerate(test_loader):
                output,gt = model(batch_data_package)
                loss= loss_func(output,gt)
                test_loss.append(loss)

                output_max = torch.max(torch.max(output["pred_map"],2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output["pred_map"].shape[0],output["pred_map"].shape[1],240,240)

                auc_j_score = auc_judd(gt["gt_map"],output["pred_map"]/ output_max)
                sauc_score = sAUC(gt["gt_map"],output["pred_map"]/ output_max)
                cc_score = cc(gt["gt_map"],output["pred_map"]/ output_max)
                nss_score = nss(gt["gt_map"],output["pred_map"]/ output_max)
                auc_j_score_l.append(auc_j_score)
                sauc_score_l.append(sauc_score)
                cc_score_l.append(cc_score)
                nss_score_l.append(nss_score)

                max_pred_mask =  torch.max(torch.max(output["pred_map"][0],1)[0],1)[0].unsqueeze(1).unsqueeze(1).expand(output["pred_map"][0].shape[0],240,240)


                gt_image_map =  ToPILImage()(gt["gt_map"][0])
                pred_image_map =  ToPILImage()(output["pred_map"][0] / max_pred_mask)



                gt_image_map.save("test_gt_image_map.png")
                pred_image_map.save("test_pred_image_map.png")


                

     
       
                print(' Test Loss %6.5f | auc_j_score %6.5f | sauc_score %6.5f | cc_score %6.5f | nss_score %6.5f' 
                    % ( torch.mean(torch.tensor(test_loss)) ,torch.mean(torch.tensor(auc_j_score_l)),torch.mean(torch.tensor(sauc_score_l)),torch.mean(torch.tensor(cc_score_l)),torch.mean(torch.tensor(nss_score_l))  )   )



                    


