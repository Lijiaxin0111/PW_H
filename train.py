
import os
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num


from models.Composed_BLIP import Composed_BLIP,Composed_Evaluator,ImageReward

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn

from dataset.AGIQA_2023 import AGIQA_2023
from dataset.AGIQA_3k import AGIQA_3k
from tqdm import *
from correlation import get_correlation_Coefficient
import sys
from correlation import get_correlation_Coefficient
from torch.utils.tensorboard import SummaryWriter
import time

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
    w_align = config["align_weight"]
    w_quality = config["quality_weight"]


    if opts.stage == 2:
        loss = w_align * torch.sum(( pred["align_score"] - gt["align_score"] )**2) + w_quality * torch.sum((pred["quality"] - gt["quality"])**2)
    elif opts.stage == 1:
        loss = w_align * torch.sum(( pred["align_score"] - gt["uniform"] )**2) + w_quality * torch.sum((pred["quality"] - gt["quality"])**2)
    return loss

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

    if opts.dataset == "AGIQA_3K":
        dataroot = r"/home/jiaxin/Composed_BLIP/data/AGIQA-3K"
        train_dataset = AGIQA_3k(data_root=dataroot,split="train",aux = opts.aux)
        valid_dataset = AGIQA_3k(data_root=dataroot,split="valid",aux = opts.aux)
        test_dataset = AGIQA_3k(data_root=dataroot,split="test",aux = opts.aux)
    elif opts.dataset == "AGIQA_2023":
        dataroot = r"/home/jiaxin/Composed_BLIP/data/AIGCIQA2023"
        train_dataset = AGIQA_2023(data_root=dataroot,split="train")
        valid_dataset = AGIQA_2023(data_root=dataroot,split="valid")
        test_dataset = AGIQA_2023(data_root=dataroot,split="test")  

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


    if opts.composed:
        model = Composed_Evaluator(device = device)
    else:
        model = ImageReward(device= device)
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
        writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step)
  


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

            for step, batch_data_package in enumerate(train_loader):
            
                model.train()
                output,gt = model(batch_data_package)
                
                loss= loss_func(output,gt)
                # loss regularization
                loss = loss / opts.accumulation_steps
                losses.append(loss)
                # back propagation
                # print("begin backward")
                loss.backward()
                # print("finished")
                iterations = epoch * len(train_loader) + step + 1
                train_iteration = iterations / opts.accumulation_steps

                
                # update parameters of net
                if (iterations % opts.accumulation_steps) == 0:
                    # optimizer the net
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                    # train result print and log 
                    if get_rank() == 0:

                        print('Iteration %d | Loss %6.5f | align_s %6.5f | gt_align_s %6.5f | quality %6.5f | gt_quality %6.5f '
                         % (train_iteration, torch.mean(torch.tensor(losses)), output["align_score"][0], gt["align_score"][0], output["quality"][0] , gt["quality"][0]  ))
                        writer.add_scalar('Train-Loss', torch.mean(torch.tensor(losses)), global_step=train_iteration)

                    losses.clear()

            
                # valid result print and log
                if (iterations % steps_per_valid) == 0:
                    if get_rank() == 0:
                        model.eval()
                        valid_loss = []
                        align_score = []
                        gt_align_score = []

                        quality_score = []
                        gt_quality_score = []
                    
                        with torch.no_grad():
                            for step, batch_data_package in enumerate(valid_loader):
                                
                                output,gt = model(batch_data_package)
                                loss = loss_func(output,gt)
                                valid_loss.append(loss)
                                align_score.append(output["align_score"])
                                gt_align_score.append(gt["align_score"])

                                quality_score.append(output["quality"])
                                gt_quality_score.append(gt["quality"])
                                
                        # record valid and save best model
                        # print(align_score)

                        align_score = torch.concat(align_score,dim= 0)
                        quality_score = torch.concat(quality_score,dim= 0)
                        gt_align_score = torch.concat(gt_align_score,dim= 0)
                        gt_quality_score = torch.concat(gt_quality_score,dim= 0)
                      
                        align_srocc, align_krocc, align_plcc =  get_correlation_Coefficient(align_score, gt_align_score)
                        quality_srocc, quality_krocc, quality_plcc =  get_correlation_Coefficient(quality_score, gt_quality_score)
                        


                        print('Validation - Iteration %d | Loss %6.5f | align_s %6.5f | gt_align_s %6.5f | quality %6.5f | gt_quality %6.5f | \n  align_srocc %6.5f , align_plcc %6.5f ,quality_srocc %6.5f  , quality_plcc %6.5f'
                         % (train_iteration, torch.mean(torch.tensor(valid_loss)), output["align_score"][0], gt["align_score"][0], output["quality"][0] , gt["quality"][0] , align_srocc , align_plcc ,quality_srocc ,quality_plcc ))

                         

                        
                        writer.add_scalar('Validation-Loss', torch.mean(torch.tensor(valid_loss)), global_step=train_iteration)
                            
                        writer.add_scalar('align/align_srocc', align_srocc, global_step=train_iteration)
                        writer.add_scalar('align/align_krocc', align_krocc, global_step=train_iteration)                    
                        writer.add_scalar('align/align_plcc', align_plcc, global_step=train_iteration) 

                        writer.add_scalar('quality/quality_srocc', quality_srocc, global_step=train_iteration)
                        writer.add_scalar('quality/quality_krocc',quality_krocc, global_step=train_iteration)                    
                        writer.add_scalar('quality/quality_plcc', quality_plcc, global_step=train_iteration) 

                        

                        if torch.mean(torch.tensor(valid_loss)) < best_loss:
                            print("Best Val loss so far. Saving model")
                            best_loss = torch.mean(torch.tensor(valid_loss))
                            print("best_loss = ", best_loss)
                            save_model(model)
            print(f"step {step}")



            epochs_tqdm.set_description(f"epoch {epoch}")
            epochs_tqdm.set_postfix(best_loss = best_loss)


    # test model
    if get_rank() == 0 and opts.split == "test":
        print("training done")
        print("test: ")
        model = load_model(model,ckpt_path = opts.preload_path)
        model.eval()

        test_loss = []
        align_score = torch.tensor([]).to(device)
        quality_score = torch.tensor([]).to(device)

        gt_align_score = torch.tensor([]).to(device)
        gt_quality_score = torch.tensor([]).to(device)


        with torch.no_grad():
            for step, batch_data_package in enumerate(test_loader):
                output,gt = model(batch_data_package)
                loss= loss_func(output,gt)
                test_loss.append(loss)
                align_score = torch.concat([ align_score, output["align_score"]], dim = 0)
                quality_score = torch.concat([quality_score , output["quality"]], dim = 0)

                gt_align_score = torch.concat([ gt_align_score, gt["align_score"]], dim = 0)
                gt_quality_score = torch.concat([ gt_quality_score , gt["quality"]], dim = 0)
                
                print(' Test Loss %6.5f | align_s %6.5f | gt_align_s %6.5f | quality %6.5f | gt_quality %6.5f | '
                    % ( torch.mean(torch.tensor(test_loss)), output["align_score"][0], gt["align_score"][0], output["quality"][0] , gt["quality"][0] ))

        align_srocc, _ , align_plcc = get_correlation_Coefficient(align_score, gt_align_score)
        quality_srocc, _ , quality_plcc = get_correlation_Coefficient(quality_score, gt_quality_score)
        print(f"align_srocc : {align_srocc}, align_plcc:{align_plcc}")
        print(f"quality_srocc: {quality_srocc}, quality_plcc:{quality_plcc}")


                    


