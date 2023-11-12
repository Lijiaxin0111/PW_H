import os
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num


from models.Composed_BLIP import Composed_BLIP

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from dataset.AGIQA_2023 import AGIQA_2023
import sys


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
   
    loss = w_align * torch.sum( pred["align_score"] - gt["align_score"] )**2 + w_quality *  torch.sum((pred["quality"] - gt["quality"])**2)
    return loss


if __name__ == "__main__":
    print("START TRAIN....")
    
    if opts.std_log:
        std_log()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:",device)

    init_seeds(opts.seed)
    writer = visualizer()

    train_dataset = AGIQA_2023(data_root=opts.dataroot, split= "train")
    valid_dataset = AGIQA_2023(data_root=opts.dataroot,split= "valid")
    test_dataset = AGIQA_2023(data_root=opts.dataroot, split="test")

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
