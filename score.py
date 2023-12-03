import numpy as np
from sklearn import metrics
from dataset.dataset  import SJTU_TIS_0
from PIL import Image
import torch


#  gt输入map
def auc_judd(gt, pred):
    """Calculate AUC-Judd."""
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    thres = 0.6 * gt.max()


    # gt = gt.view(-1)
    # thres = gt.sort()[len(gt())]


    gt_flat = (gt > thres).flatten()
    pred_flat = (pred > thres).flatten()

    auc_judd_score = metrics.roc_auc_score(gt_flat, pred_flat)
    return auc_judd_score

#  gt输入map
def sAUC(gt, pred):
    """Calculate shuffled AUC."""
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    n_pixels = gt.size

    idx = np.random.permutation(n_pixels)
    gt_shuffled = (gt).flatten()[idx]

    thres = 0.6 * gt_shuffled.max()
    
    s_auc_score = metrics.roc_auc_score(gt_shuffled > thres, (pred.flatten()[idx])> thres).flatten()
    return s_auc_score

#  gt输入map
def cc(gt, pred):
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    """Calculate Pearson correlation coefficient."""
    cc_score = np.corrcoef(gt.flatten(), pred.flatten())[0, 1]
    return cc_score


# gt要输入fix
def nss(gt_fix, pred):
    gt_fix = gt_fix.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    """Calculate Normalized Scanpath Saliency."""
    mean_pred = np.mean(pred,axis=(2,3))
    print("mean:",mean_pred.shape)
    std_pred = np.std(pred,axis=(2,3))
    pred_ = (( pred[gt_fix!=0] - mean_pred) / std_pred)
    nss_score = np.mean(pred_ )

    return nss_score

# 示例使用：
# 假设gt和pred是两个显著图的二进制掩码，0表示非显著，1表示显著

if __name__ == "__main__":

    data = SJTU_TIS_0()
    print(data[0]["fix"])
    map_ =  torch.tensor(np.array( Image.open(data[0]["map"])))
    fix_ =torch.tensor( np.array( Image.open(data[0]["fix"])))
    fix_ = fix_[:,:,0] / 255
    gt =  map_ 
    pred = map_
    print(gt.shape)
 

    print(pred.shape)
    gt = torch.tensor([[[[0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0]]] , [[[0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0]]]])


    pred = torch.tensor([[[[0.2, 0.3, 0.8, 0.7],
                    [0.1, 0.9, 0.6, 0.4],
                    [0.7, 0.5, 0.3, 0.2]]],
                      [[[0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0]]]])
    
    print(gt.shape)
    # 计算各指标
    
    auc_judd_score = auc_judd( gt, pred )
    s_auc_score = sAUC( gt, pred )
    cc_score = cc( gt, pred )
    nss_score = nss( gt, pred )

    # 打印结果
    print(f'AUC-Judd: {auc_judd_score}')
    print(f'sAUC: {s_auc_score}')
    print(f'CC: {cc_score}')
    print(f'NSS: {nss_score}')
