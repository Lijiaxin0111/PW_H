import numpy as np
from sklearn import metrics
from dataset.dataset  import SJTU_TIS_0
from PIL import Image
import torch


#  gt输入map


def auc_judd(gt, pred):
    """Calculate AUC-Judd."""
    pred = pred.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()

    thres_gt = gt.sort()[0][int(0.8 * len(gt))]
    thres_sal = pred.sort()[0][int(0.8 * len(pred))]


    # gt = gt.view(-1)
    # thres = gt.sort()[len(gt())]
    gt_flat = (gt > thres_gt).flatten()
    pred_flat = (pred >thres_sal).flatten()


    auc_judd_score = metrics.roc_auc_score(gt_flat, pred_flat)
    return auc_judd_score

#  gt输入map
def sAUC( fixationMap,saliencyMap, otherMap, Nsplits=100, stepSize=0.1, toPlot=False):
    """Calculate shuffled AUC."""
    fixationMap = fixationMap.view(-1).detach().cpu().numpy()
    saliencyMap= saliencyMap.view(-1).detach().cpu().numpy()
    otherMap=  otherMap.view(-1).detach().cpu().numpy()
    
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # otherMap is a binary fixation map (like fixationMap) by taking the union of
    # fixations from M other random images (Borji uses M=10)
    # Nsplits is the number of random splits
    # stepSize is for sweeping through the saliency map
    # if toPlot=True, displays ROC curve

    if saliencyMap.shape != fixationMap.shape:
        saliencyMap = cv2.resize(saliencyMap, (fixationMap.shape[1], fixationMap.shape[0]))
    

    # # normalize saliency map
    saliencyMap = (saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap))

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        return np.nan

    S = saliencyMap.flatten()
    F = fixationMap.flatten()
    Oth = otherMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by otherMap

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.zeros((Nfixations_oth, Nsplits))

    for i in range(Nsplits):
        randind = np.random.permutation(ind)  # randomize choice of fixation locations
        randfix[:, i] = S[randind[:Nfixations_oth]]  # sal map values at random fixation locations of other random images

    # calculate AUC per random split (set of random locations)
    auc = np.zeros(Nsplits)
    for s in range(Nsplits):
        curfix = randfix[:, s]

        allthreshes = np.flipud(np.arange(0, np.maximum(np.max(Sth), np.max(curfix)) + stepSize, stepSize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[0] = 0
        tp[-1] = 1
        fp[0] = 0
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(fp, tp)

    score = np.mean(auc)  # mean across random splits

    return  score





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
