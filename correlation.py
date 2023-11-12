import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt

# 两个示例列表

def get_correlation_Coefficient(pred,gt,is_show = False):
# 转换为NumPy数组
    arr1 = np.array(pred.cpu()).reshape(-1)
    arr2 = np.array(gt.cpu()).reshape(-1)

    # 计算Spearman相关系数（SRoCC）
    srocc, _ = spearmanr(arr1, arr2)

    # 计算Kendall相关系数（KRoCC）
    krocc, _ = kendalltau(arr1, arr2)

    # 计算Pearson相关系数（PLCC）
    plcc, _ = pearsonr(arr1, arr2)
    if is_show:

        plt.scatter(pred, gt)

        # 添加标签和标题

        plt.xlabel('List 1')
        plt.ylabel('List 2')
        plt.title('Scatter Plot')

        # 显示散点图
        plt.show()

    return srocc, krocc, plcc

# get_correlation_Coefficient([1,,3,4,5],[1,2,3,4,5])



