
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import numpy as np

def calculate_auc_precise(x, y, num_points=100):
    interp_func = interp1d(x, y, kind='linear')
    x_interp = np.linspace(min(x), max(x), num_points)
    y_interp = interp_func(x_interp)
    auc = trapezoid(y_interp, x_interp)
    return auc


def precision_at_k(ground_workflows, predicted_workflows, k):
    ground_sorted = sorted(ground_workflows.items(), key=lambda x: x[1], reverse=True)
    ground_top_k = set([workflow for workflow, score in ground_sorted[:k]])
    predicted_sorted = sorted(predicted_workflows.items(), key=lambda x: x[1], reverse=True)
    predicted_top_k = [workflow for workflow, score in predicted_sorted[:k]]
    relevant_count = sum(1 for workflow in predicted_top_k if workflow in ground_top_k)
    precision = relevant_count / k
    return precision


def reorder_ties(predicted_workflows, ground_workflows):
    """
    predicted_workflows: list of tuples [(workflow_id, predicted_score), ...]
    ground_workflows: list of tuples [(workflow_id, true_score), ...]
    """
    # 建立 ground truth 字典方便查分
    ground_dict = {wf: score for wf, score in ground_workflows}

    # 分组：1.0 / 0.0 / 其他
    ones = [(wf, score) for wf, score in predicted_workflows if score == 1.0]
    zeros = [(wf, score) for wf, score in predicted_workflows if score == 0.0]
    others = [(wf, score) for wf, score in predicted_workflows if score != 1.0 and score != 0.0]

    # 根据 ground truth 对 ones 重新排序（从高到低）
    ones_sorted = sorted(ones, key=lambda x: ground_dict.get(x[0], 0), reverse=True)
    # 根据 ground truth 对 zeros 重新排序（从高到低，也可以 reverse=False）
    zeros_sorted = sorted(zeros, key=lambda x: ground_dict.get(x[0], 0), reverse=True)

    # 合并回最终列表
    new_predicted_workflows = ones_sorted + others + zeros_sorted
    return new_predicted_workflows

def calculate_utility(args ,ground_dict ,predicted_dict):
    ground_workflows = sorted(ground_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_workflows = sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_workflows = reorder_ties(predicted_workflows, ground_workflows)
    ground_workflows = {k :v for k ,v in ground_workflows}
    predicted_workflows = {k :v for k ,v in predicted_workflows}
    num_workflow = len(ground_dict)
    x = range(1, num_workflow +1)
    perfect_overlap = [1.0 ] *len(x)
    perfect_auc = calculate_auc_precise(x, perfect_overlap)
    # for k in range(1, num_workflow+1):
    #     overlap_count = 0
    #     for workflow in list(predicted_workflows.keys())[:k]:
    #         if workflow in list(ground_workflows.keys())[:k]:
    #             overlap_count += 1
    #     lst.append(overlap_count / k)
    lst = [precision_at_k(ground_workflows ,predicted_workflows ,k) for k in x]
    utility = calculate_auc_precise(x, lst) / perfect_auc
    import matplotlib.pyplot as plt
    import os
    save_path = os.path.join(args.data_path, f'Precision@K.png')
    plt.ylabel("precision")
    plt.xlabel("k")
    plt.title("Precision@K")
    plt.xticks(np.arange(1, num_workflow +1, 10))
    plt.plot(x ,lst ,label=f'Predicted (AUC score: {utility:.2f})')
    plt.legend()

    plt.savefig(save_path)
    return utility