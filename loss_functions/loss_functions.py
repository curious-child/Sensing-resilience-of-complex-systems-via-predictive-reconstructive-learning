import torch
import sklearn


from utils.utils import pred_accuracy, kendall_rank_coffecient, set_correlation_coffecient


def train_loss(loss_metric,loss_param):
    if loss_metric=="MSELoss":
        return torch.nn.MSELoss()
    elif loss_metric=="BCELoss":
        return torch.nn.BCELoss()
    elif loss_metric=="CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")


def loss_wrapper(loss_metric,criterion,input,target=None,
                train_GNN=None,
                epoch=None,iter=None,device="cuda"):

    if loss_metric == "MSELoss":
        return criterion(input,target)
    elif loss_metric == "BCELoss":
        return criterion(input,target)
    elif loss_metric == "CrossEntropyLoss":
        return criterion(input,target)
    elif loss_metric == "CEPLoss":
        return criterion(train_GNN,input)

    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")


def evaluation_score(score_metric):
    if score_metric=="mse":
        return sklearn.metrics.mean_squared_error
    elif score_metric=="r2":
        return sklearn.metrics.r2_score
    elif score_metric=="rank_accuracy":
        return pred_accuracy
    elif score_metric=="kendall_rank":
        return kendall_rank_coffecient
    elif score_metric=="set_correlation":
        return set_correlation_coffecient
    # 注意分类任务时，应预处理预测值信息，不能直接将预测值（标签概率值直接代入衡量指标） 此处略去，但使用时应注意
    elif score_metric=="accuracy":
        return sklearn.metrics.accuracy_score
    elif score_metric == "f1":
        return sklearn.metrics.f1_score
    elif score_metric=="roc_auc":
        return sklearn.metrics.roc_auc_score
    elif score_metric=="Min_nodes":
        return None
    elif score_metric=="loss_metric":
        return None
    else:
        raise ValueError("the definition {} don't exit\n"
                         "\tyou can define it before using it".format(score_metric))