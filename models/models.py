


def Resilience_model(net_param):

    if net_param["model_task"]  == 'AE_recon':
        from models.pred_model.AE_recon_model import AE_recon_model
        return AE_recon_model(net_param)
    elif net_param["model_task"] == 'CVAE_recon':
        from models.pred_model.CVAE_recon_model import CVAE_recon_model
        return CVAE_recon_model(net_param)
    else:
        raise ValueError('Invalid model task')





