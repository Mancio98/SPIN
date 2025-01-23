import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup



def get_optimizer(optimizer, model_params, **options):

    if optimizer == 'adam':

        return torch.optim.Adam(model_params, **options)
    
    elif optimizer == 'adamw':

        return torch.optim.Adam(model_params, **options)
    
def get_scheduler(optimizer, lr_scheduler, train_steps, factor: float = None):

    if lr_scheduler == 'warmuplinear':

        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(train_steps * 0.1), num_training_steps=train_steps)
    
    elif lr_scheduler == 'warmupcos':

        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(train_steps * 0.1), num_training_steps=train_steps)
    
    elif lr_scheduler == 'onplateau':

        if factor is not None:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=1, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-7, factor=factor
                )
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=1, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-7
                )
    
    elif lr_scheduler == 'exponential':

        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor if factor else 0.1)
    else:
        return None