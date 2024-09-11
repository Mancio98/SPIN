import torch
import math
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            cos_inner = math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cos_out = (1 + math.cos(cos_inner)) / 2
            return [
                self.min_lr + (base_lr - self.min_lr) * cos_out
                for base_lr in self.base_lrs
            ]

def get_scheduler(args, optimizer):

    type = args.lr_scheduler
    
    if type == 'cosine':

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iteration, eta_min=args.min_lr)
    
    elif type == 'warmupcos':

        return WarmupCosineLR(optimizer, warmup_epochs=args.warmup, max_epochs=args.iteration, min_lr=args.min_lr)
    
    elif type == 'onplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=args.factor, patience=1,
                                                        threshold=0.0001, threshold_mode='rel',
                                                        cooldown=0, min_lr=1e-5)
    
    elif type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.factor)