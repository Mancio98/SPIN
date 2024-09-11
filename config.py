from typing import List
import torch


class TrainArgs():
    
    nametrain = "finetuning"
    l_rate :float = 1e-4
    optimizer = torch.optim.AdamW
    w_decay = 1e-2
    lr_scheduler = 'onplateau'
    iteration = None
    factor = 0.5
    warmup = 3
    min_lr = 1e-6
    epochs = 10
    early_stopping = None
    criterion = None
    odname: str = "results"
    
class ModelArgs():
    
    model_name: str = ""
    esm_name = "facebook/esm2_t12_35M_UR50D"
    num_class: int = None
    distinct_labels: List[str] = None
    lora_dropout: float = 0.1
    lora_rank: int = 4
    lora_alpha: int = 32

