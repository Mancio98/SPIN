from typing import List
import torch


class TrainArgs():
    
    nametrain = "finetuning"
    l_rate :float = 1e-3
    optimizer = torch.optim.AdamW
    lr_scheduler = True
    epochs = 10
    early_stopping = None
    criterion = None
    odname: str = "results"
    
class ModelArgs():
    
    model_name: str = "",
    esm_name = "facebook/esm2_t12_35M_UR50D"
    num_class: int = None,
    distinct_labels: List[str] = None,
    lora_dropout: int = 0.1

