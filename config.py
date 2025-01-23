from typing import List
import torch 
class TrainArgs:
    
    train_name = "finetuning"
    l_rate :float = 1e-4
    optimizer = torch.optim.AdamW
    optimizer_params = dict(weight_decay=0.01)
    w_decay = 1e-2
    lr_scheduler = None
    factor = 0.1
    epochs = 10
    criterion = None
    odname: str = "results"
    log_interval = 50
    class_weights: list|dict = None
    verbose = 1
    save = True
    val_size = 0.2


class ModelArgs:
    def __init__(self, 
                 model_name: str = "", 
                 version: str = "facebook/esm2_t12_35M_UR50D",
                 num_class: int = None, 
                 labels: List[str] = None,
                 freeze: bool = False, 
                 dropout: float = 0.1, 
                 max_length: int = 1024):
        self.model_name = model_name
        self.version = version
        self.num_class = num_class
        self.labels = labels
        self.freeze = freeze
        self.dropout = dropout
        self.max_length = max_length

    def to_dict(self) -> dict:
        
        return {
            "model_name": self.model_name,
            "version": self.version,
            "num_class": self.num_class,
            "labels": self.labels,
            "freeze": self.freeze,
            "dropout": self.dropout,
            "max_length": self.max_length,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            model_name=data.get("model_name", ""),
            version=data.get("version", "facebook/esm2_t12_35M_UR50D"),
            num_class=data.get("num_class"),
            labels=data.get("labels"),
            freeze=data.get("freeze", False),
            dropout=data.get("dropout", 0.1),
            max_length=data.get("max_length", 1024),
        )
