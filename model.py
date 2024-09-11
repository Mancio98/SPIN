import torch

from typing import List, Optional
import torch.nn as nn

from transformers import EsmForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

from config import ModelArgs

def get_model(args: ModelArgs) -> nn.Module:

    esm = EsmForSequenceClassification.from_pretrained(args.esm_name, num_labels=args.num_class)

    for param in esm.esm.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules= ["query","key","value"],
    )

    model = get_peft_model(esm, peft_config) # transform our classifier into a peft model
    model.print_trainable_parameters()

    model.config.label2id = {key:id for id, key in enumerate(args.distinct_labels)}
    model.config.id2label = {id:key for id, key in enumerate(args.distinct_labels)}

    return model