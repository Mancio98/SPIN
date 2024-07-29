import torch

from typing import List, Optional
import torch.nn as nn

from transformers import EsmForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

from config import ModelArgs

class ProfileESM(nn.Module):

    def __init__(self, args: ModelArgs):

        super(ProfileESM, self).__init__()


        self.model_name = args.model_name

        self.num_class = args.num_class

        esm = EsmForSequenceClassification.from_pretrained(args.esm_name, num_labels=self.num_class)

        for param in esm.esm.parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=args.lora_dropout, target_modules= ["query", "value"],
        )

        self.model = get_peft_model(esm, peft_config) # transform our classifier into a peft model
        self.model.print_trainable_parameters()

        self.config = self.model.config
        self.config.label2id = {key:id for id, key in enumerate(args.distinct_labels)}
        self.config.id2label = {id:key for id, key in enumerate(args.distinct_labels)}

    def forward( self, **inputs
        # input_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
        ):

        outputs = self.model(**inputs
                    # input_ids,
                    # attention_mask=attention_mask,
                    # position_ids=position_ids,
                    # head_mask=head_mask,
                    # inputs_embeds=inputs_embeds,
                    # output_attentions=output_attentions,
                    # output_hidden_states=output_hidden_states,
                    # return_dict=return_dict,
                )
       
        return outputs
 