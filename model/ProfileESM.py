import torch

import torch.nn as nn

from transformers import EsmModel


from config import ModelArgs

class DomainSpanESM(nn.Module):

    def __init__(self, config: ModelArgs):

        super().__init__()

        self.config = config

        self.esm = EsmModel.from_pretrained(self.config.version, add_pooling_layer=False)

        for param in self.esm.parameters():
            param.requires_grad = False

        warm_layers = self.esm.encoder.layer[-6:]
        for param in warm_layers.parameters():
            param.requires_grad = True

        hidd_size = self.esm.config.hidden_size

        in_size = hidd_size
        out_size = hidd_size #128

        self.clf_head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(in_size, out_size),
            nn.Tanh(),
            nn.Dropout(self.config.dropout),
            nn.Linear(out_size, len(self.config.labels))   
        )

        self.span_outputs = nn.Linear(hidd_size, 2)

        self.init_weights()

    def init_weights(self):

        torch.nn.init.xavier_uniform_(self.clf_head[1].weight)
        torch.nn.init.xavier_uniform_(self.clf_head[-1].weight)
        # torch.nn.init.xavier_uniform_(self.fc.weight)

        torch.nn.init.xavier_uniform_(self.span_outputs.weight)
        
    def forward(self, inputs, start=None, end=None):
        
        last_hidd = self.esm(**inputs)[0]
        last_hidd = last_hidd[:,1:,:]   # remove cls token
        # x = last_hidd[:,0,:]
        
        span_logits = self.span_outputs(last_hidd)

        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        span_loss = 0.0
        if start is not None and end is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start = start.clamp(0, ignored_index)
            end = end.clamp(0, ignored_index)

            ce_span = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = ce_span(start_logits, start)
            end_loss = ce_span(end_logits, end)
            
            span_loss = (start_loss + end_loss) / 2.

        start_pred_pos = torch.argmax(start_logits, dim=-1)
        end_pred_pos = torch.argmax(end_logits, dim=-1)

        mask = torch.zeros_like(last_hidd, dtype=torch.bool, requires_grad=False)
        for i, (s,e) in enumerate(zip(start_pred_pos, end_pred_pos)):
            
            if s < e:
                mask[i,s:e+1,:] = True
            else:
                mask[i,:,:] = inputs['attention_mask'][i,1:].unsqueeze(-1)

        eps = 1e-8
        x = torch.sum(last_hidd * mask, dim=1) / (torch.sum(mask, dim=1) + eps) 
        
        logits = self.clf_head(x)

        return logits, span_loss, (start_pred_pos, end_pred_pos)