import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.ProfileESM import ProfileESM
from torch.utils.data import DataLoader

def load_network(ckpt_weights, model_args):

    checkpoint = os.path.join("TRX_30k/checkpoints",ckpt_weights)

    net = ProfileESM(model_args)
    
    state_dict = torch.load(checkpoint)
    net.load_state_dict(state_dict)
    
    return net

@torch.no_grad
def evaluate(model: nn.Module, loader: DataLoader, device, criterion, output_attentions=False):

    model = model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    attention_list = {i+1:[]for i in range(12)} # 12 number of layers

    correct_pred, num_samples, total_loss = 0, 0, 0.

    for data_map, targets in loader:
 
        inputs = {k: v.to(device) for k, v in data_map.items()}
        targets = targets.to(device)

        class_targets = targets[:,0]

        outputs = model(**inputs)

        class_logits = outputs.logits

        if output_attentions:

            cls_attentions_layers = [layer[:,:,0,:].cpu().numpy() for layer in outputs['attentions']]

            for i,layer_attent in enumerate(cls_attentions_layers):
                
                max_scores_heads = np.max(layer_attent, axis=1) #(batch, seq_len)
                
                indices = max_scores_heads #[(scores > 0.1).nonzero()[0] for scores in max_scores_heads]

                attention_list[i+1].append(indices)
        
        loss = criterion(class_logits.view(-1, model.num_class), class_targets.view(-1))
        
        total_loss += loss.item() * len(targets)

        #compute accuracy
        probs = F.softmax(class_logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1)
        target_id = class_targets.detach()

        correct_pred += (pred_id == target_id).sum().item()

        num_samples += len(targets)

        y_pred.extend(pred_id.cpu().numpy())
        y_true.extend(class_targets.cpu().numpy())

    y_pred, y_true = np.array(y_pred), np.array(y_true)

    total_loss /= num_samples
    total_acc = 100. * correct_pred / num_samples

    return {'y_pred':y_pred,
            'y_true':y_true,
            'loss':total_loss,
            'accuracy':total_acc,
            'attentions': attention_list
        }