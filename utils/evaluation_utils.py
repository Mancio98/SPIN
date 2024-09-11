
import enum
import pandas as pd
import seaborn as sn

from sympy import N
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from config import ModelArgs
from model import get_model
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.utils import compute_class_weight, compute_sample_weight

from settings import BASE_PATH
from matplotlib import pyplot as plt

from timeit import default_timer as timer

def load_network(ckpt_weights, model_args=None):

    checkpoint = BASE_PATH / ckpt_weights

    ckpt = torch.load(checkpoint)

    state_dict = ckpt['model_state_dict']
    config = ckpt['config']
    lora_config = config['lora_config']['default'].to_dict()
    if not model_args:

        model_args = ModelArgs()
        model_args.esm_name = config['esm_name']
        model_args.distinct_labels = config['labels']
        model_args.num_class = config['num_labels']
        model_args.lora_rank = lora_config['r']
        model_args.lora_alpha = lora_config['lora_alpha']
        model_args.lora_dropout = lora_config['lora_dropout']
        
    net = get_model(model_args)
    net.load_state_dict(state_dict)
    
    return net

def build_confusion_matrix(y_true, y_pred, tags, color_map=None, verbose=False, odpath=None, prefix=""):
    ''' Builds and plots the confusion matrix '''

    def annot_colors(ax):
                
         # Apply colors to the x and y tick labels
        for label in ax.get_xticklabels():
            label.set_color(color_map[label.get_text()])
        
        for label in ax.get_yticklabels():
            label.set_color(color_map[label.get_text()])

                
    cf = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(cf, index=tags, columns=tags)

    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar=True)

    if color_map is not None:
        annot_colors(ax)

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    if odpath is not None:

        fpath = odpath / f"{prefix + '_'}conf_matrix.png"
        plt.savefig(fpath)

    if verbose:
        plt.show()
    plt.clf()

    return df_cm

class MulticlassAccuracy:

    def __init__(self, labels) -> None:
        
        self.labels = labels

        self.y_true = []
        self.y_pred = []

    def compute(self, normalize=True):
        

        class_weights = compute_class_weights(len(self.labels), self.y_true)
        sample_weight = compute_sample_weight(class_weight={i:w for i,w in enumerate(class_weights)}, y=self.y_true)

        accuracy = accuracy_score(self.y_true, self.y_pred, sample_weight=sample_weight, normalize=normalize)
        
        return accuracy

    def __call__(self, y_true, y_pred):
        
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)


def compute_class_weights(num_labels, y_true):

    founded_classes = np.unique(y_true)

    founded_class_weight = compute_class_weight(
        class_weight="balanced", classes=founded_classes, y=y_true
        )
    
    class_weights = torch.ones(num_labels)

    for c,w in zip(founded_classes, founded_class_weight):
        class_weights[c] = w

    return class_weights

@torch.no_grad
def evaluate(model: nn.Module, loader: DataLoader, device, criterion, output_attentions=False):

    model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    y_prob = []

    attention_list = {i+1:[]for i in range(12)} # 12 number of layers

    total_loss = 0.0

    compute_accuracy = MulticlassAccuracy(range(model.num_labels))
    start_time = timer()
    with tqdm(range(len(loader))) as pbar:
        for i, (data_map, targets) in zip(pbar,loader):
    
            inputs = {k: v.to(device) for k, v in data_map.items()}
            class_targets = targets.to(device)

            outputs = model(**inputs)

            class_logits = outputs.logits

            if output_attentions:

                cls_attentions_layers = [layer[:,:,0,:].cpu().numpy() for layer in outputs['attentions']]

                for i,layer_attent in enumerate(cls_attentions_layers):
                    
                    max_scores_heads = np.max(layer_attent, axis=1) #(batch, seq_len)
                    
                    indices = max_scores_heads #[(scores > 0.1).nonzero()[0] for scores in max_scores_heads]

                    attention_list[i+1].append(indices)

            loss = criterion(class_logits.view(-1, model.num_labels), class_targets.view(-1))
            total_loss += loss.item() * len(targets)

            probs = F.softmax(class_logits, dim=-1)
            y_prob.extend(probs.cpu().numpy())

            pred = torch.argmax(probs, dim=-1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(class_targets.cpu().numpy())

            compute_accuracy(class_targets.cpu().numpy(), pred)

            pbar.set_postfix({
                        f'accuracy': np.round(compute_accuracy.compute(), 2),
                        })
        
            pbar.update(0)

    end_time = timer()

    tot_time = end_time - start_time

    print(f"total time (s): {tot_time :.4f} - Single sample time (s): {tot_time / len(y_true) :.4f}")

    class_weights = compute_class_weights(model.num_labels, y_true)

    sample_weight = compute_sample_weight(class_weight={i:w for i,w in enumerate(class_weights)}, y=y_true)

    tot_accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    tot_accuracy2 = compute_accuracy.compute()

    return {'y_pred':y_pred,
            'y_true':y_true,
            'y_prob':y_prob,
            'loss':total_loss/len(y_true),
            'accuracy':tot_accuracy,
            'accuracy2':tot_accuracy2,
            'attentions': attention_list
        }