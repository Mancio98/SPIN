
from collections import defaultdict

import pandas as pd
import seaborn as sn

from sympy import N
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


from sklearn.metrics import confusion_matrix,  accuracy_score, f1_score
from sklearn.utils import compute_class_weight, compute_sample_weight

from matplotlib import pyplot as plt

    
def build_confusion_matrix(y_true, y_pred, tags, color_map=None, verbose=False, odpath=None, prefix=""):
    ''' Builds and plots the confusion matrix '''

    def annot_colors(ax):
                
         # Apply colors to the x and y tick labels
        for label in ax.get_xticklabels():
            label.set_color(color_map[label.get_text()])
        
        for label in ax.get_yticklabels():
            label.set_color(color_map[label.get_text()])

    print(y_true, y_pred)
    cf = confusion_matrix(y_true, y_pred, normalize='true')

    print(tags)
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

class EvalMetrics:

    def __init__(self, n_labels, class_weights=None):

        self.num_labels = n_labels

        if class_weights and not isinstance(class_weights, dict):
            class_weights = dict(zip(range(len(class_weights)),class_weights))

        self.class2weights = class_weights

        self.accuracy = []

        self.f1score = []

        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def clear(self):

        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def step(self):
        """
        Calculate and collect metrics
        """
        report = self.compute()

        self.accuracy.append(report['accuracy'])
        self.f1score.append(report['f1_score'])

        self.clear()

    def update(self, y_true, y_pred, y_prob=None):

        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

        if y_prob is not None:
            self.y_prob.extend(y_prob)

    def compute(self) -> defaultdict:
        """
        Compute two evaluation metrics:
            - f1score
            - accuracy
        All metrics are inversely frequency weighted with sklearn.utils.compute_class_weight
        """

        sample_weights = None
        if self.class2weights is not None:
            sample_weights = compute_sample_weight(self.class2weights, self.y_true)
        
        report = defaultdict()

        report['accuracy'] = accuracy_score(
            self.y_true, self.y_pred, sample_weight=sample_weights
            )
        report['f1_score'] = f1_score(
            self.y_true, self.y_pred, average='macro', sample_weight=sample_weights, zero_division=1.0
            )

        return report


def compute_class_weights(num_labels, y_true):

    founded_classes = np.unique(y_true)

    founded_class_weight = compute_class_weight(
        class_weight="balanced", classes=founded_classes, y=y_true
        )
    
    class_weights = torch.ones(num_labels)

    for c,w in zip(founded_classes, founded_class_weight):
        class_weights[c] = w

    return class_weights

@torch.no_grad()
def evaluate(model, criterion, metrics: EvalMetrics, loader, verbose , device, return_span=False):

    num_samples = 0

    tot_loss = 0.0
    tot_loss_span = 0.0

    start_correct_pred, end_correct_pred = 0,0
    pbar = tqdm(range(len(loader))) if verbose > 1 else None

    iterator = zip(pbar, loader) if verbose > 1 else enumerate(loader)

    start_pred, end_pred = list(), list()

    model.to(device)
    model.eval()
    for i, (inputs, targets) in iterator:

        seq_input = {k:v.to(device) if v is not None else None for k,v in inputs['input'].items()}
        
        start_pos = None
        end_pos = None
        if inputs['start'] is not None and inputs['end'] is not None:
            start_pos = inputs['start'].to(device)
            end_pos = inputs['end'].to(device)

        targets = targets.to(device)

        logits, span_loss, (start_pred_pos, end_pred_pos) = model(seq_input, start_pos, end_pos)

        loss_class = criterion(logits.view(-1, metrics.num_labels), targets.view(-1))

        loss = 0.65 * loss_class + 0.35 * span_loss

        tot_loss += loss.item() * len(targets)
        tot_loss_span += span_loss.item() * len(targets)

        num_samples += len(targets)

        y_probs = F.softmax(logits, dim=-1)
        y_pred = torch.argmax(y_probs, dim=-1).cpu().numpy()

        metrics.update(targets.cpu().numpy(), y_pred, y_probs.cpu().numpy())

        print(start_pos, start_pred_pos)
        print(end_pos, end_pred_pos)
        print(torch.abs(start_pos - start_pred_pos) <= 5)
        print(torch.abs(end_pos - end_pred_pos) <= 5)
        start_correct_pred += (torch.abs(start_pos - start_pred_pos) <= 5).sum()
        end_correct_pred += (torch.abs(end_pos - end_pred_pos) <= 5).sum()

        start_pred.extend(start_pred_pos.cpu().numpy())
        end_pred.extend(end_pred_pos.cpu().numpy())

    tot_loss /= num_samples
    tot_loss_span /= num_samples

    start_accuracy = start_correct_pred / num_samples
    end_accuracy = end_correct_pred / num_samples

    if return_span:
        return tot_loss, tot_loss_span, start_accuracy, end_accuracy, start_pred, end_pred
    else:
        return tot_loss, tot_loss_span, start_accuracy, end_accuracy