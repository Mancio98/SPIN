import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from collections import defaultdict


def make_splits(eval_size, targets, seed, stratify=True):

    data_idx = np.arange(len(targets))

    train_idx, test_idx = train_test_split(
        data_idx, test_size=eval_size, random_state=seed, stratify=targets if stratify else None
        )
    
    split_dict = defaultdict(pd.DataFrame)

    split_dict['train'] = train_idx
    split_dict['val'] = test_idx

    return split_dict

def make_plot_distribution(dataset_info, len_datasets, class2colors):
        
    x = np.arange(len_datasets) # the label locations
    width = 0.1
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    new_dict = defaultdict(dict)

    for value in dataset_info.keys():
        for att in dataset_info[value]:
           
            new_dict[att][value] = dataset_info[value][att]

    for attribute in new_dict.keys():
        offset = width * multiplier

        color = class2colors[attribute] if class2colors is not None else None
        rects = ax.bar(
            x + offset , list(new_dict[attribute].values()), width, label=attribute, color=color
            )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('# Samples')
    ax.set_title('Classes distribution')
    ax.set_xticks(x + width*2, ["Train","Test","Val"])
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 10000)

    plt.show()
    plt.clf()

def compute_dataset_info(labels, **datasets): #y_train, y_test, y_val):
    """
    Count number of class elements for each dataset
    """
    def _get_split_count(splitlabel, labels):
        """
        Count elements for each dataset
        """
        split_info = {}

        foundlabels, countlabels = np.unique(splitlabel, axis=0, return_counts=True)

        for i,label in enumerate(foundlabels):
            split_info[labels[label]] = countlabels[i]

        return split_info
    
    train_val_test_len = {}
    dataset_info = {}

    for key in datasets.keys():

        train_val_test_len[key] = len(datasets[key])
        dataset_info[key] =_get_split_count(datasets[key], labels)

    return train_val_test_len, dataset_info