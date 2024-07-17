import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def make_splits(_data, _data_intlabel, data_targets, seed, val_set=True, stratify=True):

    data_idx = np.arange(0,len(_data_intlabel),dtype=int)

    train_idx, test_idx, train_y, test_y = train_test_split(data_idx, data_targets, test_size=0.2, random_state=seed,
                                                                stratify=_data_intlabel if stratify else None)
    
    test_data = _data[test_idx]

    if not val_set:
        train_data = _data[train_idx]

        return {'train': (train_data, train_y),
                'test' : (test_data, test_y)}
    
    train_idx, val_idx, train_y, val_y = train_test_split(train_idx, train_y, test_size=0.2, random_state=seed,
                                                            stratify=_data_intlabel[train_idx] if stratify else None)

    train_data = _data[train_idx]
    val_data = _data[val_idx]
    
    return {'train': (train_data, train_y),
            'val'  : (val_data, val_y),
            'test' : (test_data, test_y)}

def make_plot_distribution(dataset_info, len_datasets):
        
    x = np.arange(len_datasets) # the label locations
    width = 0.1
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    new_dict = dict()

    for value in dataset_info.keys():
        for att in dataset_info[value]:
            if att not in new_dict:
                new_dict.update({att: dict()})

        new_dict[att].update({value : dataset_info[value][att]})

    print(new_dict)
    for attribute in new_dict.keys():
        offset = width * multiplier
        rects = ax.bar(x + offset , list(new_dict[attribute].values()), width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('# Samples')
    ax.set_title('Classes distribution')
    ax.set_xticks(x + width*2, ["Train","Val","Test"])
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
        dataset_info[key] =_get_split_count(datasets[key])

    return train_val_test_len, dataset_info