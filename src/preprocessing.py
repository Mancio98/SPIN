
import numpy as np
import pandas as pd
import torch

from utils.data_utils import compute_dataset_info, make_plot_distribution, make_splits
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from typing import Dict


class PreprocessData:

    def __init__(self, fpath, max_len=1024) -> None:

        self._data_path = fpath
        self._max_len = max_len

        self._prepare_data()


        
    def _prepare_data(self) -> None:

        data_df = pd.read_csv(self._data_path, sep=',')

        data_id = data_df['id_seq'].values
        data_seq = data_df['sequence'].values
        # data_start = data_df['start'].values
        # data_end = data_df['end'].values
        data_label = data_df['group'].values

        data_idx_filter = self._filter_idx_by_length(data_seq)

        data_id = data_id[data_idx_filter]
        data_seq = data_seq[data_idx_filter]
        # data_start = data_start[data_idx_filter]
        # data_end = data_end[data_idx_filter]
        self._targets = data_label[data_idx_filter]

        self._data = np.array([seq.replace('-','').upper() for seq in data_seq])

        self._labels = np.unique(data_label)

        self._data_intlabel = np.zeros(self._targets.shape, dtype=int)

        for i,label in enumerate(data_label):

            self._data_intlabel[i] = np.where(self._labels == label)[0][0]

        #data_targets = np.column_stack((_data_intlabel,data_start,data_end))

    def _filter_idx_by_length(self, seqs):

        data_len_list = np.array([len(seq) for seq in seqs])

        data_idx_filter = (data_len_list <= self._max_len).nonzero()[0]

        return data_idx_filter
    
    def get_data(self):

        return self._data, self._targets

class MyDataset:

    def __init__(self, data, targets=None) -> None:
        
        self.data = data
        self.targets = targets

        if self.targets:
            self._labels = np.unique(self.targets)

            self._data_intlabel = np.zeros(self._targets.shape, dtype=int)

            for i,label in enumerate(self._labels):

                self._data_intlabel[i] = np.where(self._labels == label)[0][0]
        
        self._is_splitted = False
        
    def split_data(self, seed, val_set=True, stratify=True) -> None:

        if not self.targets:

            print("No meaning in splitting without targets")
            return
        
        splits = make_splits(self.data, self._data_intlabel, self._targets, seed, val_set, stratify=stratify)

        self._train_data, self._train_y = splits['train']
        self._test_data, self._test_y = splits['test']

        if val_set:
            self._val_data, self._val_y = splits['val']

        self._is_splitted = True

    def get_data_loaders(self, model_name, batch_size=8) ->  Dict|DataLoader:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self._is_splitted:

            train_ds = _MyDataset(self._train_data, self._train_y, tokenizer)
            test_ds = _MyDataset(self._test_data, self._test_y, tokenizer)

            #TODO check for pin memory
            train_loader = DataLoader(train_ds,sampler=BatchSampler(RandomSampler(train_ds),batch_size=batch_size,drop_last=False), batch_size=None, pin_memory=True)
            test_loader = DataLoader(test_ds, sampler=BatchSampler(SequentialSampler(test_ds), batch_size=batch_size, drop_last=False), batch_size=None, pin_memory=True)
            

            if self._val_set:
                val_ds = _MyDataset(self._val_data, self._val_y,tokenizer)
                val_loader = DataLoader(val_ds, sampler=BatchSampler(SequentialSampler(val_ds), batch_size=batch_size, drop_last=False), batch_size=None, pin_memory=True)

                return {    'train_loader': train_loader,
                            'val_loader'  : val_loader,
                            'test_loader' : test_loader}
            
            return {    'train_loader': train_loader,
                        'test_loader' : test_loader}
        else:

            ds = _MyDataset(self.data,self.targets, tokenizer)
            loader = DataLoader(ds, sampler=BatchSampler(SequentialSampler(ds), batch_size=batch_size, drop_last=False), batch_size=None, pin_memory=True)

            return loader
        
    def _get_split_count(self, splitlabel):
        """
        Count elements for each dataset
        """
        split_info = {}

        foundlabels, countlabels = np.unique(splitlabel,axis=0,return_counts=True)

        for i,label in enumerate(foundlabels):
            split_info[self._labels[label]] = countlabels[i]

        return split_info

    def plot_distribution(self):

        if not self._is_splitted:
            self.split_data(self.seed, self._val_set, self._stratify)

        train_val_test_len, dataset_info = compute_dataset_info(self._train_y, self._test_y, self._val_y)
        make_plot_distribution(dataset_info, self._train_y,self._test_y,self._val_y)

class _MyDataset(Dataset):

    def __init__(self, data, label, tokenizer):

        self.X = data
        self.y = torch.LongTensor(label)
        
        self.tokenizer = tokenizer
        self.num_class = np.unique(label)[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, batch_idx):

        batch_x = self.tokenizer.batch_encode_plus([self.X[ix] for ix in batch_idx],
                                            return_tensors="pt",
                                            padding=True)
        batch_y = self.y[batch_idx]

        return batch_x, batch_y