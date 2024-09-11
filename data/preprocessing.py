import numpy as np
import pandas as pd
import torch

from utils.data_utils import compute_dataset_info, make_plot_distribution, make_splits
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from typing import Dict

from sklearn.utils.class_weight import compute_class_weight
from settings import BASE_PATH, MAX_LENGTH

class PreprocessData:

    def __init__(self, data, max_len=1024) -> None:

        self._data_input = data
        self._max_len = max_len

        self._base_dir = BASE_PATH

        self._prepare_data()
        
    def _prepare_data(self) -> None:

        if self._data_input.endswith(".csv"):

            self._process_csv()
        elif self._data_input.endswith(".fasta"):
            
            #TODO 
            self._process_fasta()
        #elif is_sequence(self._data_input):

            
        #data_targets = np.column_stack((_data_intlabel,data_start,data_end))

    def _process_csv(self):
        
        
        data_df = pd.read_csv(self._base_dir / self._data_input, sep=',')

        data_df.drop_duplicates(subset=['id_seq'], inplace=True)

        # data_id = data_df['id_seq'].values
        data_seq = data_df['sequence'].values

        data_seq = np.array([seq.replace('-','').upper() for seq in data_seq])
        # data_start = data_df['start'].values
        # data_end = data_df['end'].values
        data_label = data_df['group'].values

        self._data_ids = self._filter_idx_by_length(data_seq)
        #self._data_ids = np.arange(len(data_seq), dtype=int)

        self._targets = data_label[self._data_ids]

        assert len(self._data_ids) == len(self._targets)

    def _filter_idx_by_length(self, seqs):

        data_len_list = np.array([len(seq) for seq in seqs])

        data_idx_filter = (data_len_list <= self._max_len).nonzero()[0]

        return data_idx_filter
    
    def get_data(self):

        return self._data_ids, self._targets

class MyDataset:

    def __init__(self, data_ids, targets=None) -> None:
        
        self.data_ids = data_ids

        self.targets = targets

        self.labels = np.unique(targets)

        self.targets = np.zeros(targets.shape, dtype=int)

        for i,label in enumerate(targets):

            self.targets[i] = np.where(self.labels == label)[0][0]
        
        self._is_splitted = False
        
    def split_data(self, seed, val_set=True, stratify=True, undersampling=False) -> None:

        if self.targets is None:

            print("No meaning in splitting without targets")
            return
        
        splits = make_splits(self.targets, seed, val_set, stratify=stratify)

        self._train_idx = self.data_ids[splits['train']]
        self._train_y = self.targets[splits['train']]

        self._test_idx = self.data_ids[splits['test']]
        self._test_y = self.targets[splits['test']]

        self._val_y = None

        if val_set:
            self._val_idx = self.data_ids[splits['val']]
            self._val_y = self.targets[splits['val']]

        if undersampling:

            _, counts = np.unique(self._train_y, return_counts=True)

            sort_counts = np.argsort(counts)[::-1]

            first_class = sort_counts[0]
            second_class = sort_counts[1]

            first_idx = np.array(self._train_y == first_class).nonzero()[0]
            
            remove_part = first_idx[counts[second_class]:]

            print(remove_part)

            removing_y = self._train_y[remove_part]
            removing_idx = self._train_idx[remove_part]

            self._train_y = np.delete(self._train_y, remove_part)
            self._train_idx = np.delete(self._train_idx, remove_part)

            self._test_y = np.concatenate([self._test_y,  removing_y])
            self._test_idx = np.concatenate([self._test_idx, removing_idx])

        self._is_splitted = True

    def compute_class_weights(self):

        if not self._is_splitted:

            print("Data not splitted")
            return

        class_array, class_sampls = np.unique(self._train_y, return_counts=True)

        tot_samples = class_sampls.sum()
        # class_weights = torch.FloatTensor([1. - (sample / tot_samples) for sample in class_sampls])


        class_weights = compute_class_weight(class_weight="balanced", classes=class_array, y=self._train_y)
        class2weight = {i:class_weights[i] for i in range(len(class_weights))}

        print(f"Weights per class: {class2weight}")
        
        return class2weight

    def get_data_loaders(self, model_name, fpath=None, batch_size=8) ->  Dict[str,DataLoader]|DataLoader:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self._is_splitted:

            train_ds = _MyDataset(self._train_idx, self._train_y, tokenizer, fpath)
            test_ds = _MyDataset(self._test_idx, self._test_y, tokenizer, fpath)

            train_loader = DataLoader(
                train_ds, batch_sampler=BatchSampler(RandomSampler(train_ds),batch_size=batch_size,drop_last=False), pin_memory=True, num_workers=4
                )
            test_loader = DataLoader(
                test_ds, batch_sampler=BatchSampler(SequentialSampler(test_ds), batch_size=batch_size, drop_last=False), pin_memory=True, num_workers=4
                )

            # train_loader = DataLoader(
            #     train_ds, batch_size=batch_size, drop_last=False,pin_memory=True, num_workers=4
            #     )
            # test_loader = DataLoader(
            #     train_ds, batch_size=batch_size, drop_last=False,pin_memory=True, num_workers=4
            #     )
            

            if self._val_y is not None:

                val_ds = _MyDataset(self._val_idx, self._val_y, tokenizer, fpath)
                val_loader = DataLoader(
                    val_ds, batch_sampler=BatchSampler(SequentialSampler(val_ds), batch_size=batch_size, drop_last=False), pin_memory=True, num_workers=4
                    )

                return {    'train_loader': train_loader,
                            'val_loader'  : val_loader,
                            'test_loader' : test_loader}
            
            return {    'train_loader': train_loader,
                        'test_loader' : test_loader}
        else:

            ds = _MyDataset(self.data_ids, self.targets, tokenizer, fpath)
            loader = DataLoader(
                ds, batch_sampler=BatchSampler(SequentialSampler(ds), batch_size=batch_size, drop_last=False), pin_memory=True, num_workers=4
                )

            return loader
        
    def plot_distribution(self, class2colors=None):

        if self._is_splitted:
            
            datasets = { 'train': self._train_y, 'test': self._test_y}
            if self._val_y is not None:
                datasets['val'] = self._val_y
        else:
            datasets = {'dataset':self.targets}

        datasets_len, dataset_info = compute_dataset_info(self.labels, **datasets)

        make_plot_distribution(dataset_info, len(datasets), class2colors)

class _MyDataset(Dataset):

    def __init__(self, data, label, tokenizer, fpath):

        self.X = data
        self.y = torch.LongTensor(label)

        self.data_df = None
        if fpath:
            self.data_df = pd.read_csv(BASE_PATH / fpath, sep=',')

        self.tokenizer = tokenizer
        self.num_class = np.unique(label)[0]

    def __len__(self):
        return len(self.y) 

    def __getitem__(self, batch_idx):

        if self.data_df is not None:
            batch_seqs = self.data_df.loc[self.X[batch_idx],['sequence']].values.tolist()
        else:
            batch_seqs = self.X[batch_idx]

        batch_x = self.tokenizer(
            batch_seqs, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt'
            )
        
        batch_x = {key: val.squeeze(0) for key, val in batch_x.items()}  # Remove batch dimension

        batch_y = self.y[batch_idx]
            
        return batch_x, batch_y