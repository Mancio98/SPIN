from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from traitlets import default

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

        print(f"Dataset shape: {data_df.shape}")
        print(data_df.head(10))

        #drop possible duplicates
        print("Drop duplicates...")
        data_df.drop_duplicates(subset=['id_seq'], inplace=True)
        print(f"New shape: {data_df.shape}")

        # remove possible gaps
        data_df['sequence'] = data_df['sequence'].apply(lambda s: s.replace('-','').upper())

        # remove sequence longer than the maximum length handled by the model
        print(f"Drop sequence with length > {self._max_len}...")
        filter_indices = self._filter_idx_by_length(data_df['sequence'])
        data_df = data_df.iloc[filter_indices].reset_index(drop=True)

        print(f"New shape: {data_df.shape}")

        self.data_df = data_df

    def _filter_idx_by_length(self, seqs):

        data_len_list = np.array([len(seq) for seq in seqs])

        data_idx_filter = (data_len_list <= self._max_len).nonzero()[0]

        return data_idx_filter
    
    def get_data(self):

        return self.data_df

class MyDataset:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        
        self.dataframe = dataframe

        if 'group' in self.dataframe.columns:
            
            pd_cat = pd.Categorical(self.dataframe['group'])

            self.labels = pd_cat.categories
            self.dataframe['label'] = pd_cat.codes
        
        self._is_splitted = False
        
    def split_data(self, seed, val_set=True, stratify=True, undersampling=False) -> None:

        if 'label' in self.dataframe.columns:

            print("No meaning in splitting without targets")
            return
        
        splits = make_splits(self.dataframe['labels'], seed, val_set, stratify=stratify)

        self.train_df = self.dataframe.iloc[splits['train']].reset_index(names=['old_index'])
        self.test_df = self.dataframe.iloc[splits['test']].reset_index(names=['old_index'])
        self.val_df = None

        if val_set:
            self.val_df = self.dataframe.iloc[splits['val']].reset_index(names=['old_index'])

        if undersampling:
            
            _, counts = np.unique(self.train_df['labels'], return_counts=True)

            sort_counts = np.argsort(counts)[::-1]

            first_class = sort_counts[0]
            second_class = sort_counts[1]

            first_idx = np.nonzero(self.train_df['labels'] == first_class)[0]

            remove_part = first_idx[counts[second_class]:]

            print(remove_part)

            train_idx = self.train_df.index.values
            test_idx = self.test_df.index.values

            removing_idx = train_idx[remove_part]
            train_idx = np.delete(train_idx, remove_part)
            test_idx = np.concatenate([test_idx, removing_idx])

            self.train_df.iloc[train_idx].reset_index(drop=True)
            self.test_df.iloc[test_idx].reset_index(drop=True)

        self._is_splitted = True

    def compute_class_weights(self):

        if not self._is_splitted:

            print("Data not splitted")
            return

        class_array, _ = np.unique(self._train_y, return_counts=True)

        class_weights = compute_class_weight(class_weight="balanced", classes=class_array, y=self.train_df['labels'])
        class2weight = {i:class_weights[i] for i in range(len(class_weights))}

        print(f"Weights per class: {class2weight}")
        
        return class2weight

    def get_data_loaders(self, model_name, batch_size=8) ->  Dict[str,DataLoader]|DataLoader:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self._is_splitted:

            train_ds = _MyDataset(self.train_df, tokenizer)
            test_ds = _MyDataset(self.test_df, tokenizer)

            loader_dict = defaultdict(DataLoader)
            loader_dict['train_loader'] = DataLoader(
                train_ds, batch_sampler=BatchSampler(RandomSampler(train_ds),batch_size=batch_size,drop_last=False), pin_memory=True, num_workers=4
                )
            loader_dict['test_loader'] = DataLoader(
                test_ds, batch_sampler=BatchSampler(SequentialSampler(test_ds), batch_size=batch_size, drop_last=False), pin_memory=True, num_workers=4
                )
            
            if self.val_df is not None:

                val_ds = _MyDataset(self.val_df, tokenizer)
                loader_dict['val_loader'] = DataLoader(
                    val_ds, batch_sampler=BatchSampler(SequentialSampler(val_ds), batch_size=batch_size, drop_last=False), pin_memory=True, num_workers=4
                    )
            
            return loader_dict
        
        else:

            ds = _MyDataset(self.dataframe, tokenizer)
            loader = DataLoader(
                ds, batch_sampler=BatchSampler(SequentialSampler(ds), batch_size=batch_size, drop_last=False), pin_memory=True, num_workers=4
                )

            return loader
        
    def plot_distribution(self, class2colors=None):

        if self._is_splitted:
            
            datasets = {}
            datasets['train'] = self.train_df['label']
            datasets['test'] = self.test_df['label']
            if self._val_y is not None:
                datasets['val'] = self.val_df['label']
        else:
            datasets = {'dataset':self.dataframe['label']}

        datasets_len, dataset_info = compute_dataset_info(self.labels, **datasets)

        make_plot_distribution(dataset_info, len(datasets), class2colors)

class _MyDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        
        self.X = dataframe['sequence']
        
        if 'label' in dataframe.columns:
            self.y = torch.LongTensor(dataframe['label'])

        self.data_df = None
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx):

        batch_x = self.tokenizer(
            self.X[idx], max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt'
            )
        
        batch_x = {key: val.squeeze(0) for key, val in batch_x.items()}  # Remove batch dimension

        batch_y = self.y[idx]
            
        return batch_x, batch_y