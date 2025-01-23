from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from utils.data_utils import compute_dataset_info, make_plot_distribution, make_splits
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import Dict

from sklearn.utils.class_weight import compute_class_weight
from settings import BASE_PATH

class PreprocessData:

    def __init__(self, fpath, max_len=1024) -> None:

        self._data_input = Path(fpath)
        self._max_len = max_len

        self._base_dir = BASE_PATH

        self._prepare_data()
        
    def _prepare_data(self) -> None:

        print(self._data_input)
        if not self._data_input.is_absolute():
                self._data_input = BASE_PATH / self._data_input 

        if not self._data_input.exists():
            raise Exception("path doesn't exist")
        
        if self._data_input.suffix == ".csv":
            data_df = self._process_csv()
        elif self._data_input.suffix == ".fasta": 
            data_df = self._process_fasta()
        else:
            raise Exception("format not supported (only csv or fasta)")

        
        self._data_cleaning(data_df)
        
    def _process_fasta(self):

        with open(self._base_dir / self._data_input, 'r') as f:

            seqs = f.read().split('>')[1:]
            fasta_id = [s.split()[0] for s in seqs]
            fasta_seq = [''.join(s.split('\n')[1:]) for s in seqs]

        return pd.DataFrame({'id_seq':fasta_id, 'sequence':fasta_seq})

    def _process_csv(self):
        
        return pd.read_csv(self._base_dir / self._data_input, sep=',')

    def _data_cleaning(self, data_df):

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

class EsmSpanDataset:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        
        self.dataframe = dataframe

        if 'group' in self.dataframe.columns:
            
            pd_cat = pd.Categorical(self.dataframe['group'])

            self.labels = pd_cat.categories
            self.dataframe['label'] = pd_cat.codes
        
        self._is_splitted = False
        
    def split_data(self, seed, val_size=0.2, stratify=True) -> None:

        if 'label' not in self.dataframe.columns:

            print("No meaning in splitting without targets")
            return
        
        if val_size == 0.0:
            print("No meaning in splitting with eval size equals to 0")
            return
        
        splits = make_splits(val_size, self.dataframe['label'], seed, stratify=stratify)

        self.train_df = self.dataframe.iloc[splits['train']].reset_index(names=['old_index'])
        self.val_df = self.dataframe.iloc[splits['val']].reset_index(names=['old_index'])

        print(f"Splits: train={len(splits['train'])} val={len(splits['val'])}")
        
        self._is_splitted = True

    def compute_class_weights(self):

        dataframe = self.train_df if self._is_splitted else self.dataframe

        class_weights = compute_class_weight(
            class_weight="balanced", classes=dataframe['label'].unique(), y=dataframe['label']
            )

        return class_weights

    def get_data_loaders(self, model_version, batch_size=8, shuffle=True) ->  Dict[str,DataLoader]|DataLoader:

        tokenizer = AutoTokenizer.from_pretrained(model_version)


        if self._is_splitted:

            train_ds = _MyDataset(self.train_df, tokenizer)
            loader_dict = defaultdict(DataLoader)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4
                )
            
            val_ds = _MyDataset(self.val_df, tokenizer)
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
                )
            
            return [train_loader, val_loader]
        
        else:

            ds = _MyDataset(self.dataframe, tokenizer)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4
                )

            return [loader]
        
    def plot_distribution(self, class2colors=None):

        if self._is_splitted:
            
            datasets = {}
            datasets['train'] = self.train_df['label']
            datasets['test'] = self.test_df['label']
            if self.val_df is not None:
                datasets['val'] = self.val_df['label']
        else:
            datasets = {'dataset':self.dataframe['label']}

        datasets_len, dataset_info = compute_dataset_info(self.labels, **datasets)
        make_plot_distribution(dataset_info, len(datasets), class2colors)

class _MyDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, tokenizer=None):
        
        self.X = dataframe['sequence']
        
        self.start_dom, self.end_dom = None, None

        if 'start' in dataframe.columns:
            self.start_dom = dataframe['start']
        
        if 'end' in dataframe.columns:
            self.end_dom = dataframe['end']

        self.y = None
        if 'label' in dataframe.columns:
            self.y = torch.LongTensor(dataframe['label'])
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, idx):

        batch_x = self.X[idx]
        
        if self.tokenizer is not None:

            token_dict = self.tokenizer(batch_x, max_length=1024, padding='max_length', truncation=True, return_tensors='pt')
            batch_x = {k:v.squeeze(0) for k,v in token_dict.items()}

        batch_start, batch_end = None, None
        
        if self.start_dom is not None and self.end_dom is not None:
            batch_start = self.start_dom[idx]
            batch_end = self.end_dom[idx]
    
        batch_y = None
        if self.y is not None:
            batch_y = self.y[idx]

        input_dict = dict(input=batch_x, start=batch_start, end=batch_end)

        return input_dict, batch_y