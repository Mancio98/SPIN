import argparse
from ast import Store
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List

from config import ModelArgs
from data.preprocessing import MyDataset
from settings import BASE_PATH
from utils.evaluation_utils import build_confusion_matrix, evaluate, load_network
from utils.torch_utils import fix_random, get_device
from pathlib import Path

class InferencePipeline:

    def __init__(self, device, odpath='results') -> None:

        self.device = device
        
        self.odir_path = Path(odpath)
        
        self.odir_path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, data_in, model_in, model_args=None, batch_size=None,  save=False, verbose=False, prefix=""
    ):  

        loader = self.prepare(data_in, model_in, model_args, batch_size)
        
        predictions = self.inference(loader)

        labels = list(self.model.config.label2id.keys())

        build_confusion_matrix(
            predictions['y_true'], predictions['y_pred'], [labels[c] for c in np.unique(predictions['y_true'])] , verbose=verbose, odpath=self.odir_path if save else None, prefix=prefix
            )

        if isinstance(data_in, str):

            if data_in.endswith('.csv'):
                
                fpath = Path(data_in)
                
                if not fpath.is_absolute():

                    fpath = BASE_PATH / fpath

                df = pd.read_csv(str(fpath), sep=',')
                df['prediction'] = [self.model.config.id2label[pred] for pred in predictions['y_pred']]
                df['probability'] = predictions['y_prob']

                print()
                df.to_csv(self.odir_path / f"{fpath.stem}_pred.csv", index=False)
        return predictions

    def prepare(self, data_in, model_in, model_args, batch_size):

        self.model = None
        if isinstance(model_in, str): 
            # must be a ckpt path
            fpath = Path(model_in)

            if not fpath.is_absolute():
                fpath = BASE_PATH / fpath

            self.model = load_network(model_in, model_args)

        elif isinstance(model_in, nn.Module):
            self.model = model_in
        else:
            print("Model or ckpt path not provided")
            return
        
        if isinstance(data_in, str):
            
            if data_in.endswith('.csv'):
                
                fpath = Path(data_in)

                if not fpath.is_absolute():

                    fpath = BASE_PATH / fpath

                df = pd.read_csv(fpath, sep=',')
                data_in = df['sequence'].values.tolist()

        if isinstance(data_in, list) and all(isinstance(item, str) for item in data_in):
            
            num_labels = len(list(self.model.config.label2id.keys()))

            dataset = MyDataset(data_in, np.random.randint(0,num_labels,len(data_in)))
            loader = dataset.get_data_loaders(
                model_name=self.model.config._name_or_path, batch_size=batch_size if batch_size else 8
            )
        else:
            loader = data_in

        return loader

    def inference(self, loader) -> Dict:
        
        criterion = torch.nn.functional.cross_entropy

        outputs = evaluate(self.model, loader, self.device, criterion)

        return outputs

if __name__ == "__main__":
    # options for using from the command line

    parser = argparse.ArgumentParser(description='My parser')#
    # input filename ifpath
    parser.add_argument(
        "-f", "--sequence", dest="data", default=None, help="File path fasta or string of sequences"
        )
    parser.add_argument(
        "-c", "--ckptpath", dest="fpath", type=str, default=None, help="File path for checkpoint"
        )
    parser.add_argument(
        "-o", "--odname", dest="odpath",  type=str, default="results", help="Output directory path relative or absolute"
        )
    parser.add_argument(
        "-p", "--prefix", dest="prefix", type=str, default="", help="File prefix name"
        )
    parser.add_argument(
        "--device", dest="device", type=str, default='auto', choices=['auto','cpu','cuda','mps'], help="device"
        )
    parser.add_argument(
        "-b","--batchsize", dest="batch_size", type=int, default=16, help="batch size"
        )
    parser.add_argument(
        "-s","--save", dest="save", action='store_true', default=False, help="Save plots"
        )
    parser.add_argument(
        "-v","--verbose", dest="verbose", action='store_true', default=False, help="Show plots"
        )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=42, help="random seed"
        )

    options = parser.parse_args()

    seed = options.seed
    fix_random(seed)

    device = get_device() if options.device == 'auto' else torch.device(options.device)

    data_in = options.data
    
    pipeline = InferencePipeline(device)
    predictions = pipeline(
            data_in=data_in,
            model_in=options.fpath,
            batch_size=options.batch_size,
            odpath=options.odpath,
            save=options.save,
            verbose=options.verbose,
            prefix=options.prefix
        )
