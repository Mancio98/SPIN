import argparse
from ast import Store
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Dict

from data.preprocessing import EsmSpanDataset, PreprocessData
from settings import BASE_PATH
from utils.evaluation_utils import EvalMetrics, build_confusion_matrix, evaluate
from utils.torch_utils import fix_random, get_device, load_model
from pathlib import Path

class InferencePipeline:

    def __init__(self, device, odpath='results') -> None:

        self.device = device
        
        self.odir_path = Path(odpath)
        
        self.odir_path.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, data_in, model_in, model_args=None, batch_size=None, save=False, verbose=False, prefix=""
    ):  

        loader, dataframe = self.prepare(data_in, model_in, model_args, batch_size)
        
        predictions = self.inference(loader, verbose)

        labels = self.model.config.labels

        if dataframe is not None:

            if not Path(data_in).is_absolute():
                data_in = BASE_PATH / data_in

            dataframe['prediction'] = [labels[pred] for pred in predictions['y_pred']]
            dataframe['probability'] = predictions['y_prob']
            dataframe['start_pred'] = predictions['start_pred']
            dataframe['end_pred'] = predictions['end_pred']

            print(dataframe)
            if save:
                dataframe.to_csv(self.odir_path / f"{data_in.stem}_pred.csv", index=False)

        return predictions

    def prepare(self, data_in, model_in, model_args=None, batch_size=16):

        # load model 
        self.model = None
        if isinstance(model_in, str): # must be a ckpt path
            
            fpath = Path(model_in)

            if not fpath.is_absolute():
                fpath = BASE_PATH / fpath

            self.model = load_model(model_in, model_args)

        elif isinstance(model_in, nn.Module):
            self.model = model_in
        else:
            raise Exception("Model or ckpt path not provided")
        
        # load data
        dataframe = None
        if isinstance(data_in, DataLoader):
            loader = data_in
        else:
            if isinstance(data_in, str):
        
                dataframe = PreprocessData(data_in).get_data()

            elif isinstance(data_in, list) and all(isinstance(item, str) for item in data_in):
                
                seq_ids = [f'seq_{i}' for i in range(len(data_in))]

                dataframe = pd.DataFrame({'seq_id': seq_ids, 'sequence': data_in})

            dataset = EsmSpanDataset(dataframe)
            loader = dataset.get_data_loaders(
                model_version=self.model.config.version, batch_size=batch_size, shuffle=False
            )[0]
            

        return loader, dataframe

    def inference(self, loader, verbose=1) -> Dict:
        
        criterion = torch.nn.functional.cross_entropy
        metric = EvalMetrics(len(self.model.config.labels))

        tot_loss, tot_loss_span, start_acc, end_acc, start_pred, end_pred = evaluate(
            self.model, criterion, metric, loader, verbose=verbose, device=self.device, return_span=True
            )

        print(f"Total loss: {tot_loss : 4f} - "
              f"Span loss: {tot_loss_span : 4f} - "
              f"Span start acc: {start_acc: 4f} ± 5 - "
              f"Span end acc: {end_acc: 4f} ± 5"
            )
        
        return {'y_true': metric.y_true,
                'y_pred': metric.y_pred,
                'y_prob': metric.y_prob,
                'start_pred': start_pred,
                'end_pred': end_pred
                }
if __name__ == "__main__":
    # options for using from the command line

    parser = argparse.ArgumentParser(description='My parser')#
    # input filename ifpath
    parser.add_argument(
        "-f", "--sequence", dest="data", required=True, help="File path fasta or csv or string of sequences"
        )
    parser.add_argument(
        "-c", "--ckptpath", dest="fpath", type=str, required=True, help="File path for checkpoint"
        )
    parser.add_argument(
        "-o", "--odpath", dest="odpath",  type=str, default="results", help="Output directory path relative or absolute"
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
        "-v","--verbose", dest="verbose", type=int, default=1, help="verbose"
        )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=42, help="random seed"
        )

    options = parser.parse_args()

    seed = options.seed
    fix_random(seed)

    device = get_device() if options.device == 'auto' else torch.device(options.device)

    data_in = options.data
    
    pipeline = InferencePipeline(device, options.odpath)
    predictions = pipeline(
            data_in=data_in,
            model_in=options.fpath,
            batch_size=options.batch_size,
            save=options.save,
            verbose=options.verbose,
            prefix=options.prefix
        )
