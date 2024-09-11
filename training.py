import argparse

import torch
import torch.nn as nn
from typing import Dict

from model import get_model
from data.preprocessing import MyDataset, PreprocessData
from trainer.trainer_profilesm import TrainProfileESM
from utils.evaluation_utils import evaluate, load_network
from utils.torch_utils import fix_random, get_device
from config import TrainArgs, ModelArgs

from settings import BASE_PATH
from utils.trainer_utils import get_scheduler


class TrainingPipeline:

    def __init__(self, mode, seed, device) -> None:

        self.mode = mode
        self.seed = seed
        self.device = device

        self.base_dir = BASE_PATH

    def __call__(
        self, train_args, data_in=None, batch_size=8, validate=False, train_loader=None, val_loader=None, model_args=None
    ) -> Dict:
        
        results = None

        if self.mode == 'auto':
            
            outputs = self.prepare_auto(data_in, train_args, batch_size, validate)
            train_args, model_args, loaders_dict = outputs

            results = self.train(
                train_args, model_args, loaders_dict['train_loader'], loaders_dict['val_loader']
            )

            results['test_loader'] = loaders_dict['test_loader']

        elif self.mode == 'manual':

            results = self.train(train_args, model_args, train_loader=train_loader,val_loader=val_loader)
        
        return results

    def prepare_auto(
            self, fpath, train_args, batch_size, validate
            ):


        if not (self.base_dir/ fpath).exists() or not fpath.endswith(".csv"):
            raise Exception("file type incorrect or path doesn't exist")
        
        data, targets = PreprocessData(fpath).get_data()

        dataset = MyDataset(data, targets)
        dataset.split_data(self.seed,val_set=validate)
        dataset.plot_distribution()

        model_args = ModelArgs()
        model_args.distinct_labels = dataset.labels
        model_args.num_class = len(dataset.labels)

        class_weights = dataset.compute_class_weights()
        train_args.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        loaders_dict = dataset.get_data_loaders(
            model_name=model_args.esm_name, fpath=fpath, batch_size=batch_size
        )

        return train_args, model_args, loaders_dict

    
    def train(self, args: TrainArgs, model_args: ModelArgs, train_loader, val_loader=None) -> Dict:
        
        #model = ProfileESM(model_args)
        model = get_model(model_args)

        trainer = TrainProfileESM(
            args.nametrain, self.device, args.odname
        )

        optimizer = args.optimizer(model.parameters(), lr=args.l_rate, weight_decay=args.w_decay)

        lr_scheduler = None
        if args.lr_scheduler:

            if args.lr_scheduler in ['cosine']:

                args.iteration = len(train_loader) * args.epochs
            else:
                args.iteration = args.epochs

            lr_scheduler = get_scheduler(args, optimizer)
           
        result = trainer.fit(
            model,
            train_loader,
            val_loader,
            args.criterion,
            args.epochs,
            optimizer,
            args.early_stopping,
            lr_scheduler,
            save_metrics=True
        )

        return result

if __name__ == "__main__":
    # options for using from the command line

    # EXAMPLE > training.py -f path_to_csv 
    parser = argparse.ArgumentParser(description='My parser')
    # input filename ifpath
    parser.add_argument(
        "-f", "--fpath", dest="fpath", default=None, help="File csv"
        )
    parser.add_argument(
        "-o", "--odname", dest="odname", default="results", help="Output directory name relative or absolute"
        )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=42, help="random seed"
        )
    parser.add_argument(
        "--device", dest="device", type=str, default='auto', choices=['auto','cpu','cuda','mps'], help="device"
        )

    parser.add_argument(
        "-n", "--nametrain", dest="name_train", type=str, default='finetune', help="output training name"
        )
    parser.add_argument(
        "-lr", dest="lr", type=float, default=1e-3, help="learning rate"
        )
    parser.add_argument(
        "-ep","--epochs", dest="epochs", type=int, default=10, help="training epochs"
        )
    parser.add_argument(
        "--no-scheduler", dest="lr_scheduler", action='store_false', default=True, help="Doesn't apply ReduceLROnPlateau"
        )

    parser.add_argument(
        "-b","--batchsize", dest="batch_size", type=int, default=8, help="batch size"
        )
    parser.add_argument(
        "--no-valset", dest="validate", action='store_false', default=True, help="No validation set"
        )

    options = parser.parse_args()

    seed = options.seed
    fix_random(seed)

    device = get_device() if options.device == 'auto' else options.device

    train_args = TrainArgs()
    train_args.nametrain = options.name_train
    train_args.l_rate = options.lr
    train_args.epochs = options.epochs
    train_args.odname = options.odname
    train_args.lr_scheduler = options.lr_scheduler

    pipeline = TrainingPipeline(seed=seed, mode='auto', device=device)
    results = pipeline(
        data_in=options.fpath, train_args=train_args, batch_size=options.batch_size, validate=options.validate
        )