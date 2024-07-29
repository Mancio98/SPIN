import argparse

import torch
import torch.nn as nn
from typing import Dict

from model.ProfileESM import ProfileESM
from data.preprocessing import MyDataset, PreprocessData
from trainer.trainer_profilesm import TrainProfileESM
from utils.evaluation_utils import evaluate, load_network
from utils.torch_utils import fix_random, get_device
from config import TrainArgs, ModelArgs

from settings import BASE_PATH


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
            train_args, model_args, train_loader, val_loader = outputs

            results = self.train(
                train_args, model_args, train_loader, val_loader
            )

        elif self.mode == 'manual':

            results = self.train(train_args, model_args, train_loader=train_loader,val_loader=val_loader)

        
        return results

    def prepare_auto(
            self, data_in, train_args, batch_size, validate
            ):


        fpath = data_in

        if (self.base_dir/ fpath).exists() or not fpath.endswith(".csv"):
            raise Exception("file type incorrect or path doesn't exist")
        
        model_args = ModelArgs()

        data, targets = PreprocessData(fpath).get_data()

        dataset = MyDataset(data, targets)
        dataset.split_data(self.seed,val_set=validate)
        
        model_args.distinct_labels = dataset.labels
        model_args.num_class = len(dataset.labels)

        class_weights = dataset.compute_class_weights()
        train_args.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        loaders_dict = dataset.get_data_loaders(
            model_name=model_args.esm_name, batch_size=batch_size
        )

        return train_args, model_args, loaders_dict['train_loader'], loaders_dict['val_loader']

    
    def train(self, args: TrainArgs, model_args: ModelArgs, train_loader, val_loader=None) -> Dict:
        
        model = ProfileESM(model_args)

        trainer = TrainProfileESM(
            args.nametrain, self.device, args.odname
        )

        optimizer = args.optimizer(model.parameters(), lr=args.l_rate)

        lr_scheduler = None
        if args.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=1,
                                                        threshold=0.0001, threshold_mode='rel',
                                                        cooldown=0, min_lr=1e-5)
        result = trainer.fit(
            model,
            train_loader,
            val_loader,
            args.criterion,
            args.epochs,
            optimizer,
            args.early_stopping,
            lr_scheduler,
        )

        return result

    def inference(self, loader, model_args, ckpt_path) -> Dict:
        
        if ckpt_path:
            model = load_network(ckpt_path, model_args)
        else:
            model = ProfileESM(**model_args)

        criterion = nn.CrossEntropyLoss

        outputs = evaluate(model, loader, self.device, criterion)

        return outputs

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
        "-s", "--seed", dest="seed", type=int, default=42, help="random seed"
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
        "--no-scheduler", dest="lr_scheduler", type=bool, action='store_false', default=True, help="Doesn't apply ReduceLROnPlateau"
        )

    parser.add_argument(
        "-b","--batchsize", dest="batch_size", type=int, default=8, help="batch size"
        )
    parser.add_argument(
        "--no-valset", dest="validate", type=bool, action='store_false', default=True, help="No validation set"
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

    pipeline = TrainingPipeline(seed=seed, mode='train', device=device)
    pipeline(
        data_path=options.fpath, train_args=train_args, batch_size=options.batch_size, validate=options.validate
        )