import argparse
from model import DomainSpanESM
from trainer.trainer_domspanesm import TrainDomSpanESM

import torch

from typing import Dict

from data.preprocessing import EsmSpanDataset, PreprocessData
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
        self, train_args: TrainArgs, data_in=None, batch_size=8, train_loader=None, val_loader=None, model_args: ModelArgs=None
    ) -> Dict:
        
        results = None

        if self.mode == 'auto':
            
            outputs = self.prepare_auto(data_in, train_args, batch_size)
            train_args, model_args, loaders = outputs

            results = self.train(
                train_args, model_args, *loaders
            )


        elif self.mode == 'manual': # useful if imported

            results = self.train(train_args, model_args, train_loader=train_loader,val_loader=val_loader)
        
        return results

    def prepare_auto(
            self, fpath, train_args, batch_size
            ):

        dataframe = PreprocessData(fpath).get_data()

        dataset = EsmSpanDataset(dataframe)
        dataset.split_data(self.seed, val_size=train_args.val_size)
        # dataset.plot_distribution()

        model_args = ModelArgs()
        model_args.labels = dataset.labels
        model_args.num_class = len(dataset.labels)

        class_weights = dataset.compute_class_weights()
        train_args.class_weights = class_weights

        print(f"Weights per class: {" - ".join([f"{model_args.labels[i]}: {v: .2f}" for i,v in enumerate(class_weights)])}")

        train_args.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        )

        loaders = dataset.get_data_loaders(
            model_version=model_args.version, batch_size=batch_size
        )


        return train_args, model_args, loaders

    
    def train(self, args: TrainArgs, model_args: ModelArgs, train_loader, val_loader=None) -> Dict:
        
        model = DomainSpanESM(model_args)

        trainer = TrainDomSpanESM(
            args.verbose, self.device, args.train_name, args.odname
        )

        result = trainer.fit(
            args,
            model,
            train_loader,
            val_loader,
            args.save,
            args.verbose
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
        "-optim", dest="optimizer", type=str, default='adam', choices=['adam','adamw'], help="optimizer"
        )
    parser.add_argument(
        "-sched", dest="lr_scheduler", type=str, default=None, choices=['warmuplinear', 'warmupcos', 'onplateau','exponential'], help="Lr scheduler"
        )

    parser.add_argument(
        "-b","--batchsize", dest="batch_size", type=int, default=8, help="batch size"
        )
    parser.add_argument(
        "-val", dest="val_size", type=float, default=0.2, help="validate set size"
        )
    parser.add_argument(
        "-log", dest="log_interval", type=int, default=50, help="log"
        )
    parser.add_argument(
        "--save", dest="save", action='store_false', default=True, help="Save checkpoints"
    )
    parser.add_argument(
        "-v","--verbose", dest="verbose", type=int, default=1, help="verbose"
    )
    
    options = parser.parse_args()

    seed = options.seed
    fix_random(seed)

    device = get_device() if options.device == 'auto' else options.device

    train_args = TrainArgs()
    train_args.train_name = options.name_train
    train_args.l_rate = options.lr
    train_args.epochs = options.epochs
    train_args.odname = options.odname
    train_args.optimizer = options.optimizer
    train_args.lr_scheduler = options.lr_scheduler
    train_args.log_interval = options.log_interval
    train_args.verbose = options.verbose
    train_args.val_size = options.val_size

    pipeline = TrainingPipeline(seed=seed, mode='auto', device=device)
    results = pipeline(
        data_in=options.fpath, train_args=train_args, batch_size=options.batch_size
        )