import os

import torch.nn as nn
from typing import Dict

from model.ProfileESM import ProfileESM
from preprocessing import MyDataset, PreprocessData
from trainer.TrainProfileESM import TrainProfileESM
from utils.evaluation_utils import evaluate, load_network


class Pipeline:
    def __init__(self, mode, seed, device, fpath=None) -> None:
        self.data_path = fpath
        self.mode = mode
        self.seed = seed
        self.device = device

    def __call__(
        self, **args
    ):  # data: str, model_args, args=None, batch_size=8) -> Any:
        
        msg = []

        argskeys = list(args.keys())  
       
        if self.mode == "train":
            
            for key in ['data_path','model_args','train_args','batch_size','validate']:
                if key not in argskeys:
                    msg.append(f"{key} not provided")

            if len(msg) > 0:
                raise Exception(", ".join(msg))
           
            fpath = args['data_path']

            if not os.path.exists(fpath):
                raise Exception("file path doesn't exist")
            
            model_args = args['model_args']
            train_args = args['train_args']
            batch_size = args['batch_size']
            validate = args['validate']

            data, targets = PreprocessData(fpath).get_data()

            dataset = MyDataset(data, targets)
            dataset.split_data(self.seed,val_set=validate)
            
            loaders_dict = dataset.get_data_loaders(
                model_name=model_args.model_name, batch_size=batch_size
            )

            self.train(
                train_args,
                model_args,
                loaders_dict['train_loader'],
                loaders_dict['val_loader']
            )

        elif self.mode == 'kfold':

            for key in ['loaders','model_args','train_args','kfold']:
                if key not in argskeys:
                    msg.append(f"{key} not provided")

            if len(msg) > 0:
                raise Exception(", ".join(msg))

            train_loader = args['train_loader']
            val_loader = args['val_loader']
            model_args = args['model_args']
            train_args = args['train_args']

            self.train(train_args, model_args, train_loader=train_loader,val_loader=val_loader)

        elif self.mode in ['predict', 'inference']:
            
            for key in ['data','model_args','ckpt']:
                if key not in argskeys:
                    msg.append(f"{key} not provided")

            if len(msg) > 0:
                raise Exception(", ".join(msg))
       
            data = args['data']
            model_args = args['model_args']
            ckpt_path = args['ckpt_path']

            if ckpt_path and not os.path.exists(ckpt_path):
                raise Exception("ckpt path doesn't exist")
            
            dataset = MyDataset(data, None)
            dataset.split_data(self.seed,val_set=validate)
            
            loaders_dict = dataset.get_data_loaders(
                model_name=model_args.model_name, batch_size=batch_size
            )
            self.inference(data, model_args, ckpt_path)

    def train(self, args, model_args, train_loader, val_loader=None) -> Dict:
        

        # optimizer = args.optimizer(model.parameters(), lr=args.l_rate)

        # # lr_scheduler = get_scheduler(
        # #     name="linear", optimizer=train_args.optimizer, num_warmup_steps=0, num_training_steps=train_args.epochs * len(train_loader)
        # # )

        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        #                                             factor=0.5, patience=1,
        #                                             threshold=0.0001, threshold_mode='rel',
        #                                             cooldown=0, min_lr=1e-5)

        model = ProfileESM(**model_args)

        trainer = TrainProfileESM(
            args.nametrain, args.device, args.dir, args.ckpt_name_dir
        )

        result = trainer.fit(
            model,
            train_loader,
            val_loader,
            args.criterion,
            args.epochs,
            args.optimizer,
            args.early_stopping,
            args.lr_scheduler,
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