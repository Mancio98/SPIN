import os
import torch
import numpy as np
import json
import math

from typing import DefaultDict
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler
from timeit import default_timer as timer

from utils.trainer_utils import WarmupCosineLR

class TrainModel:

    def __init__(self, nametrain, device, path_dir="."):

        self.device = device
        self.name_train = nametrain
        self.path_dir = path_dir
        
        self._prepare_training_dirs()
        self._configure_logging()

    def _configure_logging(self):

        self.log_interval = 50

        log_dir = os.path.join(self.path_dir,"runs", self.name_train)

        self.writer = SummaryWriter(log_dir)

    def _prepare_training_dirs(self):

        self.ckpt_dir = os.path.join(self.path_dir,"checkpoints",self.name_train)
        os.makedirs(self.ckpt_dir, exist_ok=True)
    
    def _save_weights(self, model_name):
        
        path = os.path.join(self.ckpt_dir, model_name)
        #print(f"check_point path: {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                 'esm_name': self.model.config._name_or_path,
                 'num_labels': self.model.config.num_labels,
                 'labels': list(self.model.config.label2id.keys()),
                 'lora_config': self.model.peft_config['default'].to_dict()
                },
            }, path)

    def _train(self, train_loader, criterion, epoch):
        pass
    
    def _validation(self, val_loader, criterion, epoch):
        pass

    def fit(self, model, train_loader, val_loader, criterion, num_epochs, optimizer, early_stopping=None, lr_scheduler=None, verbose=True, save_metrics=False):

        self.model = model

        initial_lr = optimizer.param_groups[0]['lr']
        self.lr = optimizer.param_groups[0]['lr']
        self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        #collect loss and metrics
        train_loss_list = []
        train_acc_list = []

        val_loss_list = []
        val_acc_list = []

        #check time for training NN
        loop_start = timer()

        self.model.to(self.device)

        best_acc = 0.
        best_model_name = ""
        
        early_counter = 0
        interrupt_loop = False

        curr_acc = 0.
        epoch = 0

        metrics_fname = f"{self.name_train}-LR_{initial_lr}-{optimizer.__class__.__name__}_metrics.json"
        metrics_dict = {
            'train_loss': train_loss_list, 'val_loss': val_loss_list, 'train_acc': train_acc_list, 'val_acc': val_acc_list, 'time': 0., 'best_model': best_model_name
            }
        
        while not interrupt_loop and epoch < num_epochs: 

            time_start = timer()

            loss_train, acc_train = self._train(train_loader, criterion, epoch)

            train_loss_list.append(loss_train)
            train_acc_list.append(acc_train)

            curr_acc = acc_train

            if val_loader is not None:

                loss_val, acc_val = self._validation(val_loader, criterion, epoch)
                val_loss_list.append(loss_val)
                val_acc_list.append(acc_val)

                curr_acc = acc_val

            time_end = timer()
           
            if curr_acc > best_acc:
                best_acc = curr_acc
                
                best_model_name =  f"LR_{self.lr}-Epoch_{epoch}"\
                                   f"-Val_{best_acc:.2f}.tar" if val_loader else f"-Train_{best_acc:.2f}.tar"
                self._save_weights(best_model_name)

                metrics_dict['best_model'] = best_model_name

                early_counter = 0

            elif early_stopping is not None:

                if early_counter >= early_stopping:
                    interrupt_loop = True
                else:
                    early_counter +=1

            if self.lr_scheduler and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):

                self.lr = self.optimizer.param_groups[0]['lr']

                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(loss_val if val_loader else loss_train)
                else:
                    self.lr_scheduler.step()

            if verbose:
                self._log_printing(
                    epoch, loss_train, acc_train, time_end - time_start, loss_val=loss_val, acc_val=acc_val
                    )

            #self._writer_update(epoch, loss_train, loss_val)

            if save_metrics:
                self._save_metrics(metrics_dict, os.path.join(self.path_dir, "metrics", self.name_train), metrics_fname)

            epoch += 1

        self.writer.close()

        loop_end = timer()
        time_loop = loop_end - loop_start

        metrics_dict['time'] = time_loop
        if save_metrics:
            self._save_metrics(metrics_dict, os.path.join(self.path_dir, "metrics", self.name_train), metrics_fname)

        if verbose:
            print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

        best_epoch = np.argmax(val_acc_list if val_loader else train_acc_list).astype(int) + 1  #choose best epochs
        best_acc = val_acc_list[best_epoch - 1] if val_loader else train_acc_list[best_epoch - 1]  #choose best loss

        print(f'Best acc: {best_acc:.2f} epoch: {best_epoch}.'+'\n')

        return metrics_dict

    def _log_printing(self, epoch, loss_train, acc_train, time, loss_val=None, acc_val=None):

            log =   f' Epoch: {epoch} '\
                    f' Lr: {self.lr:.7f} '\
                    f' Loss: Train = [{loss_train:.4f}]'

            if loss_val:
                    log += f' - Val = [{loss_val:.4f}] '
            
            log += f' Accuracy: Train = [{acc_train:.2f}%]'

            if acc_val:
                    log += f' - Val = [{acc_val:.2f}%] '
            
            log += f' Time one epoch (s): {(time):.4f} '

            print(log)

    def _save_metrics(self, metric, dirpath, fname):

        json_obj = json.dumps(metric,indent=4)

        os.makedirs(dirpath, exist_ok=True)

        with open(os.path.join(dirpath,fname),"w") as f:
            f.write(json_obj)

    def _writer_update(self, epoch, loss_train, loss_val):

        # Plot to tensorboard

        self.writer.add_scalar('Hyperparameters/Learning Rate', self.lr, epoch) #scheduling of learning rate useful if change
        self.writer.add_scalars('Metrics/Losses', {"Train": loss_train, "Val": loss_val}, epoch)

        self.writer.flush()  #write logs on log folder/file