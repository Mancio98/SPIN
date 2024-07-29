import torch
import torch.nn.functional as F
import numpy as np

from trainer.base_trainer import TrainModel
from tqdm import tqdm


class TrainProfileESM(TrainModel):

    def __init__(self, nametrain, device, path_dir="."):
        super().__init__(nametrain, device, path_dir)

    def _train(self, train_loader, criterion, epoch):

        num_batches = len(train_loader)

        num_samples, loss_train, correct_pred = 0,0,0
        
        self.model.train()

        with tqdm(range(num_batches)) as pbar:
            for idx_batch, (data_map, targets) in zip(pbar, train_loader):
                
                self.optimizer.zero_grad(set_to_none=True)

                inputs = {k: v.to(self.device) for k, v in data_map.items()}
         
                class_targets = targets.to(self.device)

                outputs = self.model(**inputs)
                class_logits = outputs.logits

                loss = criterion(class_logits.view(-1, self.model.num_class), class_targets.view(-1))

                loss_train += loss.cpu().item() * len(targets)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    #compute accuracy
                    
                    probs = F.softmax(class_logits, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1)
                    target_id = class_targets.detach()
                    correct_pred += (pred_id == target_id).sum().cpu().item()

                    num_samples += len(targets)

                    self._print_log(idx_batch, pbar, loss_train, num_samples,
                                    epoch, num_batches, correct_pred,
                                      'train', self.lr)

        loss_train /= num_samples
        acc_train = 100. * correct_pred / num_samples

        return loss_train, acc_train

    def _validation(self, val_loader, criterion, epoch):

        num_batches = len(val_loader)

        num_samples, loss_val, correct_pred = 0,0,0

        with torch.no_grad():
            self.model.eval()

            with tqdm(range(num_batches)) as pbar:
                for idx_batch, (data_map, targets) in zip(pbar, val_loader):

                    inputs = {k: v.to(self.device) for k, v in data_map.items()}
                    
                    class_targets = targets.to(self.device)
                    
                    outputs = self.model(**inputs)

                    class_logits = outputs.logits
                    loss = criterion(class_logits.view(-1, self.model.num_class), class_targets.view(-1))
                    
                    loss_val += loss.cpu().item() * len(targets)

                    #compute accuracy
                    probs = F.softmax(class_logits, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1)
                    target_id = class_targets

                    correct_pred += (pred_id == target_id).sum().item()

                    num_samples += len(targets)

                    self._print_log(idx_batch, pbar, loss_val,
                                   num_samples, epoch, num_batches,
                                   correct_pred,'val', self.lr)

            loss_val /= num_samples
            acc_val = 100. * correct_pred / num_samples
        return loss_val, acc_val
    
    def _print_log(self, idx_batch, pbar, loss, num_samples, epoch, num_batches, correct_pred, type: str, lr=None):

        if self.log_interval > 0:
            if idx_batch % self.log_interval == 0:

                running_loss = loss / num_samples
                global_step = idx_batch + (epoch * num_batches)
                self.writer.add_scalar(f'Metrics/Loss_{type}', running_loss, global_step)

                loss_interm = loss / num_samples
                acc_interm = 100. * correct_pred / num_samples

                pbar.set_postfix({'lr':lr if lr else '',
                                f'{type}_loss': np.round(loss_interm, 5),
                                f'{type}_acc': np.round(acc_interm, 2),
                                'MPS mem': torch.mps.driver_allocated_memory() / 1024 ** 3
                                })
                
                pbar.update(0)