from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import numpy as np

from trainer.base_trainer import TrainModel
from tqdm import tqdm

from utils.evaluation_utils import MulticlassAccuracy
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score

from utils.trainer_utils import WarmupCosineLR

class TrainProfileESM(TrainModel):

    def __init__(self, nametrain, device, path_dir="."):
        super().__init__(nametrain, device, path_dir)

    def _train(self, train_loader, criterion, epoch):

        num_batches = len(train_loader)

        num_samples, loss_train, correct_pred = 0,0,0
        
        self.model.train()

        compute_accuracy = MulticlassAccuracy(range(self.model.num_labels))

        class_weights = criterion.weight.cpu().numpy()
        class2weight = {i:class_weights[i] for i in range(len(class_weights))}

        accuracy = 0.
        with tqdm(range(num_batches)) as pbar:
            for idx_batch, (data_map, targets) in zip(pbar, train_loader):
                
                self.optimizer.zero_grad(set_to_none=True)

                inputs = {k: v.to(self.device) for k, v in data_map.items()}
         
                class_targets = targets.to(self.device)

                outputs = self.model(**inputs)
                class_logits = outputs.logits

                loss = criterion(class_logits.view(-1, self.model.num_labels), class_targets.view(-1))

                loss_train += loss.cpu().item() * len(targets)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    #compute accuracy
                    
                    probs = F.softmax(class_logits, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).cpu().numpy()
                    # correct_pred += (pred_id == class_targets).sum().cpu().item()
                    class_targets = class_targets.cpu().numpy()
                    num_samples += len(targets)
                    compute_accuracy(class_targets, pred_id)#, compute_sample_weight(class_weight=class2weight, y=class_targets))

                    if self.log_interval > 0 and idx_batch % self.log_interval == 0:

                        accuracy = compute_accuracy.compute()
                        # accuracy = balanced_accuracy_score(class_targets, pred_id, )
                        self._print_log(
                            idx_batch, pbar, loss_train, num_samples, epoch, num_batches, accuracy, 'train', self.lr
                            )
                    
                if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):

                    self.lr = self.optimizer.param_groups[0]['lr']
                    self.lr_scheduler.step()

        loss_train /= num_samples
        #acc_train = 100. * correct_pred / num_samples
        return loss_train, accuracy

    def _validation(self, val_loader, criterion, epoch):

        num_batches = len(val_loader)

        num_samples, loss_val, correct_pred = 0,0,0

        compute_accuracy = MulticlassAccuracy(range(self.model.num_labels))

        accuracy = 0.

        with torch.no_grad():
            self.model.eval()

            with tqdm(range(num_batches)) as pbar:
                for idx_batch, (data_map, targets) in zip(pbar, val_loader):

                    inputs = {k: v.to(self.device) for k, v in data_map.items()}
                    
                    class_targets = targets.to(self.device)
                    
                    outputs = self.model(**inputs)

                    class_logits = outputs.logits
                    loss = criterion(class_logits.view(-1, self.model.num_labels), class_targets.view(-1))
                    
                    loss_val += loss.cpu().item() * len(targets)

                    #compute accuracy
                    probs = F.softmax(class_logits, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).cpu().numpy()

                    class_targets = class_targets.cpu().numpy()

                    # correct_pred += (pred_id == class_targets).sum().item()

                    num_samples += len(targets)

                    compute_accuracy(class_targets, pred_id)#, compute_sample_weight(class_weight=class2weight, y=class_targets))

                    if self.log_interval > 0 and idx_batch % self.log_interval == 0:
                        
                        accuracy = compute_accuracy.compute()

                        self._print_log(
                            idx_batch, pbar, loss_val, num_samples, epoch, num_batches, accuracy,'val', self.lr
                            )

            loss_val /= num_samples

        return loss_val, accuracy
    
    def _print_log(self, idx_batch, pbar: tqdm, loss, num_samples, epoch, num_batches, acc_interm, type: str, lr=None):

        loss_interm = loss / num_samples
        global_step = idx_batch + (epoch * num_batches)
        #self.writer.add_scalar(f'Metrics/Loss_{type}', running_loss, global_step)


        # acc_interm = 100. * correct_pred / num_samples
        # acc1 = 100. * acc_interm[0]
        # acc2 = 100. * acc_interm[1]

        pbar.set_postfix({'lr':lr if lr else '',
                        f'{type}_loss': np.round(loss_interm, 5),
                        f'{type}_acc': np.round(acc_interm, 2),
                        #'GPU mem': torch.mps.driver_allocated_memory() / 1024 ** 3
                        })
        
        pbar.update(0)