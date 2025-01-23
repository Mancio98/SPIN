from collections import defaultdict
from pathlib import Path

from config import TrainArgs
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from utils.torch_utils import save_model
from utils.evaluation_utils import EvalMetrics, evaluate

from timeit import default_timer as timer
from utils.trainer_utils import get_optimizer, get_scheduler

class TrainDomSpanESM:

    def __init__(self, verbose, device, train_name, odir: Path|str = "results"):
       
       self.verbose = verbose
       self.device = device

       if isinstance(odir, str):
           odir = Path(odir)
           
       self.ckpt_dir = odir / "checkpoints"
       self.train_name = train_name

    def fit(
        self, args: TrainArgs, model, train_loader,  val_loader=None, save=False, verbose=1
        ):

        optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.l_rate, **args.optimizer_params)
        
        scheduler = get_scheduler(
            optimizer, args.lr_scheduler, len(train_loader) * args.epochs, args.factor
            )

        odir = self.ckpt_dir / self.train_name

        return self._train(
            model,
            args.epochs,
            args.criterion,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            class_weights=args.class_weights,
            log_interval = args.log_interval,
            odir=odir,
            save=save,
            verbose=verbose
            )

    def _train(
            self, model, epochs, criterion, optimizer, scheduler, train_loader, val_loader=None, class_weights=None, log_interval=50, odir='', save=False, verbose:int = 1
        ):

        len_train = len(train_loader)
        model.to(self.device)

        def nest_dict():
            return defaultdict(nest_dict)
        history = defaultdict(nest_dict)

        history['train']['loss'] = []
        history['train']['accuracy'] = []
        
        metrics_val = None
        if val_loader:
            history['val']['loss'] = []
            history['val']['accuracy'] = []

            metrics_val = EvalMetrics(len(model.config.labels), class_weights)

        metrics_train = EvalMetrics(len(model.config.labels), class_weights)
        

        loop_start = timer()

        best_f1 = 0.0
        best_model_path = ""
        
        patience_counter = 0
        patience = 1

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        print(f"Start training for {epochs} epochs")
        for epoch in range(epochs):

            num_samples = 0

            loss_train = 0.0
            loss_span_train = 0.0

            if verbose > 1:
                pbar = tqdm(range(len_train))
                iterator = zip(pbar, train_loader)
            else:
                iterator = enumerate(train_loader)

            model.train()
            for i, (inputs, targets) in iterator:

                optimizer.zero_grad()

                seq_input = {k:v.to(self.device) for k,v in inputs['input'].items()}

                start_pos = None
                end_pos = None
                if inputs['start'] is not None and inputs['end'] is not None:
                    start_pos = inputs['start'].to(self.device)
                    end_pos = inputs['end'].to(self.device)

                targets = targets.to(self.device)

                logits, span_loss, _ = model(seq_input, start_pos, end_pos)

                loss_class = criterion(logits.view(-1, len(model.config.labels)), targets.view(-1))

                loss = 0.65 * loss_class + 0.35 * span_loss
                
                loss_train += loss.item() * len(targets)
                loss_span_train += span_loss.item() * len(targets)
                loss.backward()

                optimizer.step()

                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    
                    scheduler.step()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']


                with torch.no_grad():
                    
                    num_samples += len(targets)

                    y_probs = F.softmax(logits, dim=-1)
                    y_pred = torch.argmax(y_probs, dim=-1).cpu().numpy()

                    metrics_train.update(targets.cpu().numpy(), y_pred)

                    if log_interval > 0 and i % log_interval == 0 and verbose > 1:

                        lr = optimizer.param_groups[0]['lr']

                        report = metrics_train.compute()
                        pbar.set_postfix({
                            'TRAIN '
                            'lr': lr,
                            'loss': np.round(loss_train/num_samples, 5),
                            'accuracy': np.round(report['accuracy'],4),
                            'f1': np.round(report['f1_score'],4)
                            })

                        pbar.update(0)

                break

            loss_train /= num_samples

            history['train']['loss'].append(loss_train)
            metrics_train.step()

            loss_val = None
            if val_loader:
                loss_val = evaluate(model, criterion, metrics_val, val_loader, verbose, self.device)

                history['val']['loss'].append(loss_val)
                metrics_val.step()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_val if val_loader else loss_train)
                elif not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
             
            if val_loader:
                f1score_epoch = metrics_val.f1score[epoch]
            else:
                f1score_epoch = metrics_train.f1score[epoch]

            if f1score_epoch > best_f1:

                best_f1 = f1score_epoch
                patience_counter = 0
                
                if save:
                    fname = f"epoch-{epoch}_f1-{best_f1:.2f}.tar"
                    save_model(model, odir/fname)
                    best_model_path = odir/fname
            else:
                patience_counter += 1


            if verbose > 0 or (not verbose and epoch==epochs-1) or patience_counter > patience:
                self._print_log(epoch, lr, loss_train, metrics_train, loss_val, metrics_val)
            
            if patience_counter > patience:
                break

        loop_end = timer()

        tot_time = loop_end - loop_start

        print(f"Total time (s): {tot_time}",'\n')

        history['train']['accuracy'] = metrics_train.accuracy
        history['train']['f1_score'] = metrics_train.f1score

        if val_loader:
            history['val']['accuracy'] = metrics_val.accuracy
            history['val']['f1_score'] = metrics_val.f1score
            
        history['best_model_path'] = best_model_path
        
        return history

    
    def _print_log(self, epoch, lr, loss_train, metrics_train, loss_val=None, metrics_val=None):

        print(
            f'Epoch: {epoch+1}'
            f' Lr: {lr :4f}'
            ' Train/Val:' if loss_val else 'Train:'
            f' Loss = [{loss_train :.4f}]'
            f',{loss_val :.4f}]' if loss_val else ''
            f' Accuracy = [{metrics_train.accuracy[epoch] :.4f}]'
            f',{metrics_val.accuracy[epoch] :.4f}]' if metrics_val else ''
            f' F1 = [{metrics_train.f1score[epoch] :.4f}]'
            f',{metrics_val.f1score[epoch] :.4f}]' if metrics_val else ''
        )