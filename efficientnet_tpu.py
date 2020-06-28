import torch
from torch import nn
from datetime import datetime
import time
import os 
from glob import glob
from efficientnet_pytorch import EfficientNet
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

# from apex import amp
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


import numpy as np
import warnings
warnings.filterwarnings("ignore")


from loss_fn import LabelSmoothing
from utils import seed_everything, AverageMeter, RocAucMeter
import config as global_config


# EfficientNet
class EfficientNet_Model:
    
    def __init__(self, device, config, steps):
        self.config = config
        self.epoch = 0
        self.steps = steps
        
        self.base_dir = './checkpoints'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        # get pretrained models
        self.model = EfficientNet.from_pretrained(global_config.EfficientNet_Level)
        self.model._fc = nn.Linear(in_features=global_config.EfficientNet_OutFeats, out_features=4, bias=True) 

        xm.master_print(">>> Model loaded!")
        self.device = device
        self.model = self.model.to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        lr = config.lr*xm.xrt_world_size()
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        # self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        num_train_steps = int(self.steps * (global_config.n_epochs))
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_train_steps * 0.05), # WARMUP_PROPORTION = 0.1 as default
            num_training_steps=num_train_steps,
            num_cycles=0.5
        )

        self.criterion = LabelSmoothing()
        self.log(f'>>> Model is loaded. Device is {self.device}')


    def fit(self, train_loader, validation_loader):

        for e in range(self.config.n_epochs):
            
            ####### Training
            t = time.time()

            train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
            summary_loss, final_scores = self.train_one_epoch(train_device_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            ####### Validation
            t = time.time()

            val_device_loader = pl.MpDeviceLoader(validation_loader, self.device)
            summary_loss, final_scores = self.validation(val_device_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                try:
                    self.scheduler.step(metrics=summary_loss.avg)
                except:
                    self.scheduler.step()
                    
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    xm.master_print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
            with torch.no_grad():
                targets = targets#.to(self.device).float()
                batch_size = images.shape[0]
                images = images#.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                try: 
                    final_scores.update(targets, outputs)
                except:
                    xm.master_print("outputs: ", list(outputs.data.cpu().numpy())[:10])
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):

        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (images, targets) in enumerate(train_loader):

            t0 = time.time()
            targets = targets#.to(self.device).float()
            images = images#.to(self.device).float()
            batch_size = images.shape[0]

            outputs = self.model(images)


            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()                         # compute and sum gradients on params
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_config.CLIP_GRAD_NORM) 

            xm.optimizer_step(self.optimizer)
            if self.config.step_scheduler:
                self.scheduler.step()

            try: 
                final_scores.update(targets, outputs)
            except:
                xm.master_print("outputs: ", list(outputs.data.cpu().numpy())[:10])
            summary_loss.update(loss.detach().item(), batch_size)

            if self.config.verbose:
                if step % self.config.verbose_step == 0:

                    t1 = time.time()
                    cur_lr = np.format_float_scientific(self.scheduler.get_last_lr()[0], unique=False, precision=1)
                    opt_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
                    xm.master_print(f":::({str(step).rjust(4, ' ')}/{len(train_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.5f} | LR: {cur_lr}/{opt_lr} | BTime: {t1-t0 :.2f}s | ETime: {int((t1-t0)*(len(train_loader)-step)//60)}m")

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        xm.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.model.eval()
        
    def log(self, message):
        if self.config.verbose:
            xm.master_print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
