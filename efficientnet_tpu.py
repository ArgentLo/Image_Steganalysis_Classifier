import torch
from torch import nn
import torch.nn.functional as F
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
import torch_xla.utils.serialization as xser

import numpy as np
import warnings
warnings.filterwarnings("ignore")


from loss_fn import LabelSmoothing
from utils import seed_everything, AverageMeter, RocAucMeter
import config as global_config


def GlobalAvgPooling(x):
    return x.mean(axis=-1).mean(axis=-1)

class Customized_ENSModel(nn.Module):
    def __init__(self, EfficientNet_Level):
        super(Customized_ENSModel, self).__init__()

        self.efn = EfficientNet.from_pretrained(EfficientNet_Level)
        self.efn._fc = nn.Linear(in_features=global_config.EfficientNet_OutFeats, 
                                 out_features=4, bias=True)

        # self.avgpool   = GlobalAvgPooling
        # self.fc1       = nn.Linear(global_config.EfficientNet_OutFeats, global_config.EfficientNet_OutFeats//2)
        # self.bn1       = nn.BatchNorm1d(global_config.EfficientNet_OutFeats//2)
        # self.fc2       = nn.Linear(global_config.EfficientNet_OutFeats//2, global_config.EfficientNet_OutFeats//4)
        # self.bn2       = nn.BatchNorm1d(global_config.EfficientNet_OutFeats//4)
        # self.dense_out = nn.Linear(global_config.EfficientNet_OutFeats//4, 4)
        
    def forward(self, x):
        # x = self.efn.extract_features(x)
        # x = F.gelu(self.avgpool(x))
        # x = F.gelu(self.fc1(x))
        # x = self.bn1(x)  # bn after activation fn
        # x = F.gelu(self.fc2(x))
        # x = self.bn2(x)  # bn after activation fn
        # x = self.dense_out(x)
        # return x
        return self.efn(x)


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
        self.model = Customized_ENSModel(global_config.EfficientNet_Level)
        xm.master_print(">>> Model loaded!")
        self.device = device
        # self.model = self.model.to(device)

        if global_config.LOSS_FN_LabelSmoothing:
            self.criterion = LabelSmoothing()
        else: 
            self.criterion = torch.nn.CrossEntropyLoss()
        self.log(f'>>> Model is loaded. Main Device is {self.device}')


    def fit(self, train_loader, validation_loader):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        # Try use different LR for HEAD and EffNet
        # self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.GPU_LR)
        LR = self.config.TPU_LR
        if global_config.CONTINUE_TRAIN: # Continue training proc -> Hand-tune LR 
            LR = self.config.TPU_LR # [9e-4, 1e-3]
        self.optimizer = torch.optim.AdamW([
                    {'params': self.model.efn.parameters(),       'lr': LR[0]},
                    # {'params': self.model.fc1.parameters(),       'lr': LR[1]},
                    # {'params': self.model.bn1.parameters(),       'lr': LR[1]},
                    # {'params': self.model.fc2.parameters(),       'lr': LR[1]},
                    # {'params': self.model.dense_out.parameters(), 'lr': LR[1]}
                    ])

        ############################################## 
        self.scheduler = self.config.SchedulerClass(self.optimizer, **self.config.scheduler_params)

        # num_train_steps = int(self.steps * (global_config.GPU_EPOCH))
        # self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=int(num_train_steps * 0.05), # WARMUP_PROPORTION = 0.1 as default
        #     num_training_steps=num_train_steps,
        #     num_cycles=0.5
        # )

        ############################################## 

        for e in range(self.config.TPU_EPOCH):
            
            ####### Training
            t = time.time()

            xm.master_print("---" * 31)
            train_device_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())
            summary_loss, final_scores = self.train_one_epoch(train_device_loader)


            effNet_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
            head_lr   = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1) 
            self.log(f":::[Train RESULT] | Epoch: {str(self.epoch).rjust(2, ' ')} | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | LR: {effNet_lr}/{head_lr} | Time: {int((time.time() - t)//60)}m")
            
            self.save(f'{self.base_dir}/last_ckpt.bin')

            ####### Validation
            t = time.time()

            # Skip Validation
            # val_device_loader = pl.MpDeviceLoader(validation_loader, xm.xla_device())
            # summary_loss, final_scores = self.validation(val_device_loader)

            self.log(f":::[Valid RESULT] | Epoch: {str(self.epoch).rjust(2, ' ')} | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | LR: {effNet_lr}/{head_lr} | Time: {int((time.time() - t)//60)}m")

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/{global_config.SAVED_NAME}_{str(self.epoch).zfill(3)}ep.bin')

                # keep only the best 3 checkpoints
                # for path in sorted(glob(f'{self.base_dir}/{global_config.SAVED_NAME}_*ep.bin'))[:-3]:
                #     os.remove(path)

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
                if step % (self.config.verbose_step * 20) == 0:
                    xm.master_print(f"::: Valid Step({step}/{len(val_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | Time: {int((time.time() - t))}s")

            with torch.no_grad():
                targets = targets
                batch_size = images.shape[0]
                images = images
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                try: 
                    final_scores.update(targets, outputs)
                except:
                    # xm.master_print("outputs: ", list(outputs.data.cpu().numpy())[:10])
                    pass
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):

        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (images, targets) in enumerate(train_loader):

            t0 = time.time()
            targets = targets
            images = images
            batch_size = images.shape[0]
            outputs = self.model(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()                         # compute and sum gradients on params
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_config.CLIP_GRAD_NORM) 

            xm.optimizer_step(self.optimizer)
            if self.config.step_scheduler:
                self.scheduler.step()

            try: 
                final_scores.update(targets, outputs)
            except:
                # xm.master_print("outputs: ", list(outputs.data.cpu().numpy())[:10])
                pass
            summary_loss.update(loss.detach().item(), batch_size)

            if self.config.verbose:
                if step % self.config.verbose_step == 0:

                    t1 = time.time()
                    effNet_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
                    head_lr   = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
                    xm.master_print(f":::({str(step).rjust(4, ' ')}/{len(train_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.5f} | LR: {effNet_lr}/{head_lr} | BTime: {t1-t0 :.2f}s | ETime: {int((t1-t0)*(len(train_loader)-step)//60)}m")

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        # xser.save({
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'scheduler_state_dict': self.scheduler.state_dict(),
        #     'best_summary_loss': self.best_summary_loss,
        #     'epoch': self.epoch,
        # }, path, master_only=True)
        xm.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # checkpoint = xser.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.model.eval()
        
    def log(self, message):
        if self.config.verbose:
            xm.master_print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
