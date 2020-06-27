import torch
from torch import nn
from datetime import datetime
import time
import os 
import glob
from efficientnet_pytorch import EfficientNet
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from apex import amp
import numpy as np


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
        self.model = self.model.cuda()
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        if global_config.FP16:
            # from apex.optimizers import FusedAdam
            # self.optimizer = FusedAdam(optimizer_grouped_parameters, lr=config.lr)

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
            # self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

            num_train_steps = int(self.steps * (global_config.n_epochs) )
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(num_train_steps * 0.05), # WARMUP_PROPORTION = 0.1 as default
                num_training_steps=num_train_steps,
                num_cycles=0.5
            )

            # APEX initialize -> FP16 training (half-precision)
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", verbosity=1)

        else: 
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
            self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = LabelSmoothing().to(self.device)
        self.log(f'>>> Model is loaded. Device is {self.device}')


    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)

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
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                try: 
                    final_scores.update(targets, outputs)
                except:
                    print("outputs: ", list(outputs.data.cpu().numpy())[:10])
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):

            t0 = time.time()
          
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            outputs = self.model(images)


            if global_config.ACCUMULATION_STEP > 1:
                loss = self.criterion(outputs, targets)
                # loss = loss / global_config.ACCUMULATION_STEP  # Normalize loss (if averaged)

                # APEX clip grad  # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                if global_config.FP16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()                # in apex, loss.backward() becomes
                else:
                    loss.backward()                         # compute and sum gradients on params

                if (step + 1) % global_config.ACCUMULATION_STEP == 0:
                    print(f"Step: {step} accum_optimizing")
                    # clip grad btw backward() and step() # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                    # if config.FP16:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=config.CLIP_GRAD_NORM)
                    # else:
                    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_GRAD_NORM) 
                    self.optimizer.step()       # backprop according to accumulated losses
                    self.optimizer.zero_grad()  # clear gradients
                    if self.config.step_scheduler:
                        self.scheduler.step()       # scheduler.step() after opt.step() -> update LR 

            else: 
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets)

                # APEX clip grad  # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                if global_config.FP16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()                # in apex, loss.backward() becomes
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_norm=global_config.CLIP_GRAD_NORM)
                else:
                    loss.backward()                         # compute and sum gradients on params
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=global_config.CLIP_GRAD_NORM) 

                self.optimizer.step()
                if self.config.step_scheduler:
                    self.scheduler.step()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            if self.config.verbose:
                if step % self.config.verbose_step == 0:

                    t1 = time.time()
                    cur_lr = np.format_float_scientific(self.scheduler.get_last_lr()[0], unique=False, precision=1)
                    opt_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
                    print(f":::({str(step).rjust(4, ' ')}/{len(train_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.5f} | LR: {cur_lr}/{opt_lr} | BTime: {t1-t0 :.2f}s | ETime: {int((t1-t0)*(len(train_loader)-step)//60)}m", end='\r')

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        torch.save({
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
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
