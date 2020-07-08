import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import time
import os 
from glob import glob
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

# from apex import amp
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser

import numpy as np
import warnings
import gc
warnings.filterwarnings("ignore")


from loss_fn import LabelSmoothing
from utils import seed_everything, AverageMeter, RocAucMeter
import config as global_config



class Srnet_Base(nn.Module):
	def __init__(self, in_channels):
		super(Srnet_Base, self).__init__()
		# Layer 1
		# self.layer1 = nn.Conv2d(in_channels=1, out_channels=64,
		# 	kernel_size=3, stride=1, padding=1, bias=False)
		self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		# Layer 2
		self.layer2 = nn.Conv2d(in_channels=64, out_channels=32,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)
		# Layer 2_Extent
		self.layer2_ext = nn.Conv2d(in_channels=32, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2_ext = nn.BatchNorm2d(16)
		# Layer 3
		self.layer31 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn31 = nn.BatchNorm2d(16)
		self.layer32 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn32 = nn.BatchNorm2d(16)
		# Layer 4
		self.layer41 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn41 = nn.BatchNorm2d(16)
		self.layer42 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn42 = nn.BatchNorm2d(16)
		# Layer 5
		self.layer51 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn51 = nn.BatchNorm2d(16)
		self.layer52 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn52 = nn.BatchNorm2d(16)
		# Layer 6
		self.layer61 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn61 = nn.BatchNorm2d(16)
		self.layer62 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn62 = nn.BatchNorm2d(16)
		# Layer 7
		self.layer71 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn71 = nn.BatchNorm2d(16)
		self.layer72 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn72 = nn.BatchNorm2d(16)
		# Layer 8
		self.layer81 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn81 = nn.BatchNorm2d(16)
		self.layer82 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn82 = nn.BatchNorm2d(16)
		self.layer83 = nn.Conv2d(in_channels=16, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn83 = nn.BatchNorm2d(16)
		self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		# Layer 9
		self.layer91 = nn.Conv2d(in_channels=16, out_channels=64,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn91 = nn.BatchNorm2d(64)
		self.layer92 = nn.Conv2d(in_channels=16, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn92 = nn.BatchNorm2d(64)
		self.layer93 = nn.Conv2d(in_channels=64, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn93 = nn.BatchNorm2d(64)
		self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		# Layer 10
		self.layer101 = nn.Conv2d(in_channels=64, out_channels=128,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn101 = nn.BatchNorm2d(128)
		self.layer102 = nn.Conv2d(in_channels=64, out_channels=128,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn102 = nn.BatchNorm2d(128)
		self.layer103 = nn.Conv2d(in_channels=128, out_channels=128,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn103 = nn.BatchNorm2d(128)
		self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		# Layer 11
		self.layer111 = nn.Conv2d(in_channels=128, out_channels=256,
			kernel_size=1, stride=2, padding=0, bias=False)
		self.bn111 = nn.BatchNorm2d(256)
		self.layer112 = nn.Conv2d(in_channels=128, out_channels=256,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn112 = nn.BatchNorm2d(256)
		self.layer113 = nn.Conv2d(in_channels=256, out_channels=256,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn113 = nn.BatchNorm2d(256)
		self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

		# Layer 12
		self.layer121 = nn.Conv2d(in_channels=256, out_channels=512,
			kernel_size=3, stride=2, padding=0, bias=False)
		self.bn121 = nn.BatchNorm2d(512)
		self.layer122 = nn.Conv2d(in_channels=512, out_channels=512,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn122 = nn.BatchNorm2d(512)
		# avgp = torch.mean() in forward before fc
		# Fully Connected layer
		self.fc = nn.Linear(512*1*1, 4)

	def forward(self, inputs):
		# Layer 1
		conv = self.layer1(inputs)
		actv = F.gelu(self.bn1(conv))
		# Layer 2
		conv = self.layer2(actv)
		actv = F.gelu(self.bn2(conv))
		conv = self.layer2_ext(actv)
		actv = F.gelu(self.bn2_ext(conv))
		
		# Layer 3
		conv1 = self.layer31(actv)
		actv1 = F.gelu(self.bn31(conv1))
		conv2 = self.layer32(actv1)
		bn = self.bn32(conv2)
		res = torch.add(actv, bn)
		# Layer 4
		conv1 = self.layer41(res)
		actv1 = F.gelu(self.bn41(conv1))
		conv2 = self.layer42(actv1)
		bn = self.bn42(conv2)
		res = torch.add(res, bn)
		# Layer 5
		conv1 = self.layer51(res)
		actv1 = F.gelu(self.bn51(conv1))
		conv2 = self.layer52(actv1)
		bn = self.bn52(conv2)
		res = torch.add(res, bn)
		# Layer 6
		conv1 = self.layer61(res)
		actv1 = F.gelu(self.bn61(conv1))
		conv2 = self.layer62(actv1)
		bn = self.bn62(conv2)
		res = torch.add(res, bn)
		# Layer 7
		conv1 = self.layer71(res)
		actv1 = F.gelu(self.bn71(conv1))
		conv2 = self.layer72(actv1)
		bn = self.bn72(conv2)
		res = torch.add(res, bn)
		# Layer 8
		convs = self.layer81(res)
		convs = self.bn81(convs)
		conv1 = self.layer82(res)
		actv1 = F.gelu(self.bn82(conv1))
		conv2 = self.layer83(actv1)
		bn = self.bn83(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 9
		convs = self.layer91(res)
		convs = self.bn91(convs)
		conv1 = self.layer92(res)
		actv1 = F.gelu(self.bn92(conv1))
		conv2 = self.layer93(actv1)
		bn = self.bn93(conv2)
		pool = self.pool2(bn)
		res = torch.add(convs, pool)
		# Layer 10
		convs = self.layer101(res)
		convs = self.bn101(convs)
		conv1 = self.layer102(res)
		actv1 = F.gelu(self.bn102(conv1))
		conv2 = self.layer103(actv1)
		bn = self.bn103(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 11
		convs = self.layer111(res)
		convs = self.bn111(convs)
		conv1 = self.layer112(res)
		actv1 = F.gelu(self.bn112(conv1))
		conv2 = self.layer113(actv1)
		bn = self.bn113(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 12
		conv1 = self.layer121(res)
		actv1 = F.gelu(self.bn121(conv1))
		conv2 = self.layer122(actv1)
		bn = self.bn122(conv2)
		# print("L12:",res.shape)
		avgp = torch.mean(bn, dim=(2,3), keepdim=True)
		# fully connected
		flatten = avgp.view(avgp.size(0),-1)
		# print("flatten:", flatten.shape)
		fc = self.fc(flatten)
		# print("FC:",fc.shape)
		# out = F.log_softmax(fc, dim=1)
		return fc

class Customized_Srnet(nn.Module):
    def __init__(self, in_channels):
        super(Customized_Srnet, self).__init__()

        # using RGB 3 channels
        self.srnet = Srnet_Base(in_channels=in_channels)

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
        return self.srnet(x)


# Srnet_Model
class Srnet_Model:
    
    def __init__(self, device, config, steps):
        self.config = config
        self.epoch = 0
        self.steps = steps
        
        self.base_dir = './checkpoints'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        # get pretrained models
        self.model = Customized_Srnet(in_channels=3)
        xm.master_print(">>> Model loaded!")
        self.device = device

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
                    {'params': self.model.parameters(),       'lr': LR[0]}
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
        # DataLoader should init only once (outside the epoch loop) 
        train_device_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())
        if validation_loader == 1:
            pass
        else:
            val_device_loader   = pl.MpDeviceLoader(validation_loader, xm.xla_device())
        ############################################## 

        for e in range(self.config.TPU_EPOCH):

            ############## Training
            gc.collect()
            t = time.time()
            xm.master_print("---" * 31)
            summary_loss, final_scores = self.train_one_epoch(train_device_loader)

            effNet_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
            head_lr   = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1) 
            self.log(f":::[Train RESULT]| Epoch: {str(self.epoch).rjust(2, ' ')} | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | LR: {effNet_lr}/{head_lr} | Time: {int((time.time() - t)//60)}m")
            self.save(f'{self.base_dir}/last_ckpt.pt')

            ############## Validation
            gc.collect()
            t = time.time()
            # Skip Validation
            if validation_loader == 1:
                pass
            else:
                summary_loss, final_scores = self.validation(val_device_loader)

            self.log(f":::[Valid RESULT]| Epoch: {str(self.epoch).rjust(2, ' ')} | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | LR: {effNet_lr}/{head_lr} | Time: {int((time.time() - t)//60)}m")

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/{global_config.SAVED_NAME}_{str(self.epoch).zfill(3)}ep.pt')
                # keep only the best 3 checkpoints
                # for path in sorted(glob(f'{self.base_dir}/{global_config.SAVED_NAME}_*ep.pt'))[:-3]:
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

            with torch.no_grad():
                batch_size = images.shape[0]
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                try: 
                    final_scores.update(targets, outputs)
                except:
                    if step % (self.config.verbose_step * 2) == 0:
                        xm.master_print("final_scores update failed...")
                    pass
                summary_loss.update(loss.detach().item(), batch_size)

            if self.config.verbose:
                if step % (self.config.verbose_step * 2) == 0:
                    xm.master_print(f"::: Valid Step({step}/{len(val_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | Time: {int((time.time() - t))}s")

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):

        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (images, targets) in enumerate(train_loader):

            t0 = time.time()
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
        #xser.save(self.model.state_dict(), path, master_only=True, global_master=True )
        xm.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        # checkpoint = xser.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        self.model.eval()

    def log(self, message):
        if self.config.verbose:
            xm.master_print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

