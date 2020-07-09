
import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import time
import os 
from glob import glob
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from apex import amp
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from loss_fn import LabelSmoothing
from utils import seed_everything, AverageMeter, RocAucMeter
import config as global_config


class Srnet_Base(nn.Module):
	def __init__(self):
		super(Srnet_Base, self).__init__()
		# Layer 1
		self.layer1 = nn.Conv2d(in_channels=1, out_channels=64,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		# Layer 2
		self.layer2 = nn.Conv2d(in_channels=64, out_channels=16,
			kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(16)
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
		self.fc = nn.Linear(512*1*1, 2)

	def forward(self, inputs):
		# Layer 1
		conv = self.layer1(inputs)
		actv = F.relu(self.bn1(conv))
		# Layer 2
		conv = self.layer2(actv)
		actv = F.relu(self.bn2(conv))
		# Layer 3
		conv1 = self.layer31(actv)
		actv1 = F.relu(self.bn31(conv1))
		conv2 = self.layer32(actv1)
		bn = self.bn32(conv2)
		res = torch.add(actv, bn)
		# Layer 4
		conv1 = self.layer41(res)
		actv1 = F.relu(self.bn41(conv1))
		conv2 = self.layer42(actv1)
		bn = self.bn42(conv2)
		res = torch.add(res, bn)
		# Layer 5
		conv1 = self.layer51(res)
		actv1 = F.relu(self.bn51(conv1))
		conv2 = self.layer52(actv1)
		bn = self.bn52(conv2)
		res = torch.add(res, bn)
		# Layer 6
		conv1 = self.layer61(res)
		actv1 = F.relu(self.bn61(conv1))
		conv2 = self.layer62(actv1)
		bn = self.bn62(conv2)
		res = torch.add(res, bn)
		# Layer 7
		conv1 = self.layer71(res)
		actv1 = F.relu(self.bn71(conv1))
		conv2 = self.layer72(actv1)
		bn = self.bn72(conv2)
		res = torch.add(res, bn)
		# Layer 8
		convs = self.layer81(res)
		convs = self.bn81(convs)
		conv1 = self.layer82(res)
		actv1 = F.relu(self.bn82(conv1))
		conv2 = self.layer83(actv1)
		bn = self.bn83(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 9
		convs = self.layer91(res)
		convs = self.bn91(convs)
		conv1 = self.layer92(res)
		actv1 = F.relu(self.bn92(conv1))
		conv2 = self.layer93(actv1)
		bn = self.bn93(conv2)
		pool = self.pool2(bn)
		res = torch.add(convs, pool)
		# Layer 10
		convs = self.layer101(res)
		convs = self.bn101(convs)
		conv1 = self.layer102(res)
		actv1 = F.relu(self.bn102(conv1))
		conv2 = self.layer103(actv1)
		bn = self.bn103(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 11
		convs = self.layer111(res)
		convs = self.bn111(convs)
		conv1 = self.layer112(res)
		actv1 = F.relu(self.bn112(conv1))
		conv2 = self.layer113(actv1)
		bn = self.bn113(conv2)
		pool = self.pool1(bn)
		res = torch.add(convs, pool)
		# Layer 12
		conv1 = self.layer121(res)
		actv1 = F.relu(self.bn121(conv1))
		conv2 = self.layer122(actv1)
		bn = self.bn122(conv2)
		# print("L12:",res.shape)
		avgp = torch.mean(bn, dim=(2,3), keepdim=True)
		# fully connected
		flatten = avgp.view(avgp.size(0),-1)
		# print("flatten:", flatten.shape)
		out = self.fc(flatten)
		# print("FC:",fc.shape)
		# out = F.log_softmax(fc, dim=1)
		return out

class Customized_Srnet(nn.Module):
    def __init__(self, in_channels):
        super(Customized_Srnet, self).__init__()

        # using RGB 3 channels
        self.srnet = Srnet_Base()
        ckpt = torch.load("../pretrained_models/srnet_weights.pt")
        self.srnet.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(">>> Loaded Pre-trained SRNet!")

    def forward(self, x):
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

        self.model = Customized_Srnet(in_channels=3)
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
            # self.optimizer = FusedAdam(optimizer_grouped_parameters, lr=config.GPU_LR)

            # Try use different LR for HEAD and EffNet
            # self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.GPU_LR)
            LR = config.GPU_LR
            self.optimizer = torch.optim.AdamW([
                        {'params': self.model.parameters(),       'lr': LR[0]},
                        # {'params': self.model.fc1.parameters(),       'lr': LR[1]},
                        # {'params': self.model.bn1.parameters(),       'lr': LR[1]},
                        # {'params': self.model.fc2.parameters(),       'lr': LR[1]},
                        # {'params': self.model.bn2.parameters(),       'lr': LR[1]},
                        # {'params': self.model.dense_out.parameters(), 'lr': LR[1]}
                        ])

            ############################################## 
            self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

            # num_train_steps = int(self.steps * (global_config.GPU_EPOCH))
            # self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            #     self.optimizer,
            #     num_warmup_steps=int(num_train_steps * 0.05), # WARMUP_PROPORTION = 0.1 as default
            #     num_training_steps=num_train_steps,
            #     num_cycles=0.5
            # )

            ############################################## 

            # APEX initialize -> FP16 training (half-precision)
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", verbosity=1)

        else: 
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.GPU_LR)
            self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        if global_config.LOSS_FN_LabelSmoothing:
            self.criterion = LabelSmoothing().to(self.device)
        else: 
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.log(f'>>> Model is loaded. Device is {self.device}')


    def fit(self, train_loader, validation_loader):

        # Continue training proc -> Hand-tune LR 
        if global_config.CONTINUE_TRAIN:

            LR = global_config.GPU_LR

            self.optimizer = torch.optim.AdamW([
                        {'params': self.model.parameters(),       'lr': LR[0]},
                        # {'params': self.model.fc1.parameters(),       'lr': LR[1]},
                        # {'params': self.model.bn1.parameters(),       'lr': LR[1]},
                        # {'params': self.model.dense_out.parameters(), 'lr': LR[1]}
                        ])
            ############################################## 
            self.scheduler = global_config.SchedulerClass(self.optimizer, **global_config.scheduler_params)
            # APEX initialize -> FP16 training (half-precision)
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", verbosity=1)

        for e in range(self.config.GPU_EPOCH):

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)
            
            effNet_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
            head_lr   = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1) 
            print("---" * 31)
            self.log(f":::[Train RESULT] | Epoch: {str(self.epoch).rjust(2, ' ')} | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | LR: {effNet_lr}/{head_lr} | Time: {int((time.time() - t)//60)}m")

            self.save(f'{self.base_dir}/last_ckpt.pt')

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)

            self.log(f":::[Valid RESULT] | Epoch: {str(self.epoch).rjust(2, ' ')} | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | LR: {effNet_lr}/{head_lr} | Time: {int((time.time() - t)//60)}m")

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
            if self.config.verbose:
                if step % self.config.verbose_step == 0:

                    print(f"::: Valid Step({step}/{len(val_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.4f} | Time: {int((time.time() - t))}s") # , end='\r')

                    # print(
                    #     f'Val Step {step}/{len(val_loader)}, ' + \
                    #     f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                    #     f'time: {(time.time() - t):.5f}') #, end='\r'
                    # )
            with torch.no_grad():
                targets = targets.to(self.device)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                try: 
                    final_scores.update(targets, outputs)
                except:
                    # print("outputs: ", list(outputs.data.cpu().numpy())[:10])
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
            targets = targets.to(self.device)
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

            try: 
                final_scores.update(targets, outputs)
            except:
                # print("outputs: ", list(outputs.data.cpu().numpy())[:10])
                pass

            summary_loss.update(loss.detach().item(), batch_size)

            if self.config.verbose:
                if step % self.config.verbose_step == 0:

                    t1 = time.time()
                    effNet_lr = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1)
                    head_lr   = np.format_float_scientific(self.optimizer.param_groups[0]['lr'], unique=False, precision=1) 
                    print(f":::({str(step).rjust(4, ' ')}/{len(train_loader)}) | Loss: {summary_loss.avg:.4f} | AUC: {final_scores.avg:.5f} | LR: {effNet_lr}/{head_lr} | BTime: {t1-t0 :.2f}s | ETime: {int((t1-t0)*(len(train_loader)-step)//60)}m") #, end='\r')

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
        self.model.eval()
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
