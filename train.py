from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
from apex import amp
import warnings
warnings.filterwarnings("ignore")


from utils import seed_everything, AverageMeter, RocAucMeter
import config as global_config
from efficientnet import EfficientNet_Model


SEED = 42
seed_everything(SEED)



# GroupKFold splitting

dataset = []

for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('../dataset/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('/')[-1],
            'label': label
        })

random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

gkf = GroupKFold(n_splits=5)

dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    # if fold_number < 5:
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number
    # else: 
    #     break


# Simple Augs: Flips

def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)


# Dataset 

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{global_config.DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
            
        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)


train_fold_num = 1 
val_fold_num = 0

train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] != val_fold_num].kind.values,
    image_names=dataset[dataset['fold'] != val_fold_num].image_name.values,
    labels=dataset[dataset['fold'] != val_fold_num].label.values,
    transforms=get_train_transforms(),
)

# train_dataset = DatasetRetriever(
#     kinds=dataset[dataset['fold'] == train_fold_num].kind.values,
#     image_names=dataset[dataset['fold'] == train_fold_num].image_name.values,
#     labels=dataset[dataset['fold'] == train_fold_num].label.values,
#     transforms=get_train_transforms(),
# )


validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] == val_fold_num].kind.values,
    image_names=dataset[dataset['fold'] == val_fold_num].image_name.values,
    labels=dataset[dataset['fold'] == val_fold_num].label.values,
    transforms=get_valid_transforms()
)


# Class Balance "on fly" from @CatalystTeam
from catalyst.data.sampler import BalanceClassSampler

def run_training():
    device = torch.device('cuda:0')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
        batch_size=global_config.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=global_config.num_workers,
    )
    
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=global_config.batch_size,
        num_workers=global_config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
    )

    print(f"\n>>> Total training examples: {len(train_loader) * global_config.batch_size}")
    net = EfficientNet_Model(device=device, config=global_config, steps=len(train_loader))
#     net.load(f'{net.base_dir}/last-checkpoint.bin')
    net.fit(train_loader, val_loader)


# Training 
run_training()



# Inference
# checkpoint = torch.load('../input/alaska2-public-baseline/best-checkpoint-033epoch.bin')
# net.load_state_dict(checkpoint['model_state_dict']);
# net.eval()

# print("Checkpoint Keys: ", checkpoint.keys())


# class DatasetSubmissionRetriever(Dataset):

#     def __init__(self, image_names, transforms=None):
#         super().__init__()
#         self.image_names = image_names
#         self.transforms = transforms

#     def __getitem__(self, index: int):
#         image_name = self.image_names[index]
#         image = cv2.imread(f'{global_config.DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#         image /= 255.0
#         if self.transforms:
#             sample = {'image': image}
#             sample = self.transforms(**sample)
#             image = sample['image']

#         return image_name, image

#     def __len__(self) -> int:
#         return self.image_names.shape[0]



# dataset = DatasetSubmissionRetriever(
#     image_names=np.array([path.split('/')[-1] for path in glob('../dataset/Test/*.jpg')]),
#     transforms=get_valid_transforms(),
# )


# data_loader = DataLoader(
#     dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=2,
#     drop_last=False,
# )


# result = {'Id': [], 'Label': []}
# for step, (image_names, images) in enumerate(data_loader):
#     print(step, end='\r')
    
#     y_pred = net(images.cuda())
#     y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
    
#     result['Id'].extend(image_names)
#     result['Label'].extend(y_pred)

# submission = pd.DataFrame(result)
# submission.to_csv('submission.csv', index=False)
# submission.head()



