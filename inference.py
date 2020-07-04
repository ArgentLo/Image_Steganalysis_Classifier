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
from tqdm import tqdm
warnings.filterwarnings("ignore")


from utils import seed_everything, AverageMeter, RocAucMeter
import config as global_config
from efficientnet import EfficientNet_Model


# Inference
def run_inference():
    
    def get_valid_transforms():
        return A.Compose([
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
                # A.Normalize(p=1.0)
            ], p=1.0)
            
    class DatasetSubmissionRetriever(Dataset):

        def __init__(self, image_names, transforms=None):
            super().__init__()
            self.image_names = image_names
            self.transforms = transforms

        def __getitem__(self, index: int):
            image_name = self.image_names[index]
            image = cv2.imread(f'{global_config.DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            if self.transforms:
                sample = {'image': image}
                sample = self.transforms(**sample)
                image = sample['image']

            return image_name, image

        def __len__(self) -> int:
            return self.image_names.shape[0]

    testset = DatasetSubmissionRetriever(
        image_names=np.array([path.split('/')[-1] for path in glob('../dataset/Test/*.jpg')]),
        transforms=get_valid_transforms(),
    )

    test_loader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    device = torch.device('cuda:0')
    net = EfficientNet_Model(device=device, config=global_config, steps=len(test_loader))

    net.load("./checkpoints/last_ckpt.bin")

    result = {'Id': [], 'Label': []}
    for step, (image_names, images) in enumerate(tqdm(test_loader)):        
        y_pred = net.model(images.cuda())
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
        
        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)

    submission = pd.DataFrame(result)
    submission.to_csv('./node_submissions/test_cuda_inference.csv', index=False)
    submission.head()

run_inference()