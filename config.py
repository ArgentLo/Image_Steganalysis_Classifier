import torch

EfficientNet_Level = 'efficientnet-b2'

SAVED_NAME = "test_2smoothing"

CLIP_GRAD_NORM  = 1




########   GPU Apex Setting   ########
 
FP16 = True # using APEX fp16
GPU_BATCH_SIZE = 26
GPU_EPOCH      = 10
GPU_LR         = 3e-3  # 1e-3


########   XLA TPU Setting   #########

TPU_BATCH_SIZE = 26
TPU_EPOCH      = 10
TPU_LR         = 4e-4

########   XLA TPU Setting   #########

verbose = True
verbose_step = 2
num_workers = 8

# -------------------

# SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
# scheduler_params = dict(
#     max_lr=0.001,
#     epochs=n_epochs,
#     steps_per_epoch=100,  # int(len(train_dataset) / batch_size)
#     pct_start=0.1,
#     anneal_strategy='cos', 
#     final_div_factor=10**5
# )

# SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
# scheduler_params = dict(
#     mode='min',
#     factor=0.5,
#     patience=1,
#     verbose=False, 
#     threshold=0.0001,
#     threshold_mode='abs',
#     cooldown=0, 
#     min_lr=1e-8,
#     eps=1e-08
#     )

step_scheduler = True  # do scheduler.step after optimizer.step
validation_scheduler = False  # do scheduler.step after validation stage loss

DATA_ROOT_PATH = '../dataset'

# Endpoint features from EfficientNet
EfficientNet_OutFeats = {
    'efficientnet-b0': 1280,
    'efficientnet-b1': 1280,
    'efficientnet-b2': 1408,
    'efficientnet-b3': 1536,
    'efficientnet-b4': 1792,
    'efficientnet-b5': 2048,
    'efficientnet-b6': 2304,
    'efficientnet-b7': 2560
}
EfficientNet_OutFeats = EfficientNet_OutFeats[EfficientNet_Level]

ACCUMULATION_STEP = 1  # accum Grad performs WORSE with BN (BERT is fine)