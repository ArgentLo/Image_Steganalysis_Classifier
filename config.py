import torch
EfficientNet_Level = 'efficientnet-b7'

SAVED_NAME = "PNasnet5Large"

LOSS_FN_LabelSmoothing = False # LabelSmoothing -> onehot; crossEnt: class_label
CLASS_WEIGHTS = [2, 1.0, 1.05, 0.95] # COVER : JMiPOD : JUNIWARD : UERD'


########   GPU Apex Setting   ########

FP16 = True # using APEX fp16
GPU_BATCH_SIZE = 6
GPU_EPOCH      = 40
GPU_LR         = [1e-3, 1.5e-3] # [EffNet, HEAD]

########   XLA TPU Setting   #########

TPU_BATCH_SIZE = 4 * 8  # 8*8: max for b2
TPU_EPOCH      = 40
TPU_LR         = [1e-3, 1e-3] # [EffNet, HEAD] [1e-3, 1.5e-3]
PRECISE_FTUNE  = False

########   XLA TPU Setting   #########

CONTINUE_TRAIN =  False # "./checkpoints/last_ckpt.pt"
verbose = True
verbose_step = 500

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

SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_params = dict(
        mode='min',
        factor=0.865,
        patience=0,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        min_lr=1e-9
    )

step_scheduler = False  # do scheduler.step after optimizer.step
validation_scheduler = True  # do scheduler.step after validation stage loss

DATA_ROOT_PATH = '../dataset'
num_workers = 4
TPU_num_workers = 0  # load data in the main process
CLIP_GRAD_NORM  = 1e-3

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
