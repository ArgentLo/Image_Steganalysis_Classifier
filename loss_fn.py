import torch
from torch import nn
import torch.nn.functional as F

# Label Smoothing
class LabelSmoothing(nn.Module):
    # smoothing{0~1}: 0.05: default 0.1, meaning the target answer is .9 (1 minus .1) and not 1.
    #       -> want less FPrate -> to relax model's confidence with larger SmoothingFactor
    def __init__(self, smoothing=0.02):  
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction="mean")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss