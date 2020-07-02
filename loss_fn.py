import torch
from torch import nn

# Label Smoothing
class LabelSmoothing(nn.Module):
    # smoothing{0~1}: 0.05: default 0.1, meaning the target answer is .9 (1 minus .1) and not 1.
    #       -> want less FPrate -> to relax model's confidence with larger SmoothingFactor
    def __init__(self, smoothing = 0.1):  
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()

            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
            # b = torch.max(x, dim=-1) 
            # logprobs = x - torch.logsumexp(x-b, dim=-1) - b

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)
