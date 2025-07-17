import torch.nn as nn
import torch


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1)


class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss()

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma=1e-7):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return gamma * loss


def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h * w))
    out = torch.norm(input, p='nuc')
    return out


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, input, decay=1e-5):
        input = torch.sum(input, 0, keepdim=True)
        loss = Nuclear_norm(input)
        return decay * loss


