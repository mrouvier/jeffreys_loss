import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class JeffreysLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', coeff1=0.0, coeff2=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, coeff1=0.0, coeff2=0.0):
        assert 0 <= coeff1 < 1
        assert 0 <= coeff2 < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(coeff1 / (n_classes - 1)).scatter_(1, targets.data.unsqueeze(1), 1. - coeff1-coeff2)
        return targets

    @staticmethod
    def _jeffreys_one_cold(targets: torch.Tensor, n_classes: int):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(1).scatter_(1, targets.data.unsqueeze(1),0.0)
        return targets

    @staticmethod
    def _jeffreys_one_hot(targets: torch.Tensor, n_classes: int):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device).fill_(0).scatter_(1, targets.data.unsqueeze(1),1.0)
        return targets

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        targets1 = JeffreysLoss._smooth_one_hot(targets, inputs.size(-1), self.coeff1,self.coeff2)
        sm = F.softmax(inputs, -1)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        # (cross-entropy)  & (Jeffreys part 1=label-smoothing)
        loss = -(targets1 * lsm).sum(-1)

        # Jeffreys part 2
        lsmsm = lsm * sm
        targets21 = JeffreysLoss._jeffreys_one_cold(targets, inputs.size(-1),)
        loss1 = (targets21 * lsmsm).sum(-1)

        targets22 = JeffreysLoss._jeffreys_one_hot(targets, inputs.size(-1),)
        loss2 = (targets22 * sm).sum(-1)

        loss3 = loss1/(torch.ones_like(loss2)-loss2)

        loss3 *= self.coeff2
        loss = loss + loss3
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss




criterion = JeffreysLoss(coeff1=0.1, coeff2=0.025)

l1 = torch.FloatTensor([[0,0,1],[1,0,0],[0,0,1]])
l2 = torch.LongTensor([2,2,2])

r = criterion.forward(l1,l2)
print(r)

