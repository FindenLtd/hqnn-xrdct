import torch
import torch.nn as nn

from hybrid_qnn_xrdct.model import DualHeadQNN


class DummyQNN(nn.Module):
    def forward(self, values):
        return values[:, :18]


def test_dual_head_shapes_and_softmax():
    model = DualHeadQNN(DummyQNN(), 18, phases=4)
    patterns = torch.rand(5, 64, dtype=torch.float64)
    maximum = torch.rand(5, 1, dtype=torch.float64)
    wt, sf = model(patterns, maximum)
    assert wt.shape == sf.shape == (5, 4)
    assert torch.allclose(wt.sum(dim=1), torch.ones(5, dtype=torch.float64))

