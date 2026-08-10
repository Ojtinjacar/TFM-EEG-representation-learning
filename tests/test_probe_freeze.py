import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from downstream import FullModel, Head
from models import EnhancedAttentionLSTM


def _build(freeze):
    torch.manual_seed(0)
    backbone = EnhancedAttentionLSTM(
        input_size=64, hidden_size=16, n_channels=2, sfreq=250,
        lstm_hidden_size=8,
    )
    model = FullModel(backbone, Head(16, 1), freeze_backbone=freeze)
    if freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = model.head.parameters()
    else:
        params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=0.1)
    return model, optimizer


def _bn_buffers(model):
    return {
        name: (m.running_mean.clone(), m.running_var.clone())
        for name, m in model.backbone.named_modules()
        if isinstance(m, nn.BatchNorm2d)
    }


def _train_steps(model, optimizer, n=2):
    x = torch.randn(8, 2, 64)
    y = torch.randn(8, 1)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(n):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def test_linear_probe_backbone_stays_in_eval_mode():
    model, _ = _build(freeze=True)
    model.train()
    # The backbone's dropout lives INSIDE nn.LSTM / nn.MultiheadAttention
    # (there are no standalone nn.Dropout modules), so the freeze must be
    # asserted on every module's training flag, not on Dropout instances.
    assert all(not m.training for m in model.backbone.modules())
    frozen_bn = [m for m in model.backbone.modules()
                 if isinstance(m, nn.BatchNorm2d)]
    frozen_recurrent = [m for m in model.backbone.modules()
                        if isinstance(m, (nn.LSTM, nn.MultiheadAttention))]
    assert frozen_bn, "expected BatchNorm2d modules in the backbone"
    assert frozen_recurrent, "expected LSTM/attention modules in the backbone"
    assert model.head.training


def test_linear_probe_does_not_touch_bn_buffers_but_trains_head():
    model, optimizer = _build(freeze=True)
    before_bn = _bn_buffers(model)
    before_head = [p.clone() for p in model.head.parameters()]
    _train_steps(model, optimizer)
    after_bn = _bn_buffers(model)
    assert before_bn, "expected BatchNorm2d buffers in the backbone"
    for name in before_bn:
        assert torch.equal(before_bn[name][0], after_bn[name][0]), name
        assert torch.equal(before_bn[name][1], after_bn[name][1]), name
    assert any(
        not torch.equal(b, p)
        for b, p in zip(before_head, model.head.parameters())
    )


def test_fine_tuning_still_updates_bn_buffers():
    model, optimizer = _build(freeze=False)
    before_bn = _bn_buffers(model)
    _train_steps(model, optimizer)
    after_bn = _bn_buffers(model)
    changed = any(
        not torch.equal(before_bn[name][0], after_bn[name][0])
        for name in before_bn
    )
    assert changed, "fine-tuning must keep updating BatchNorm running stats"
