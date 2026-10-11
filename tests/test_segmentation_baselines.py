"""P057: U-Net forward pass, IoU scoring, patch-majority token labels and the token probe on synthetic data."""

import numpy as np
import torch

from stratia.baselines import segmentation as S
from stratia.data.segmentation import CLOUD, IGNORE, SKY, SKY_INVALID


def test_small_unet_forward_and_size():
    model = S.SmallUNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert 1.5e6 < n_params < 2.5e6
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, S.N_CLASSES, 64, 64)


def test_confusion_and_scores():
    true = np.array([[SKY, SKY, CLOUD], [CLOUD, IGNORE, SKY_INVALID]], dtype=np.uint8)
    pred = np.array([[SKY, CLOUD, CLOUD], [CLOUD, SKY, SKY_INVALID]], dtype=np.uint8)
    cm = S.confusion(pred, true)
    assert cm.sum() == 5                                       # the IGNORE pixel is excluded
    sc = S.scores_from_confusion(cm)
    assert sc.iou["invalid"] == 1.0 and abs(sc.iou["cloud"] - 2 / 3) < 1e-9 and abs(sc.iou["sky"] - 1 / 2) < 1e-9
    assert abs(sc.miou - (1.0 + 2 / 3 + 1 / 2) / 3) < 1e-9
    assert abs(sc.cloud_fraction - 3 / 5) < 1e-9
    # a class absent from the truth is left out of the mean even when it is predicted
    true2 = np.array([[SKY, SKY, CLOUD, CLOUD]], dtype=np.uint8)
    pred2 = np.array([[SKY, SKY_INVALID, CLOUD, CLOUD]], dtype=np.uint8)
    sc2 = S.scores_from_confusion(S.confusion(pred2, true2))
    assert "invalid" not in sc2.iou and abs(sc2.miou - (1 / 2 + 1.0) / 2) < 1e-9


def test_mask_to_tokens_majority_and_ignore():
    m = np.full((64, 64), IGNORE, dtype=np.uint8)
    m[:32, :32] = SKY
    m[:32, 32:] = CLOUD
    m[32:40, :32] = SKY_INVALID                                 # a quarter of the bottom-left cell labelled
    tok = S.mask_to_tokens(m, 2)
    assert tok.tolist() == [[SKY, CLOUD], [SKY_INVALID, IGNORE]]


def test_token_probe_learns_and_predicts_a_map():
    rng = np.random.default_rng(0)
    dim, n = 16, 3000
    centres = rng.normal(size=(3, dim)) * 3
    labels = rng.integers(0, 3, n)
    tokens = centres[labels] + rng.normal(size=(n, dim)) * 0.5
    model = S.train_token_probe(tokens, labels, dim=dim, epochs=15, batch=512)
    grid = np.stack([centres[[0, 1, 2, 0]] + rng.normal(size=(4, dim)) * 0.1])    # (1, 4 tokens, D) = 2 x 2 grid
    pred = S.probe_predict(model, grid, side=2, out_size=8)
    assert pred.shape == (1, 8, 8)
    assert pred[0, 0, 0] == 0 and pred[0, 0, 7] == 1 and pred[0, 7, 0] == 2 and pred[0, 7, 7] == 0
