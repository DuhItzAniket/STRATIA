"""P068: the OOD score separates shifted features from the training distribution; the flags read the image."""

import numpy as np
import torch

from stratia.model.ood import OodScore, auroc, energy, glare_flag, night_flag


def test_ood_score_separates_shifted_features_and_auroc_is_a_rank_statistic():
    g = torch.Generator().manual_seed(0)
    d = 16
    train = torch.randn(500, d, generator=g)
    logits_in = torch.randn(500, 12, generator=g) * 3
    ood = OodScore(d)
    assert not bool(ood.fitted)
    ood.fit(train, logits_in)
    assert bool(ood.fitted)
    test_in = torch.randn(200, d, generator=g)
    test_out = torch.randn(200, d, generator=g) * 2 + 3           # shifted and wider
    s_in = ood(test_in, torch.randn(200, 12, generator=g) * 3)[:, 0].numpy()
    s_out = ood(test_out, torch.randn(200, 12, generator=g) * 0.3)[:, 0].numpy()   # flat logits: high energy
    assert s_in.shape == (200,)
    assert auroc(s_in, s_out) > 0.95
    assert abs(auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) - 1.0) < 1e-9
    assert abs(auroc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) - 0.5) < 1e-9
    assert energy(torch.tensor([[10.0, 0.0]]))[0] < energy(torch.tensor([[0.0, 0.0]]))[0]
    # buffers travel with the state dict
    state = ood.state_dict()
    ood2 = OodScore(d)
    ood2.load_state_dict(state)
    assert torch.allclose(ood2(test_in[:3], logits_in[:3]), ood(test_in[:3], logits_in[:3]))


def test_night_and_glare_flags():
    dark = np.full((64, 64, 3), 10, np.uint8)
    bright = np.full((64, 64, 3), 120, np.uint8)
    assert night_flag(dark) and not night_flag(bright)
    # ray map: 4 x 4 grid, the patch (0, 0) points at the Sun (zenith 45 deg), the rest straight up
    rm = np.zeros((4, 4, 4), np.float32)
    rm[2] = 1.0
    rm[3] = 1.0
    rm[:3, 0, 0] = [np.sin(np.radians(45)), 0.0, np.cos(np.radians(45))]
    img = bright.copy()
    assert not glare_flag(img, rm, 45.0)
    img[:16, :16] = 255                                            # saturate the Sun's patch
    assert glare_flag(img, rm, 45.0)
    assert not glare_flag(img, np.zeros_like(rm), 45.0)            # unknown geometry: no flag
    assert not glare_flag(img, rm, None)
