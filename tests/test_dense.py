"""P065 / P066: the dense decoder turns a token grid into the contract's H/4 maps and trains on a toy target."""

import torch

from stratia import contract
from stratia.model import losses as L
from stratia.model.dense import DenseHeads, tokens_to_grid

D = 32


def test_dense_heads_match_contract_shapes():
    heads = DenseHeads(d_model=D, width=16, mid=16, out=16)
    tokens = torch.randn(2, 32 * 32, D)
    out = heads(tokens)
    shapes = contract.output_shapes(2, 512, 512)
    assert tuple(out["sky_parse_logits"].shape) == shapes["sky_parse_logits"]
    assert tuple(out["layer_logits"].shape) == shapes["layer_logits"]
    assert out["dense_feat"].shape == (2, 16, 128, 128)
    g = tokens_to_grid(tokens)
    assert g.shape == (2, D, 32, 32) and torch.equal(g[0, :, 1, 2], tokens[0, 1 * 32 + 2])


def test_decoder_learns_a_spatial_pattern_from_tokens():
    torch.manual_seed(0)
    heads = DenseHeads(d_model=D, width=16, mid=16, out=16)
    # tokens carry their own position (a sky / cloud label per patch); the target is that label at pixel level
    side = 8
    label = (torch.arange(side * side) % 3 == 0).long() + 1              # 1 = sky, 2 = cloud
    tokens = torch.zeros(1, side * side, D)
    tokens[0, :, 0] = label.float() - 1.5
    target = label.reshape(side, side).repeat_interleave(4, 0).repeat_interleave(4, 1)[None]   # 32 x 32
    opt = torch.optim.Adam(heads.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = L.dense_ce(heads(tokens)["sky_parse_logits"], target)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.25 * first
    pred = heads(tokens)["sky_parse_logits"].argmax(1)
    assert (pred == target).float().mean() > 0.9
