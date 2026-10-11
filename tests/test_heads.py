"""P063: the global heads produce the contract's shapes; CORAL is rank-consistent and its probabilities sum to 1."""

import torch

from stratia import contract
from stratia.model.heads import AttentionPool, CoralHead, GlobalHeads

D = 32


def test_global_heads_match_contract_shapes():
    heads = GlobalHeads(d_model=D, hidden=48)
    tokens, cls = torch.randn(3, 64, D), torch.randn(3, D)
    out = heads(tokens, cls)
    shapes = contract.output_shapes(3)
    for name in ("genus_logits", "etage_logits", "oktas_probs"):
        assert tuple(out[name].shape) == shapes[name], name
    assert torch.allclose(out["oktas_probs"].sum(1), torch.ones(3), atol=1e-5)
    assert (out["oktas_probs"] >= 0).all()
    assert out["oktas_cum_logits"].shape == (3, contract.N_OKTAS - 1)


def test_coral_is_rank_consistent_for_any_bias_values():
    head = CoralHead(D, n_classes=9)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([-2.0, 3.0, 0.5, 1.0, -1.0, 2.0, 0.0, -0.5]))   # scrambled on purpose
    x = torch.randn(5, D)
    probs, cum = head(x)
    c = torch.sigmoid(cum)
    assert (c[:, 1:] <= c[:, :-1] + 1e-6).all()                 # P(y > k) non-increasing in k
    assert torch.allclose(probs.sum(1), torch.ones(5), atol=1e-5) and (probs >= 0).all()
    # a strong positive projection moves mass to the top classes, a negative one to the bottom
    with torch.no_grad():
        head.weight.weight.fill_(1.0)
    hi, _ = head(torch.full((1, D), 1.0))
    lo, _ = head(torch.full((1, D), -1.0))
    assert hi[0].argmax() > lo[0].argmax()
    assert (hi[0] * torch.arange(9)).sum() > (lo[0] * torch.arange(9)).sum()


def test_attention_pool_weights_sum_to_one_and_follow_the_query():
    pool = AttentionPool(D, n_heads=4)
    tokens = torch.randn(2, 10, D)
    out = pool(tokens)
    assert out.shape == (2, D)
    w = pool.weights(tokens)
    assert w.shape == (2, 10) and torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)
    # padding mask: masked tokens get zero weight
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 5:] = True
    out_masked = pool(tokens, key_padding_mask=mask)
    assert out_masked.shape == (2, D)
    out_half = pool(tokens[:, :5])
    assert torch.allclose(out_masked, out_half, atol=1e-5)


def test_heads_train_one_step_with_finite_gradients():
    heads = GlobalHeads(d_model=D, hidden=48)
    tokens, cls = torch.randn(4, 64, D), torch.randn(4, D)
    out = heads(tokens, cls)
    loss = out["genus_logits"].pow(2).mean() + out["etage_logits"].pow(2).mean() + out["oktas_cum_logits"].pow(2).mean()
    loss.backward()
    for name, p in heads.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
