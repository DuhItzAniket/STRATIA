"""P073: native-class scoring from genus probabilities, batch collation and the per-class known-negative scheme."""

import numpy as np
import torch

from stratia.contract import GENUS_CLASSES
from stratia.model import losses as L
from stratia.train import data as D


def test_native_class_sets_and_scores():
    sets = D.native_class_sets("mgcd", ["cumulus", "altocumulus", "cirrus", "clearsky", "mixed"])
    assert sets["cumulus"] == ([GENUS_CLASSES.index("Cu")], False)
    assert sets["altocumulus"][1] is True
    assert set(sets["altocumulus"][0]) == {GENUS_CLASSES.index("Ac"), GENUS_CLASSES.index("Cc")}
    assert sets["clearsky"] == ([GENUS_CLASSES.index("clear")], False)
    assert sets["mixed"] == ([], False)                                   # unnamed genera: unscorable
    probs = np.zeros((2, len(GENUS_CLASSES)))
    probs[0, GENUS_CLASSES.index("Cc")] = 0.9                             # a cirrocumulus picture
    probs[1, GENUS_CLASSES.index("clear")] = 0.8
    scores, names = D.native_scores(probs, sets)
    assert "mixed" not in names and names[np.argmax(scores[0])] == "altocumulus" and names[np.argmax(scores[1])] == "clearsky"
    sw = D.native_class_sets("swim", ["A-sky", "B-pattern"])
    assert sw["A-sky"][0] == [GENUS_CLASSES.index("clear")] and sw["B-pattern"] == ([], False)


def test_collate_and_per_class_negatives():
    g = len(GENUS_CLASSES)
    items = [{"tokens": torch.zeros(4, 3), "cls": torch.zeros(3), "mask_genus": True, "index": 0, "genus_neg": torch.ones(g)},
             {"tokens": torch.ones(4, 3), "cls": torch.ones(3), "mask_genus": False, "index": 1, "genus_neg": torch.zeros(g)}]
    b = D.collate(items)
    assert b["tokens"].shape == (2, 4, 3) and b["mask_genus"].dtype == torch.bool and b["index"].tolist() == [0, 1]
    # a 'cloud of unnamed genus' row: no positives, negative known on 'clear' only
    logits = torch.zeros(1, g, requires_grad=True)
    target = torch.zeros(1, g)
    neg = torch.zeros(1, g)
    neg[0, GENUS_CLASSES.index("clear")] = 1.0
    L.set_valued_bce(logits, target, torch.tensor([False]), torch.tensor([True]), neg).backward()
    grad = logits.grad[0]
    assert grad[GENUS_CLASSES.index("clear")] > 0 and torch.all(grad[[i for i in range(g) if GENUS_CLASSES[i] != "clear"]] == 0)
    # soft BCE with column weights: unknown columns carry no gradient
    el = torch.zeros(1, 3, requires_grad=True)
    L.soft_bce(el, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([True]), torch.tensor([[1.0, 0.0, 0.0]])).backward()
    assert el.grad[0, 0] < 0 and el.grad[0, 1] == 0 and el.grad[0, 2] == 0
