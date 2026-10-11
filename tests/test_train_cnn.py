"""P052/P053: the CNN trainer runs end to end on a tiny synthetic cache (CPU, a few steps) and reads its curves."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from stratia.baselines.train_cnn import ImageClassDataset, TrainConfig, fit_verdict, train_classifier


def _cache(tmp_path: Path, n_per=6):
    files, labels = [], []
    for k, colour in enumerate(((200, 60, 60), (60, 200, 60), (60, 60, 200))):
        for i in range(n_per):
            rel = f"ds/c{k}_{i}.jpg"
            p = tmp_path / "img768" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            img = np.full((96, 128, 3), colour, dtype=np.uint8)
            img[i * 10:(i + 1) * 10] = 255 - np.array(colour, dtype=np.uint8)
            cv2.imwrite(str(p), img[..., ::-1])
            files.append(rel)
            labels.append(f"class{k}")
    return files, np.array(labels)


def test_dataset_shapes_and_augmentation(tmp_path):
    files, labels = _cache(tmp_path)
    ds = ImageClassDataset(files, np.arange(len(files)) % 3, tmp_path, 64, train=True, seed=1)
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype.is_floating_point is False and y == 0
    ds_val = ImageClassDataset(files, np.zeros(len(files), int), tmp_path, 64, train=False)
    assert ds_val[3][0].shape == (3, 64, 64)


@pytest.mark.slow
def test_train_classifier_cpu_smoke(tmp_path):
    files, labels = _cache(tmp_path)
    cfg = TrainConfig(arch="resnet18", size=64, epochs=2, batch_size=6, lr=1e-3, workers=0, amp=False, pretrained=False,
                      schedule="cosine", seed=0)
    res = train_classifier(files, labels, files, labels, tmp_path, cfg, log=lambda *a: None)
    assert res["epochs_run"] == 2 and len(res["curves"]) == 2
    assert set(res["classes"]) == {"class0", "class1", "class2"}
    assert 0.0 <= res["best"]["macro_f1"] <= 1.0 and res["best"]["epoch"] in (1, 2)
    assert "pred" in res["best"] and len(res["best"]["pred"]) == len(files)


def test_fit_verdict_reads_the_curves():
    curves = [{"train_acc": 0.5, "val_acc": 0.45, "val_macro_f1": 0.4}, {"train_acc": 0.95, "val_acc": 0.6, "val_macro_f1": 0.55},
              {"train_acc": 0.99, "val_acc": 0.58, "val_macro_f1": 0.50}]
    assert fit_verdict(curves, 2).startswith("overfit (")
    assert fit_verdict(curves[:2], 2).startswith("overfit-leaning")
    assert fit_verdict([{"train_acc": 0.4, "val_acc": 0.38, "val_macro_f1": 0.3}], 1).startswith("underfit")
    assert fit_verdict([{"train_acc": 0.8, "val_acc": 0.75, "val_macro_f1": 0.7}], 1).startswith("ok")
    assert fit_verdict([], 1) == "no epochs"
