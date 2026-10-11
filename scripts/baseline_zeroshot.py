"""Zero-shot CLIP / SigLIP 2 baselines (P055).

    python scripts/baseline_zeroshot.py [--backbones clip_vitb16 siglip2_b16]

Image embeddings come from the P051 caches (the backbone's own pooled embedding, L2-normalised); text embeddings
from the backbone's text tower with the prompts of configs/prompts.yaml (fixed before any test evaluation), two
prompt sets per class: plain class names and WMO-style definitions. Scored on the grouped val split of CCSN, MGCD,
SWIMCAT and Montenegro (macro-F1, accuracy); test sets untouched. One run per backbone; table in
docs/data/baseline_zeroshot.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import tasks as T  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.backbones import BACKBONES, FrozenBackbone  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/baseline_zeroshot.md")
OUT = Path("data/baseline_zeroshot.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["clip_vitb16", "siglip2_b16"])
    ap.add_argument("--phase", default="P055")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cache_root = Path(load_paths()["cache_root"])
    prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text(encoding="utf-8"))
    template = prompts["template"]
    tables = T.load_tables()
    results = []
    for key in a.backbones:
        spec = BACKBONES[key]
        cache = FeatureCache(cache_root, key, spec.fixed_size or 224)
        if not cache.complete:
            print(f"{cache.key}: not built, skipped")
            continue
        backbone = FrozenBackbone(key)
        cfg = OmegaConf.create({"backbone": key, "hf_id": spec.hf_id, "prompts": "configs/prompts.yaml", "template": template,
                                "cache_fingerprint": cache.meta()["fingerprint"]})
        with Run(cfg, name=f"zeroshot_{key}", phase=a.phase, data_hash=cache.meta()["fingerprint"]) as run:
            summary = {}
            for source in T.RECOGNITION_SOURCES:
                tab = T.recognition_table(tables, source)
                va = tab[tab.split == "val"]
                classes = sorted(prompts["sources"][source])
                missing = set(va.label) - set(classes)
                if missing:
                    raise ValueError(f"{source}: labels without prompts: {sorted(missing)}")
                x = torch.from_numpy(cache.features(va.sample_id, "cls")).to(backbone.device)
                x = torch.nn.functional.normalize(x, dim=-1)
                for prompt_set in ("name", "definition"):
                    texts = [template.format(prompts["sources"][source][c][prompt_set]) for c in classes]
                    t = backbone.text_features(texts).to(x.dtype)
                    logits = backbone.logit_scale * x @ t.T + backbone.logit_bias
                    pred = np.array(classes)[logits.argmax(1).cpu().numpy()]
                    truth = va.label.to_numpy()
                    f1, acc = T.macro_f1(truth, pred, classes), T.accuracy(truth, pred)
                    results.append({"backbone": key, "source": source, "prompt_set": prompt_set, "n_val": len(va),
                                    "classes": len(classes), "macro_f1": f1, "acc": acc, "run_id": run.run_id})
                    summary[f"{source}/{prompt_set}/macro_f1"] = round(f1, 4)
                    print(f"{key} {source} {prompt_set}: macro-F1 {f1:.3f} acc {acc:.3f}", flush=True)
            run.finish(**summary)
        del backbone
        torch.cuda.empty_cache()
    df = pd.DataFrame(results)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        key_cols = ["backbone", "source", "prompt_set"]
        df = pd.concat([old[~old.set_index(key_cols).index.isin(df.set_index(key_cols).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df, template)
    print(f"done in {time.perf_counter() - t0:.0f} s: {len(results)} rows -> {DOC}")
    return 0


def write_doc(df: pd.DataFrame, template: str) -> None:
    lines = ["# Zero-shot CLIP / SigLIP 2 (P055)", "",
             f"Prompts from `configs/prompts.yaml` (template `{template}`), fixed before any test evaluation; image embeddings "
             "from the P051 caches; scored on the grouped val splits. Linear probes on the same embeddings are in "
             "`docs/data/baseline_probes.md` (phase P055 rows).", "",
             "| Backbone | Prompt set | " + " | ".join(f"{s} F1 / acc" for s in T.RECOGNITION_SOURCES) + " | Mean F1 |",
             "|---|---|" + "---|" * (len(T.RECOGNITION_SOURCES) + 1)]
    for (bb, ps), g in df.groupby(["backbone", "prompt_set"]):
        g = g.set_index("source")
        cells = " | ".join(f"{g.loc[s, 'macro_f1']:.3f} / {g.loc[s, 'acc']:.3f}" if s in g.index else "-"
                            for s in T.RECOGNITION_SOURCES)
        lines.append(f"| {bb} | {ps} | {cells} | {g.macro_f1.mean():.3f} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
