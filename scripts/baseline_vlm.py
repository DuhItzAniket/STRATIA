"""Small vision-language model zero-shot sanity check (P056, deferrable).

    python scripts/baseline_vlm.py [--model Qwen/Qwen3-VL-2B-Instruct] [--per-source 100] [--sources ccsn mgcd]

Asks the VLM to name the cloud type of grouped-val images with a closed list of class names (configs/prompts.yaml
`name` entries) and parses the answer by the first class name it contains. fp16 on the GPU (no 4-bit: bitsandbytes
has no Windows wheel in the lock), 448 px input, a random subset per source. This is a sanity check, not a tuned
baseline: no prompt search, one pass, macro-F1 and accuracy on the subset; test sets untouched.
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import tasks as T  # noqa: E402
from stratia.data.image_cache import cache_path  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/baseline_vlm.md")
OUT = Path("data/baseline_vlm.parquet")


def parse_answer(text: str, names: dict[str, str]) -> str | None:
    """First class whose display name (or key) appears in the answer; None when nothing matches."""
    low = text.lower()
    hits = []
    for key, name in names.items():
        for needle in (name.lower(), key.lower()):
            pos = low.find(needle)
            if pos >= 0:
                hits.append((pos, key))
                break
    return min(hits)[1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--sources", nargs="+", default=["ccsn", "mgcd"])
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="P056")
    a = ap.parse_args()
    t0 = time.perf_counter()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cache_root = Path(load_paths()["cache_root"])
    prompts = yaml.safe_load(Path("configs/prompts.yaml").read_text(encoding="utf-8"))
    tables = T.load_tables()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dtype).to(device).eval()
    rows, preds = [], []
    cfg = OmegaConf.create({"model": a.model, "per_source": a.per_source, "size": a.size, "seed": a.seed,
                            "prompts": "configs/prompts.yaml"})
    with Run(cfg, name="vlm_zeroshot", phase=a.phase) as run:
        summary = {}
        for source in a.sources:
            names = {k: v["name"] for k, v in prompts["sources"][source].items()}
            tab = T.recognition_table(tables, source)
            va = tab[tab.split == "val"].sample(min(a.per_source, int((tab.split == "val").sum())), random_state=a.seed)
            options = "; ".join(f"{v}" for v in names.values())
            question = (f"This is a photograph of the sky. Which one of these cloud types does it show? Options: {options}. "
                        "Answer with the option only.")
            truth, pred = [], []
            for _, r in va.iterrows():
                img = Image.open(cache_path(cache_root, r.image_file)).convert("RGB").resize((a.size, a.size))
                messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
                inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                       return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
                answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                p = parse_answer(answer, names)
                truth.append(r.label)
                pred.append(p if p is not None else "__none__")
                preds.append({"source": source, "sample_id": r.sample_id, "label": r.label, "answer": answer, "pred": p})
            truth, pred = np.array(truth), np.array(pred)
            f1, acc = T.macro_f1(truth, pred, sorted(names)), T.accuracy(truth, pred)
            unparsed = float(np.mean(pred == "__none__"))
            rows.append({"model": a.model, "source": source, "n": len(va), "macro_f1": f1, "acc": acc, "unparsed": unparsed,
                         "run_id": run.run_id})
            summary.update({f"{source}/macro_f1": round(f1, 4), f"{source}/acc": round(acc, 4),
                            f"{source}/unparsed": round(unparsed, 4)})
            print(f"{source}: n={len(va)} macro-F1 {f1:.3f} acc {acc:.3f} unparsed {unparsed:.2f}", flush=True)
        run.finish(**summary)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    pd.DataFrame(preds).to_parquet("data/baseline_vlm_answers.parquet", index=False)
    lines = ["# Small VLM zero-shot sanity check (P056)", "",
             f"`{a.model}` in fp16, {a.size} px, {a.per_source} random grouped-val images per source, closed list of class "
             "names from `configs/prompts.yaml`, greedy decoding, answer parsed by the first class name it contains (unparsed "
             "answers count "
             "as wrong). One pass, no prompt tuning.", "",
             "| Source | n | Macro-F1 | Accuracy | Unparsed |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.source} | {int(r.n)} | {r.macro_f1:.3f} | {r.acc:.3f} | {r.unparsed:.2f} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
