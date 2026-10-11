# Small VLM zero-shot sanity check (P056)

`Qwen/Qwen3-VL-2B-Instruct` in fp16, 448 px, 100 random grouped-val images per source, closed list of class names from `configs/prompts.yaml`, greedy decoding, answer parsed by the first class name it contains (unparsed answers count as wrong). One pass, no prompt tuning.

| Source | n | Macro-F1 | Accuracy | Unparsed |
|---|---|---|---|---|
| ccsn | 100 | 0.225 | 0.250 | 0.01 |
| mgcd | 100 | 0.369 | 0.400 | 0.00 |
