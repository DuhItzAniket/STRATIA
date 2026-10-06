# Near-duplicates report (P025)

DINOv3 vits16 CLS embeddings at 224 px, 10 nearest neighbours per image, pairs kept with cosine >= 0.9: **303,925 candidate pairs** among 52,032 images; pHash/dHash over all flips and rotations.

Rule (chosen from the contact sheets): a pair is a **copy** if its pHash distance over all flips and rotations is <= 2, and shows the **same scene** if its cosine is >= 0.97; either makes it a near-duplicate. Same-station Eye2Sky pairs are excluded (the camera's own time series; handled by temporal blocking, P027/P037).

- Candidate pairs that are same-station Eye2Sky time series: 196,767
- Copies: 16,459 pairs; same scene: 42,025 pairs; near-duplicate pairs in all: **45,999**, of which cross-dataset 2,229
- Exhaustive pHash check of SWINySEG against SWIMSEG, SWINSEG and SHWIMSEG (every pair, no embedding): **2,342 of 6,768 SWINySEG images (34.6%) are flipped, rotated or re-encoded copies of a source-dataset image** (by source: {'swimseg': 2096, 'swinseg': 246}); table `data/swinyseg_sources.parquet`
- Images in a near-duplicate group: **14,112** in **1,656** groups (largest 448)

## Cosine similarity of candidate pairs by dataset pair

| Datasets | Pairs | >= 0.99 | 0.97-0.99 | 0.94-0.97 | 0.90-0.94 | pHash <= 2 |
|---|---|---|---|---|---|---|
| almeria,almeria | 5,782 | 32 | 570 | 3,335 | 1,845 | 59 |
| b0268,b0268 | 3 | 0 | 0 | 0 | 3 | 0 |
| ccsn,ccsn | 539 | 81 | 58 | 53 | 347 | 146 |
| ccsn,swimcat | 3 | 0 | 0 | 0 | 3 | 0 |
| ccsn,swinyseg | 1 | 0 | 0 | 0 | 1 | 0 |
| eye2sky,eye2sky | 197,778 | 22,258 | 118,703 | 55,077 | 1,740 | 36,801 |
| mgcd,mgcd | 50,817 | 8,258 | 27,961 | 13,146 | 1,452 | 9,563 |
| montenegro,montenegro | 13,723 | 14 | 872 | 5,295 | 7,542 | 1,661 |
| shwimseg,shwimseg | 394 | 3 | 19 | 69 | 303 | 8 |
| shwimseg,swimseg | 16 | 0 | 0 | 0 | 16 | 0 |
| shwimseg,swinyseg | 19 | 0 | 0 | 0 | 19 | 0 |
| swimcat,swimcat | 2,552 | 17 | 78 | 601 | 1,856 | 19 |
| swimcat,swinyseg | 12 | 0 | 0 | 0 | 12 | 0 |
| swimseg,swimseg | 2,875 | 26 | 190 | 1,211 | 1,448 | 26 |
| swimseg,swinyseg | 5,393 | 5 | 548 | 2,603 | 2,237 | 1,973 |
| swinseg,swinseg | 77 | 0 | 0 | 6 | 71 | 0 |
| swinseg,swinyseg | 323 | 0 | 19 | 108 | 196 | 209 |
| swinyseg,swinyseg | 23,618 | 541 | 1,974 | 8,111 | 12,992 | 1,946 |

## Near-duplicate groups by dataset

| Dataset | Images in a group | Share of dataset | Groups | Largest group |
|---|---|---|---|---|
| almeria | 356 | 43.5% | 86 | 117 |
| ccsn | 310 | 12.2% | 149 | 4 |
| eye2sky | 456 | 1.6% | 4 | 448 |
| mgcd | 7,775 | 97.2% | 227 | 291 |
| montenegro | 883 | 35.0% | 106 | 248 |
| shwimseg | 42 | 26.9% | 18 | 4 |
| swimcat | 132 | 16.8% | 51 | 13 |
| swimseg | 984 | 97.1% | 719 | 9 |
| swinseg | 109 | 94.8% | 109 | 1 |
| swinyseg | 3,065 | 45.3% | 1,014 | 30 |

## Contact sheets (16 pairs each, left and right of a pair side by side)

- `docs/data/figures/near_dup_cross_dataset.jpg`: cross_dataset (5,767 pairs)
- `docs/data/figures/near_dup_band_0.97_1.00.jpg`: band_0.97_1.00 (41,453 pairs)
- `docs/data/figures/near_dup_band_0.94_0.97.jpg`: band_0.94_0.97 (34,751 pairs)
- `docs/data/figures/near_dup_band_0.90_0.94.jpg`: band_0.90_0.94 (30,382 pairs)
- `docs/data/figures/near_dup_copies.jpg`: copies (2,182 pairs)
- `docs/data/figures/near_dup_phash_3_to_6.jpg`: phash_3_to_6 (4,403 pairs)

## Groups that span datasets

| Group | Datasets | Images |
|---|---|---|
| 11327 | swimseg,swinyseg | 3 |
| 11328 | swimseg,swinyseg | 7 |
| 11329 | swimseg,swinyseg | 3 |
| 11330 | swimseg,swinyseg | 3 |
| 11331 | swimseg,swinyseg | 4 |
| 11332 | swimseg,swinyseg | 6 |
| 11333 | swimseg,swinyseg | 5 |
| 11334 | swimseg,swinyseg | 3 |
| 11336 | swimseg,swinyseg | 3 |
| 11337 | swimseg,swinyseg | 4 |
| 11338 | swimseg,swinyseg | 4 |
| 11339 | swimseg,swinyseg | 2 |
| 11340 | swimseg,swinyseg | 3 |
| 11341 | swimseg,swinyseg | 5 |
| 11343 | swimseg,swinyseg | 3 |
| 11344 | swimseg,swinyseg | 4 |
| 11345 | swimseg,swinyseg | 2 |
| 11346 | swimseg,swinyseg | 7 |
| 11347 | swimseg,swinyseg | 2 |
| 11348 | swimseg,swinyseg | 3 |
| 11350 | swimseg,swinyseg | 4 |
| 11351 | swimseg,swinyseg | 5 |
| 11352 | swimseg,swinyseg | 5 |
| 11353 | swimseg,swinyseg | 4 |
| 11354 | swimseg,swinyseg | 3 |
| 11355 | swimseg,swinyseg | 2 |
| 11356 | swimseg,swinyseg | 2 |
| 11357 | swimseg,swinyseg | 3 |
| 11358 | swimseg,swinyseg | 4 |
| 11359 | swimseg,swinyseg | 2 |
| 11360 | swimseg,swinyseg | 3 |
| 11361 | swimseg,swinyseg | 7 |
| 11362 | swimseg,swinyseg | 2 |
| 11363 | swimseg,swinyseg | 2 |
| 11365 | swimseg,swinyseg | 2 |
| 11366 | swimseg,swinyseg | 3 |
| 11368 | swimseg,swinyseg | 5 |
| 11369 | swimseg,swinyseg | 3 |
| 11371 | swimseg,swinyseg | 4 |
| 11372 | swimseg,swinyseg | 10 |
