# Augmentation policy (P048)

`configs/augment.yaml` version 1, strength 1.0, input 512 px. Geometric transforms go through `stratia.geometry.transforms` (ADR-004), so ray maps and dense labels follow; photometric ones live in `stratia.augment.photometric`. There is no hue shift anywhere.

## Geometric, by camera-type group

| Group | Camera types | Crop (area share, aspect) | Rotation | Flip |
|---|---|---|---|---|
| all_sky | fisheye_asi | on: [0.8, 1.0], [0.95, 1.05] | on: [-180, 180] deg | off |
| consumer | wide_angle_usb, consumer_photo, fixed_lowcost | on: [0.6, 1.0], [0.8, 1.25] | on: [-5, 5] deg | p = 0.5 |
| patch | wsi_crop, wsi_patch | on: [0.7, 1.0], [0.8, 1.25] | on: [-10, 10] deg | p = 0.5 |

## Photometric (applied in this order)

| Augmentation | p | Range / parameters | Physical meaning |
|---|---|---|---|
| exposure | 0.8 | {'low': 0.6, 'high': 1.5} | exposure-time factor on linear intensity; highlights clip |
| white_balance | 0.6 | {'max_gain': 0.12} | red/blue gains along the colour-temperature axis |
| sun_glare | 0.3 | {'strength': 1.0} | bloom at the Sun's pixel (geometry-aware) + ghost through the centre |
| dirt_and_drops | 0.25 | {} | dark dirt blobs and lens-like water drops on the dome |
| obstruction | 0.2 | {} | dark silhouette; label -> invalid, ray-map patches -> invalid |
| sensor_noise | 0.5 | {'read_sigma': 0.01, 'shot_scale': 0.03} | read + shot noise in linear space |
| jpeg | 0.5 | {'q_low': 40, 'q_high': 95} | recompression quality |

## Gallery

`docs/data/figures/augmentation_gallery.jpg`: one row per dataset group (Eye2Sky all-sky with mask, Sun and the ray-map valid flag outlined in cyan; CCSN photograph; Montenegro fixed camera; SWIMSEG crop with its dense label: cloud outlined in lime, invalid in red). Columns: original, each photometric augmentation alone, three draws of the full policy.

## Review

- Reviewed by the agent on generation: cloud texture survives every augmentation at strength 1; the Sun bloom sits on the Sun; obstructions remove label and ray-map validity where they are pasted. Owner approval of the gallery is pending (exit criterion of P048).
- Ablation hooks: `strength` (P080) and per-augmentation `enabled` (P091).
