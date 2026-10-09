"""Sun-disc detection in all-sky frames (P044).

At the Eye2Sky day exposure (0.15 ms) the Sun is the brightest compact, roughly round blob inside the camera mask.
Three facts measured on AURIC frames shape the detector:

* The Q25 cameras clip at about 240 of 255 (the calibration files state ``saturation_val: 240``; the disc plateaus
  at 246-250), so the threshold is 240, not 255.
* Bright cumulus edges also clip, so brightness alone is not enough: the Sun's glow makes its neighbourhood the
  brightest place in the image, and the chosen blob must contain the maximum of the blurred image.
* Under haze the clipped region grows into a glow of 80 px radius whose centroid is biased by up to a degree, and
  on a dirty dome the glow is asymmetric, so blobs above ``max_area`` (a clear-sky disc is 50-120 px at half
  resolution) are refused rather than used, and the blob must stand well above the ring around it
  (``min_prominence``: a clear-sky disc exceeds its surroundings by 50-120, a cloud edge by 10-40).

Frames with no such blob (overcast, Sun behind an obstruction) or with several (reflections) are reported as not
detected rather than guessed. Pixel coordinates follow the pixel-centre convention of the image passed in; callers
that decode at reduced size rescale them (half resolution: u_full = 2 u + 0.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SunDetection:
    ok: bool
    u: float = float("nan")      # column of the blob centroid
    v: float = float("nan")      # row
    area: int = 0                # pixels of the chosen blob
    candidates: int = 0          # blobs that passed the shape test
    reason: str = ""


def detect_sun(image: np.ndarray, mask: np.ndarray | None = None, threshold: int = 240, min_area: int = 15,
               max_area: int = 400, min_fill: float = 0.5, max_aspect: float = 1.8, min_prominence: float = 50.0,
               require_peak: bool = True, blur_sigma: float = 4.0) -> SunDetection:
    """Find the Sun in an 8-bit RGB/BGR image (H, W, 3). `mask`: True = sky pixel; None = whole image.

    A blob is a candidate when its area is within [min_area, max_area], it fills at least `min_fill` of its bounding
    box with aspect ratio at most `max_aspect`, and its mean brightness exceeds a ring around it by `min_prominence`.
    With `require_peak` the chosen blob must also contain the maximum of the Gaussian-blurred image inside the mask.
    """
    import cv2

    bright_val = image.max(axis=2)
    bright = bright_val >= threshold
    if mask is not None:
        bright &= mask
    if not bright.any():
        return SunDetection(False, reason="no pixel above threshold")
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
    peak_label = -1
    if require_peak:
        blurred = cv2.GaussianBlur(bright_val.astype(np.float32), (0, 0), blur_sigma)
        if mask is not None:
            blurred[~mask] = -1.0
        py, px = np.unravel_index(int(np.argmax(blurred)), blurred.shape)
        peak_label = int(labels[py, px])
    found = []
    h_img, w_img = bright.shape
    for i in range(1, n):
        x, y, w, h, area = (int(s) for s in stats[i])
        if area < min_area or area > max_area:
            continue
        fill = area / float(w * h)
        aspect = max(w, h) / float(min(w, h))
        if fill < min_fill or aspect > max_aspect:
            continue
        pad = max(w, h)
        y0, y1 = max(0, y - pad), min(h_img, y + h + pad)
        x0, x1 = max(0, x - pad), min(w_img, x + w + pad)
        window = bright_val[y0:y1, x0:x1].astype(float)
        member = labels[y0:y1, x0:x1] == i
        ring = ~member
        if mask is not None:
            ring &= mask[y0:y1, x0:x1]
        if ring.sum() < 8:
            continue
        if float(window[member].mean() - window[ring].mean()) < min_prominence:
            continue
        found.append((area, float(centroids[i][0]), float(centroids[i][1]), i))
    if not found:
        return SunDetection(False, reason="no disc-like prominent blob")
    found.sort(reverse=True)
    if len(found) > 1 and found[0][0] < 3 * found[1][0]:
        return SunDetection(False, candidates=len(found), reason="several candidate blobs")
    area, u, v, label = found[0]
    if require_peak and label != peak_label:
        return SunDetection(False, candidates=len(found), reason="brightest region is not the blob")
    return SunDetection(True, u=u, v=v, area=area, candidates=len(found))
