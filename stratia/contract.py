"""stratia-contract v1 constants (see docs/contract.md). Single source of truth for class orders."""

CONTRACT_VERSION = "1.0"
PATCH_SIZE = 16
DEFAULT_INPUT_SIZE = 512

GENUS_CLASSES = ("Ci", "Cc", "Cs", "Ac", "As", "Ns", "Sc", "St", "Cu", "Cb", "clear", "contrail")
ETAGE_CLASSES = ("low", "mid", "high")
SKY_PARSE_CLASSES = ("invalid", "sky", "cloud", "sun_glare")
LAYER_CLASSES = ("low", "mid", "high")
N_OKTAS = 9  # 0..8

RAY_MAP_CHANNELS = 4   # unit direction (x toward Sun azimuth, y, z up) + valid flag
META_DIM = 3           # cos(Sun zenith), sin(Sun zenith), valid

INPUT_NAMES = ("image", "ray_map", "meta")
OUTPUT_NAMES = ("genus_logits", "etage_logits", "oktas_probs", "sky_parse_logits", "layer_logits", "cbh_probs",
                "ood_score")


def output_shapes(n: int, h: int = DEFAULT_INPUT_SIZE, w: int = DEFAULT_INPUT_SIZE, k_cbh: int = 24) -> dict:
    """Expected output shapes for a batch of n images (used by contract tests)."""
    return {
        "genus_logits": (n, len(GENUS_CLASSES)),
        "etage_logits": (n, len(ETAGE_CLASSES)),
        "oktas_probs": (n, N_OKTAS),
        "sky_parse_logits": (n, len(SKY_PARSE_CLASSES), h // 4, w // 4),
        "layer_logits": (n, len(LAYER_CLASSES), h // 4, w // 4),
        "cbh_probs": (n, k_cbh + 1),
        "ood_score": (n, 1),
    }
