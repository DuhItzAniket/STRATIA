import json
from pathlib import Path

from stratia import contract

ROOT = Path(__file__).resolve().parents[1]


def test_schema_class_lists_match_contract_constants():
    schema = json.loads((ROOT / "schemas/model_card.schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]
    assert tuple(props["genus_classes"]["const"]) == contract.GENUS_CLASSES
    assert tuple(props["etage_classes"]["const"]) == contract.ETAGE_CLASSES
    assert tuple(props["sky_parse_classes"]["const"]) == contract.SKY_PARSE_CLASSES
    assert tuple(props["layer_classes"]["const"]) == contract.LAYER_CLASSES
    assert schema["properties"]["contract_version"]["pattern"].startswith("^" + contract.CONTRACT_VERSION.split(".")[0])


def test_contract_doc_mentions_every_output():
    doc = (ROOT / "docs/contract.md").read_text(encoding="utf-8")
    for name in contract.INPUT_NAMES + contract.OUTPUT_NAMES:
        assert f"`{name}`" in doc, name


def test_output_shapes():
    s = contract.output_shapes(2, 512, 512, k_cbh=24)
    assert s["genus_logits"] == (2, 12) and s["sky_parse_logits"] == (2, 4, 128, 128) and s["cbh_probs"] == (2, 25)
