import numpy as np
import pytest
import yaml

from stratia.labels import ontology as on


def test_the_shipped_ontology_is_consistent():
    ont = on.load_ontology()
    assert on.validate(ont) == [] and ont.version == 1
    assert len(ont.genera) == 10 and ont.genus_classes[-2:] == ["clear", "contrail"] and len(ont.genus_classes) == 12
    assert ont.etages_of(["Ci", "Cc", "Cs"]) == {"high"} and ont.etages_of(["Cu", "Cb", "Sc", "St"]) == {"low"}
    assert ont.etages_of(["Ac", "As", "Ns"]) == {"mid"} and ont.etages_of(["Ci", "Cu"]) == {"high", "low"}


def test_heights_and_codes():
    ont = on.load_ontology()
    assert ont.etage_heights_m("high") == (6000.0, float("inf"))
    assert [ont.etage_of_height(h) for h in (0, 1999, 2000, 5999, 6000, 12000)] == ["low", "low", "mid", "mid", "high", "high"]
    assert ont.etage_of_height(float("nan")) is None and ont.etage_of_height(None) is None
    assert ont.genera_for_code("CL", "8") == ["Cu", "Sc"] and ont.genera_for_code("CL", 0) == []
    assert ont.genera_for_code("CM", "/") is None and ont.genera_for_code("CH", "9") == ["Cc"]
    assert ont.h_range_m("5") == (600.0, 1000.0) and ont.h_range_m("9") == (2500.0, float("inf")) and ont.h_range_m("/") is None
    assert ont.oktas("8") == 8 and ont.oktas("9") is None


def test_dataset_classes_are_all_mapped_or_excluded():
    ont = on.load_ontology()
    assert ont.dataset_class("ccsn", "Ct") == {"genera": [], "extra": "contrail"}
    mixed = ont.dataset_class("mgcd", "mixed")
    assert mixed["genera"] is None and mixed["cloud"] is True
    assert ont.dataset_class("mgcd", "altocumulus")["genera"] == ["Ac", "Cc"]
    assert ont.dataset_class("swimcat", "A-sky")["extra"] == "clear"
    assert set(ont.datasets["montenegro"]["altitude_class"]) == {"Clear", "Low clouds", "Middle clouds", "High clouds",
                                                                 "Clouds of vertical development"}


def test_validation_catches_broken_files(tmp_path):
    raw = yaml.safe_load(on.DEFAULT_PATH.read_text(encoding="utf-8"))
    raw["genera"]["Cu"]["etage"] = "middle"                                   # not an étage name
    raw["datasets"]["ccsn"]["classes"]["Xx"] = {"genera": ["Zz"]}            # unknown genus
    raw["datasets"]["mgcd"]["classes"]["mixed"] = {"genera": None}           # null genera without cloud: true
    del raw["wmo_codes"]["CL"]["9"]                                           # incomplete code table
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        on.load_ontology(p)
    msg = str(e.value)
    assert "genus Cu" in msg and "unknown genus 'Zz'" in msg and "cloud: true" in msg and "missing codes ['9']" in msg
    assert np.isfinite(on.load_ontology().etage_heights_m("low")[1])
