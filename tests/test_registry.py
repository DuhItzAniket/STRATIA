import pytest
import yaml

from stratia.data.registry import load_registry


def test_repository_registry_is_valid():
    reg = load_registry()
    assert {"ccsn", "mgcd", "montenegro", "eye2sky", "almeria"} <= reg.keys()


def test_registry_rejects_bad_status(tmp_path):
    bad = {"datasets": {"x": {"name": "x", "tasks": [], "source_url": "u", "licence": "l",
                              "licence_status": "verified", "status": "somewhere"}}}
    p = tmp_path / "datasets.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_registry(p)
