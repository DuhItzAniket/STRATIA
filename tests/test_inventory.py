import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import inventory  # noqa: E402


def test_describe_assigns_dataset_and_flags_junk(tmp_path):
    root = tmp_path
    (root / "A" / "sub").mkdir(parents=True)
    (root / "A" / "__MACOSX").mkdir()
    Image.new("RGB", (40, 30)).save(root / "A" / "sub" / "x.jpg")
    (root / "A" / "__MACOSX" / "._x.jpg").write_bytes(b"\0\1")
    (root / "A" / "broken.png").write_bytes(b"not an image")
    owners = [((root / "A").resolve(), "a")]
    good = inventory.describe(root / "A" / "sub" / "x.jpg", root, owners)
    assert (good["dataset"], good["width"], good["height"], good["mode"], good["junk"]) == ("a", 40, 30, "RGB", False)
    assert inventory.describe(root / "A" / "__MACOSX" / "._x.jpg", root, owners)["junk"]
    bad = inventory.describe(root / "A" / "broken.png", root, owners)
    assert bad["error"] and bad["width"] is None


def test_nested_registration_wins(tmp_path):
    (tmp_path / "S" / "inner").mkdir(parents=True)
    Image.new("L", (8, 8)).save(tmp_path / "S" / "inner" / "m.png")
    Image.new("L", (8, 8)).save(tmp_path / "S" / "o.png")
    reg = {"outer": {"local_path": "S"}, "inner": {"local_path": "S/inner"}}
    owners = inventory.owner_map(reg, tmp_path)
    assert inventory.describe(tmp_path / "S" / "inner" / "m.png", tmp_path, owners)["dataset"] == "inner"
    assert inventory.describe(tmp_path / "S" / "o.png", tmp_path, owners)["dataset"] == "outer"
