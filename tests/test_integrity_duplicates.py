import numpy as np
import pandas as pd
from PIL import Image

from stratia.data.duplicates import duplicate_groups, hash_image, pair_summary
from stratia.data.duplicates import report_markdown as duplicates_report
from stratia.data.integrity import add_size_outliers, inspect_image, summarise
from stratia.data.integrity import report_markdown as integrity_report


def _picture(seed: int, size=(64, 48)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8), "RGB")


def test_inspect_describes_a_good_image_without_flags(tmp_path):
    p = tmp_path / "good.jpg"
    _picture(1).save(p, quality=90)
    r = inspect_image(p, 64, 48)
    assert r["flags"] == "" and r["error"] is None
    assert (r["format"], r["mode"], r["width"], r["height"], r["exif_orientation"]) == ("JPEG", "RGB", 64, 48, 1)
    assert 100 < r["mean"] < 155 and r["std"] > 40 and r["bytes"] > 0


def test_inspect_flags_each_kind_of_problem(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    assert inspect_image(tmp_path / "empty.jpg")["flags"] == "zero_bytes"
    assert inspect_image(tmp_path / "nowhere.jpg")["flags"] == "missing"

    good = tmp_path / "good.jpg"
    _picture(2, (400, 300)).save(good, quality=90)
    data = good.read_bytes()
    (tmp_path / "cut.jpg").write_bytes(data[: len(data) // 2])  # truncated: header fine, pixels missing
    cut = inspect_image(tmp_path / "cut.jpg")
    assert cut["flags"] == "corrupt" and cut["error"]
    (tmp_path / "text.jpg").write_bytes(b"this is not a picture")
    assert inspect_image(tmp_path / "text.jpg")["flags"] == "corrupt"

    Image.new("RGBA", (32, 32), (1, 2, 3, 4)).save(tmp_path / "rgba.png")
    assert inspect_image(tmp_path / "rgba.png")["flags"] == "odd_mode|blank"  # uniform colour: no contrast either

    rotated = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    _picture(3).save(rotated, exif=exif)
    assert inspect_image(rotated, 64, 48)["flags"] == "exif_rotation"

    assert inspect_image(good, 640, 480)["flags"] == "size_mismatch"


def test_size_outliers_are_judged_against_the_dataset_median():
    table = pd.DataFrame({"dataset": ["a"] * 5 + ["b"], "width": [100, 100, 100, 100, 1000, 10], "height": [100] * 5 + [10],
                          "flags": ["", "", "", "blank", "", ""]})
    out = add_size_outliers(table)
    assert list(out["flags"]) == ["", "", "", "blank", "size_outlier", ""]
    summary = summarise(out.assign(bytes=1000, image_file="x", sample_id="x", format="JPEG", mode="RGB",
                                   exif_orientation=1, mean=0.0, std=10.0, error=None))
    assert summary.set_index("dataset").loc["a", ["images", "clean", "blank", "size_outlier"]].tolist() == [5, 3, 1, 1]
    text = integrity_report(out.assign(bytes=1000, image_file="x", sample_id="x", format="JPEG", mode="RGB",
                                       exif_orientation=1, mean=0.0, std=10.0, error=None))
    assert "| a | 5 |" in text and "size_outlier" in text


def test_exact_duplicates_by_bytes_and_by_pixels(tmp_path):
    original = _picture(7)
    original.save(tmp_path / "a.png")
    (tmp_path / "a_copy.png").write_bytes((tmp_path / "a.png").read_bytes())  # same bytes
    original.save(tmp_path / "a_resaved.png", compress_level=9)             # same pixels, other bytes
    _picture(8).save(tmp_path / "b.png")
    (tmp_path / "broken.png").write_bytes(b"not a picture")

    rows = []
    for name, dataset in [("a.png", "x"), ("a_copy.png", "x"), ("a_resaved.png", "y"), ("b.png", "y"), ("broken.png", "y")]:
        rows.append({"sample_id": f"{dataset}:{name}", "dataset": dataset, "image_file": name, **hash_image(tmp_path / name)})
    table = pd.DataFrame(rows)
    assert table.loc[0, "file_sha256"] == table.loc[1, "file_sha256"] != table.loc[2, "file_sha256"]
    assert table.loc[0, "pixel_sha256"] == table.loc[1, "pixel_sha256"] == table.loc[2, "pixel_sha256"]
    assert pd.isna(table.loc[4, "pixel_sha256"]) and table.loc[4, "error"] and table.loc[4, "file_sha256"]

    groups = duplicate_groups(table)
    assert len(groups) == 3 and groups["dup_group"].nunique() == 1
    assert groups["cross_dataset"].all() and set(groups["group_datasets"]) == {"x,y"}
    assert groups.set_index("image_file")["by_file"].to_dict() == {"a.png": True, "a_copy.png": True, "a_resaved.png": False}
    assert pair_summary(groups).iloc[0].to_dict() == {"datasets": "x,y", "groups": 1, "images": 3}
    text = duplicates_report(table, groups)
    assert "**3**" in text and "| x,y | 1 | 3 |" in text and "a_resaved.png" in text


def test_no_duplicates_gives_empty_tables(tmp_path):
    _picture(1).save(tmp_path / "a.png")
    _picture(2).save(tmp_path / "b.png")
    rows = [{"sample_id": n, "dataset": "x", "image_file": n, **hash_image(tmp_path / n)} for n in ("a.png", "b.png")]
    table = pd.DataFrame(rows)
    groups = duplicate_groups(table)
    assert groups.empty and pair_summary(groups).empty
    assert "**0**" in duplicates_report(table, groups)
