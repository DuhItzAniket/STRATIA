import pytest

openpyxl = pytest.importorskip("openpyxl")

from stratia.data.mgcd import load_mgcd  # noqa: E402


def test_load_mgcd_joins_weather(tmp_path):
    for split, nums in (("train", [1, 2]), ("test", [3])):
        folder = tmp_path / split / "1_cumulus"
        folder.mkdir(parents=True)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Number", "Name", "Temperature(℃)", "Humidity(%RH)", "Pressure(hpa)", "Wind speed(m/s)"])
        for n in nums:
            (folder / f"1_cumulus_{n:06d}.jpg").write_bytes(b"x")
            ws.append([n, f"1_cumulus_{n:06d}", 30.0 + n, 50.0, 1008.0, 1.0])
        wb.save(tmp_path / split / "1_cumulus.xlsx")
    (tmp_path / "train" / "notes").mkdir()            # non-class folders are ignored
    df = load_mgcd(tmp_path)
    assert len(df) == 3 and set(df.split) == {"train", "test"} and set(df.class_name) == {"cumulus"}
    row = df[df.stem == "1_cumulus_000002"].iloc[0]
    assert row.temperature_c == 32.0 and row.image_file == "train/1_cumulus/1_cumulus_000002.jpg"
