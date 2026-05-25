import csv
import json
import zipfile
from pathlib import Path

import pytest

from cad_llm.data.text2cadquery import _uid_from_zip_path, prepare


@pytest.fixture
def sample_raw(tmp_path: Path) -> Path:
    base = tmp_path / "text2cadquery"
    raw = base / "raw"
    raw.mkdir(parents=True)

    csv_path = raw / "text2cad_v1.1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "beginner", "abstract"])
        writer.writeheader()
        writer.writerow(
            {"uid": "0001/00000001", "beginner": "Make a cube.", "abstract": "A cube."}
        )
        writer.writerow(
            {"uid": "0001/00000002", "beginner": "Make a plate.", "abstract": "A plate."}
        )
        writer.writerow(
            {"uid": "0001/00000003", "beginner": "Make a cylinder.", "abstract": "A cyl."}
        )

    zip_path = raw / "CadQuery.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "CQ/0001/00000001.py",
            "import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)",
        )
        archive.writestr(
            "CQ/0001/00000002.py",
            "import cadquery as cq\nresult = cq.Workplane('XY').box(2,2,1)",
        )
        archive.writestr(
            "CQ/0001/00000003.py",
            "import cadquery as cq\nresult = cq.Workplane('XY').cylinder(1, 2)",
        )

    return base


def test_uid_from_zip_path() -> None:
    assert _uid_from_zip_path("CQ/0001/00000001.py") == "0001/00000001"
    assert _uid_from_zip_path("CQ/0001/") is None


def test_prepare_splits(sample_raw: Path) -> None:
    summary = prepare(sample_raw, sft_size=1, grpo_size=2, seed=42)
    assert summary.total_pairs == 3
    assert summary.sft_count == 1
    assert summary.grpo_count == 2

    sft_lines = summary.sft_path.read_text().strip().splitlines()
    grpo_lines = summary.grpo_path.read_text().strip().splitlines()
    assert len(sft_lines) == 1
    assert len(grpo_lines) == 2

    row = json.loads(sft_lines[0])
    assert "prompt" in row and "completion" in row
