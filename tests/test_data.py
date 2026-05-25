import csv
import json
import zipfile
from pathlib import Path

import pytest

from cad_llm.data.text2cadquery import _uid_from_zip_path, prepare
from cad_llm.inference.generate import SYSTEM_PROMPT


@pytest.fixture
def sample_raw(tmp_path: Path) -> Path:
    base = tmp_path / "text2cadquery"
    raw = base / "raw"
    raw.mkdir(parents=True)

    csv_path = raw / "text2cad_v1.1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uid", "beginner", "intermediate", "expert", "abstract", "keywords"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uid": "0001/00000001",
                "beginner": "Make a cube.",
                "intermediate": "Extrude a cube.",
                "expert": "Sketch and extrude a cube.",
                "abstract": "A cube.",
                "keywords": "cube",
            }
        )
        writer.writerow(
            {
                "uid": "0001/00000002",
                "beginner": "Make a plate.",
                "intermediate": "Extrude a plate.",
                "expert": "Sketch and extrude a plate.",
                "abstract": "A plate.",
                "keywords": "plate",
            }
        )
        writer.writerow(
            {
                "uid": "0001/00000003",
                "beginner": "Make a cylinder.",
                "intermediate": "Extrude a cylinder.",
                "expert": "Sketch and extrude a cylinder.",
                "abstract": "A cylinder.",
                "keywords": "cylinder",
            }
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


def test_prepare_uses_all_captions(sample_raw: Path) -> None:
    summary = prepare(sample_raw, sft_size=2, grpo_size=1, seed=42, val_ratio=0.5)
    assert summary.total_scripts == 3
    assert summary.sft_scripts == 1
    assert summary.sft_val_scripts == 1
    assert summary.grpo_scripts == 1
    assert summary.sft_rows == 4
    assert summary.sft_val_rows == 4
    assert summary.grpo_rows == 4
    assert summary.sft_val_path.name == "sft_val.jsonl"

    sft_lines = summary.sft_path.read_text().strip().splitlines()
    row = json.loads(sft_lines[0])
    assert set(row.keys()) == {"messages"}
    assert row["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["messages"][1]["role"] == "user"
    assert row["messages"][2]["role"] == "assistant"

    val_lines = summary.sft_val_path.read_text().strip().splitlines()
    assert len(val_lines) == 4
    val_row = json.loads(val_lines[0])
    assert val_row["messages"][0]["content"] == SYSTEM_PROMPT

    grpo_row = json.loads(summary.grpo_path.read_text().strip().splitlines()[0])
    assert set(grpo_row.keys()) == {"prompt", "completion"}
