"""Download and prepare the Text-to-CadQuery training dataset."""

from __future__ import annotations

import csv
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from cad_llm.data.tokens import build_sft_messages

HF_REPO = "ricemonster/NeurIPS11092"
ZIP_NAME = "CadQuery.zip"
CSV_NAME = "text2cad_v1.1.csv"
SKIP_CSV_FIELDS = frozenset({"uid", "keywords", "nli_data"})


@dataclass
class Sample:
    uid: str
    code: str
    prompts: list[str]


@dataclass
class PrepareSummary:
    total_scripts: int
    total_rows: int
    sft_path: Path
    sft_val_path: Path
    grpo_path: Path
    sft_scripts: int
    sft_val_scripts: int
    grpo_scripts: int
    sft_rows: int
    sft_val_rows: int
    grpo_rows: int


def raw_dir(base: Path) -> Path:
    return base / "raw"


def prepared_dir(base: Path) -> Path:
    return base / "prepared"


def download(base: Path) -> tuple[Path, Path]:
    """Download CadQuery.zip and text2cad_v1.1.csv from Hugging Face."""
    target = raw_dir(base)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=ZIP_NAME,
            repo_type="model",
            local_dir=str(target),
        )
    )
    csv_path = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=CSV_NAME,
            repo_type="model",
            local_dir=str(target),
        )
    )
    return zip_path, csv_path


def _load_prompts(csv_path: Path) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            texts = [
                value.strip()
                for key, value in row.items()
                if key not in SKIP_CSV_FIELDS and (value or "").strip()
            ]
            if texts:
                prompts[row["uid"]] = texts
    return prompts


def _uid_from_zip_path(path: str) -> str | None:
    parts = Path(path).parts
    if len(parts) != 3 or parts[0] != "CQ" or not path.endswith(".py"):
        return None
    stem = parts[2][:-3]
    return f"{parts[1]}/{stem}"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _expand_sft_rows(samples: list[Sample]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        for prompt in sample.prompts:
            rows.append({"messages": build_sft_messages(prompt, sample.code)})
    return rows


def _expand_grpo_rows(samples: list[Sample]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in samples:
        for prompt in sample.prompts:
            rows.append({"prompt": prompt, "completion": sample.code})
    return rows


def prepare(
    base: Path,
    *,
    sft_size: int = 10_000,
    grpo_size: int = 60_000,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> PrepareSummary:
    """Join all captions + CadQuery code and write SFT/GRPO JSONL splits.

    sft_size / grpo_size are unique script counts. Each script emits one row
    per non-empty caption field in the CSV. SFT scripts are split into train
    and validation at the script level (no caption leakage across splits).
    """
    zip_path = raw_dir(base) / ZIP_NAME
    csv_path = raw_dir(base) / CSV_NAME
    if not zip_path.exists() or not csv_path.exists():
        msg = "Missing raw files. Run `cad-llm data download` first."
        raise FileNotFoundError(msg)

    prompts = _load_prompts(csv_path)
    samples: list[Sample] = []

    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            uid = _uid_from_zip_path(name)
            if uid is None or uid not in prompts:
                continue
            code = archive.read(name).decode("utf-8").strip()
            if not code:
                continue
            samples.append(Sample(uid=uid, code=code, prompts=prompts[uid]))

    rng = random.Random(seed)
    rng.shuffle(samples)

    sft_pool = samples[:sft_size]
    grpo_samples = samples[sft_size : sft_size + grpo_size]
    val_scripts = max(1, int(len(sft_pool) * val_ratio))
    if val_scripts >= len(sft_pool):
        val_scripts = max(1, len(sft_pool) // 20)
    sft_val_samples = sft_pool[:val_scripts]
    sft_train_samples = sft_pool[val_scripts:]
    sft_rows = _expand_sft_rows(sft_train_samples)
    sft_val_rows = _expand_sft_rows(sft_val_samples)
    grpo_rows = _expand_grpo_rows(grpo_samples)

    out = prepared_dir(base)
    sft_path = out / "sft_train.jsonl"
    sft_val_path = out / "sft_val.jsonl"
    grpo_path = out / "grpo_train.jsonl"
    _write_jsonl(sft_path, sft_rows)
    _write_jsonl(sft_val_path, sft_val_rows)
    _write_jsonl(grpo_path, grpo_rows)

    manifest = {
        "source": HF_REPO,
        "seed": seed,
        "val_ratio": val_ratio,
        "total_scripts": len(samples),
        "total_rows": len(sft_rows) + len(sft_val_rows) + len(grpo_rows),
        "sft_train_scripts": len(sft_train_samples),
        "sft_val_scripts": len(sft_val_samples),
        "grpo_scripts": len(grpo_samples),
        "sft_train_rows": len(sft_rows),
        "sft_val_rows": len(sft_val_rows),
        "grpo_rows": len(grpo_rows),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return PrepareSummary(
        total_scripts=len(samples),
        total_rows=len(sft_rows) + len(sft_val_rows) + len(grpo_rows),
        sft_path=sft_path,
        sft_val_path=sft_val_path,
        grpo_path=grpo_path,
        sft_scripts=len(sft_train_samples),
        sft_val_scripts=len(sft_val_samples),
        grpo_scripts=len(grpo_samples),
        sft_rows=len(sft_rows),
        sft_val_rows=len(sft_val_rows),
        grpo_rows=len(grpo_rows),
    )
