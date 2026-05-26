"""Text2CAD-Bench download and prompt loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_DATASET = "AICAD/Text2CAD-Bench"
PROMPTS_FILE = "release.csv"
PROMPT_FIELD = "geo_prompt_en"


@dataclass
class BenchPrompt:
    id: str
    prompt: str


def download(base: Path) -> Path:
    target = base
    target.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=HF_DATASET,
            filename=PROMPTS_FILE,
            repo_type="dataset",
            local_dir=str(target),
        )
    )


def load_prompts(path: Path) -> list[BenchPrompt]:
    """Load Text2CAD-Bench geo prompts."""
    prompts: list[BenchPrompt] = []

    with path.open(newline="", encoding="latin-1") as handle:
        for row in csv.DictReader(handle):
            text = (row.get(PROMPT_FIELD) or "").strip()
            if text:
                prompts.append(BenchPrompt(id=row["id"], prompt=text))

    return prompts


def default_prompts_path(base: Path) -> Path:
    return base / PROMPTS_FILE
