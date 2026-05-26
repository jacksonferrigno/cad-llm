from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_lm import generate, load

from cad_llm.config import settings
from cad_llm.inference.extract import extract_python_code

SYSTEM_PROMPT = (
    "You are a CAD code generator. Output only executable Python.\n"
    "Import rules (required):\n"
    "- CadQuery: always begin with `import cadquery as cq`. Never write `import cq`.\n"
    "- build123d: use `from build123d import ...` or `import build123d as bd`.\n"
    "CadQuery API (cadquery.readthedocs.io):\n"
    "- Start from `cq.Workplane(\"XY\")` and chain documented methods "
    "(.rect, .circle, .box, .extrude, .cut, .union, .faces, .chamfer).\n"
    "- Do not use `cq.Cylinder`, `cq.Cube`, or `Workplane.hexagon`; they are not CadQuery APIs.\n"
    "Output rules:\n"
    "- Output Python code only. No meta-commentary, explanation, or prose.\n"
    "- No markdown fences, no explanation, no comments unless required.\n"
    "- Assign the final shape to a variable named `result`."
)

VERTEX_SFT_MLX_MODEL = settings.resolve(Path("artifacts/models/qwen3-4b-vertex-sft-4bit"))


def resolve_model_id(model_id: str) -> str:
    """Resolve repo-relative paths before mlx_lm.load().

    mlx_lm treats non-existent paths as Hugging Face repo ids. A relative path
    like ``artifacts/models/foo`` fails when the shell cwd is not the repo root.
    """
    path = Path(model_id)
    if path.is_absolute() and path.exists():
        return str(path.resolve())

    if path.exists():
        return str(path.resolve())

    rooted = settings.resolve(path)
    if rooted.exists():
        return str(rooted.resolve())

    return model_id


@dataclass
class GenerationResult:
    prompt: str
    raw_response: str
    code: str


class CadGenerator:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = resolve_model_id(model_id or settings.mlx_model_id)
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model, self._tokenizer = load(self.model_id)

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        assert self._tokenizer is not None
        return self._tokenizer

    def format_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        user_prompt: str,
        *,
        max_tokens: int = 512,
        verbose: bool = False,
    ) -> GenerationResult:
        formatted = self.format_prompt(user_prompt)
        raw = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=verbose,
        )
        return GenerationResult(
            prompt=user_prompt,
            raw_response=raw,
            code=extract_python_code(raw),
        )
