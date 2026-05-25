"""SFT warm-up via mlx-tune."""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer
from mlx_tune.trainer import prepare_dataset

from cad_llm.config import settings
from cad_llm.data.tokens import build_sft_messages, count_sft_messages

DEFAULT_DATA = settings.resolve(settings.text2cadquery_dir) / "prepared" / "sft_train.jsonl"
DEFAULT_OUTPUT = settings.resolve(settings.checkpoints_dir) / "sft"
MAX_SEQ_LENGTH = 2048


def _format_sample(sample: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    messages = sample.get("messages")
    if messages is not None:
        return {"messages": messages}  # type: ignore[return-value]
    return {
        "messages": build_sft_messages(
            str(sample["prompt"]),
            str(sample["completion"]),
        )
    }


def _sample_messages(sample: dict[str, object]) -> list[dict[str, str]]:
    messages = sample.get("messages")
    if messages is not None:
        return messages  # type: ignore[return-value]
    return build_sft_messages(str(sample["prompt"]), str(sample["completion"]))


@contextlib.contextmanager
def _tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:

        class Tee:
            def write(self, data: str) -> None:
                sys.__stdout__.write(data)
                log_file.write(data)

            def flush(self) -> None:
                sys.__stdout__.flush()
                log_file.flush()

        old = sys.stdout
        sys.stdout = Tee()  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old


def run_sft(
    data_path: Path = DEFAULT_DATA,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    max_samples: int | None = None,
) -> Path:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run `cad-llm data prepare` first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    dataset = prepare_dataset(dataset_path=str(data_path))

    model, tokenizer = FastLanguageModel.from_pretrained(
        settings.mlx_model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        use_gradient_checkpointing=True,
    )

    keep = [
        i
        for i in range(len(dataset))
        if count_sft_messages(tokenizer, _sample_messages(dataset[i])) <= MAX_SEQ_LENGTH
    ]
    dropped = len(dataset) - len(keep)
    if dropped:
        print(f"Dropped {dropped} samples longer than {MAX_SEQ_LENGTH} tokens")
    dataset = dataset.select(keep)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now(tz=UTC).isoformat(),
                "model_id": settings.mlx_model_id,
                "data_path": str(data_path),
                "samples": len(dataset),
                "dropped_over_limit": dropped,
                "max_seq_length": MAX_SEQ_LENGTH,
                "learning_rate": 3e-5,
                "batch_size": 1,
                "grad_accum": 8,
                "epochs": 1,
                "logging_steps": 10,
                "grad_checkpoint": True,
            },
            indent=2,
        )
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset.select([0]),
        tokenizer=tokenizer,
        formatting_func=_format_sample,
        adapter_path="adapters",
        args=SFTConfig(
            output_dir=str(output_dir),
            learning_rate=3e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            max_seq_length=MAX_SEQ_LENGTH,
            logging_steps=10,
            save_steps=500,
            val_batches=0,
            grad_checkpoint=True,
        ),
    )

    with _tee_stdout(log_path):
        print(f"Logging to {log_path}")
        trainer.train()

    return output_dir / "adapters"
