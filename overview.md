# Local CAD LLM — Project Roadmap

## The pitch
4B parameter model that generates engineering CAD from plain English. 

---

## Stack
- **Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) (Vertex managed tuning: `qwen/qwen3@qwen3-4b`) → [mlx-community/Qwen3-4B-4bit](https://huggingface.co/mlx-community/Qwen3-4B-4bit) (local inference)
- **Training:** mlx-tune (Apple MLX, LoRA → GRPO)
- **Training data:** Text-to-CadQuery (~170k text + CadQuery pairs)
- **CAD kernel:** CadQuery + build123d (Python, OpenCascade)
- **Benchmark:** Text2CAD-Bench
- **Inference server:** mlx_lm.server (OpenAI-compatible local endpoint)
- **Demo UI:** FastAPI backend + Three.js 3D viewer frontend

---

## Roadmap

### Step 1 — Environment setup ✅
- Install mlx-tune, CadQuery, build123d
- Load Qwen3-4B via mlx_lm
- Verify inference works at usable speed on 48GB Mac

### Step 2 — Baseline (Text2CAD-Bench)
- Run vanilla Qwen zero-shot on Text2CAD-Bench prompts
- Measure compile rate, geometric validity, watertight rate
- This is the number to beat

### Step 3 — Data pipeline
- Download Text-to-CadQuery from Hugging Face (`CadQuery.zip` + captions CSV)
- Join text descriptions with CadQuery code → JSONL
- Split: **10k SFT** + **60k GRPO** (disjoint, fixed seed)

### Step 4 — SFT warm-up
- LoRA fine-tune on 10k samples via mlx-tune (rank 16, ~1–2 hrs)
- Re-run Text2CAD-Bench

### Step 5 — GRPO
- Reward: compile success + watertight solid + Chamfer Distance vs target code
- Fine-tune on 60k disjoint samples on top of SFT checkpoint
- Re-run Text2CAD-Bench

### Step 6 — Demo UI
- FastAPI server wrapping mlx_lm.server
- Frontend: text input → POST to local API → render STEP/GLB in Three.js viewer
- STEP file download button

### Step 7 — Demo recording
- Record screen: type prompt, watch 3D model appear, rotate it, download STEP
- Show Activity Monitor in corner — model running fully on-device

---

## Metrics

| Metric | What it measures |
|---|---|
| Compile rate | Does generated code run without errors |
| Geometric validity | Is output a watertight solid |
| Chamfer Distance | How close is generated shape to ground truth |
| Text2CAD-Bench score | Comparison to published baselines |

---

## Reward function (GRPO)

```python
def reward(generated_code: str, target_mesh) -> float:
    score = 0.0
    try:
        result = execute_cadquery(generated_code)
        score += 0.4  # compile success
        if is_watertight(result):
            score += 0.2  # valid solid
        cd = chamfer_distance(result, target_mesh)
        score += 0.4 * (1 / (1 + cd))  # geometry accuracy
    except Exception:
        pass
    return score
```
