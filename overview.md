# Local CAD LLM — Project Roadmap

## The pitch
7B parameter model, fine-tuned entirely on a MacBook Pro (48GB), generates engineering CAD from plain English. No cloud, no API, no GPU rental.

---

## Stack
- **Model:** Qwen2.5-Coder-7B-Instruct (4-bit, ~5GB)
- **Training:** mlx-tune (Apple MLX, LoRA)
- **CAD kernel:** CadQuery + build123d (Python, OpenCascade)
- **Inference server:** mlx_lm.server (OpenAI-compatible local endpoint)
- **Demo UI:** FastAPI backend + Three.js 3D viewer frontend

---

## Roadmap

### Step 1 — Environment setup
- Install mlx-tune, CadQuery, build123d
- Load Qwen2.5-Coder-7B-Instruct via mlx_lm
- Verify inference works at usable speed on 48GB Mac

### Step 2 — Baseline
- Run vanilla model zero-shot on 20-30 CAD prompts
- Measure compile rate, geometric validity, eyeball output quality
- This is the number to beat

### Step 3 — Data pipeline
- Scrape real CadQuery scripts from:
  - `github.com/CadQuery/awesome-cadquery`
  - `cadquery-contrib`
  - `github.com/gumyr/build123d` examples folder
  - Raw GitHub search: `language:Python cadquery`
- Filter: run each script, keep only ones that compile and produce a watertight solid
- For each passing script, call an LLM to generate a natural language description — this gives you the prompt side of the pair
- Output: JSONL file of `{ "prompt": "...", "completion": "..." }` pairs

### Step 4 — Dataset curation
- From scraped pairs, filter down to 8-10k highest quality
- Weight toward engineering-intent vocabulary: flanges, brackets, threads, heat exchangers, shafts, enclosures, gears, manifolds
- Split into train / eval sets

### Step 5 — SFT run
- LoRA fine-tune on curated dataset overnight (rank 16, ~4-6hrs)
- Eval compile rate + Chamfer Distance against baseline from Step 2
- Iterate: look at failures, add targeted examples, re-run

### Step 6 — GRPO
- Reward function: compile success + watertight solid check + Chamfer Distance vs target
- No neural reward model — reward is deterministic geometry execution
- Run RL training on top of SFT checkpoint

### Step 7 — Benchmark eval
- Run against Text2CAD-Bench (600 human-curated samples, published May 2026)
- Compare compile rate, Chamfer Distance, geometric validity to published baselines
- Document delta vs vanilla Qwen2.5-Coder-7B

### Step 8 — Demo UI
- FastAPI server wrapping mlx_lm.server
- Frontend: text input → POST to local API → render STEP/GLB in Three.js viewer
- STEP file download button
- Optional: side-by-side baseline vs fine-tuned output on same prompt

### Step 9 — Demo recording
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