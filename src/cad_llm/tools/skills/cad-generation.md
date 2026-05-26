# CAD generation

## Layout

- **`src/main.py`** — entry point; wires parts and exports.
- **`src/parts/`** — optional reusable modules.

Keep `main` thin.

## Use the docs already in your system prompt

Turn 1 includes CadQuery doc snippets for the user's request. **Use those first.**
Only call `search_cadquery_docs` if you need a symbol not covered there or after a sandbox failure.

## Typical pattern (box + through-hole)

```python
import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(50, 50, 50, centered=(True, True, True))
    .faces(">Z")
    .workplane()
    .circle(10)  # 20mm diameter hole
    .cutThruAll()
)
cq.exporters.export(result, "outputs/part.step")
```

- `box(..., centered=(True, True, True))` — centered at origin. Do **not** use `Workplane.center()` for this; `center(x, y)` only offsets a workplane on a face.
- Holes: `.cut()`, `.cutBlind()`, or `.cutThruAll()` — never `.union()` for material removal.

## Export only

- Export to `outputs/` with `cq.exporters.export(result, "outputs/name.step")`.
- Do **not** call `cq.show`, `show_object`, or `cq.Viewer`.

## Review (required before finishing)

After sandbox `exit_code=0`:

1. Re-read the code (you will be shown `src/main.py`).
2. Check every requirement: dimensions, position, cut vs add, axis, exports.
3. If wrong: fix, sandbox again, review again.
4. Only then give a brief final summary.

Do not claim success without verifying the code matches the request.
