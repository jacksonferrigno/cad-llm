# CAD generation

## Layout

- **`src/main.py`** — entry point; wires parts and exports.
- **`src/parts/`** — optional reusable modules.

Keep `main` thin.

## Use the docs already in your system prompt

Turn 1 includes CadQuery doc snippets for the user's request. **Use those first.**
Only call `search_cadquery_docs` if you need a symbol not covered there or after a sandbox failure.

## Typical pattern (box + circular through-hole)

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

## Example: circular flange + square through-hole

**Example query:** `generate 90mm circular flange, 15mm thick, 30mm square through-hole at center`

This is a reference example — adapt dimensions to the user's request:

```python
import cadquery as cq

FLANGE_DIAMETER = 90.0
FLANGE_THICKNESS = 15.0
HOLE_SIDE = 30.0

result = (
    cq.Workplane("XY")
    .circle(FLANGE_DIAMETER / 2)
    .extrude(FLANGE_THICKNESS)
    .faces(">Z")
    .workplane()
    .rect(HOLE_SIDE, HOLE_SIDE)
    .cutThruAll()
)
cq.exporters.export(result, "outputs/flange.step")
```

- Circular parts: `.circle(radius).extrude(thickness)` — **not** `.box()` for a disk/flange.
- Square through-hole: `.rect(width, height).cutThruAll()` on the top face workplane.
- Do **not** use `forConstruction=True` for cutting geometry.
- Do **not** call `.vertices()` before `.cutThruAll()` unless placing features at corners.

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
