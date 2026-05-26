# CAD generation

## Layout

- Prefer a multi-file layout inside the sandbox project:
  - **`src/main.py`** — entry point that wires parts together and produces the final solid or export step.
  - **`src/parts/`** — one module per reusable component (named clearly, imported from `main`).

Keep `main` thin: orchestration, parameters, and final assembly belong there; reusable geometry lives under `parts/`.

## CadQuery APIs

Before using **unfamiliar CadQuery APIs** or patterns you have not used confidently in this session:

1. Call **`search_cadquery_docs`** with a focused query (symbol, error message, or behavior).
2. Use ONLY API names and patterns from the returned snippets — never guess.
3. After sandbox errors, read the auto-fetched doc snippets before editing again.

## Export only

- Export meshes to `outputs/` with `cq.exporters.export(result, "outputs/name.step")`.
- Do **not** call `cq.show`, `show_object`, or `cq.Viewer` — the desktop app previews exports.
