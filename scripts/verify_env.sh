#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> cad-llm environment check"
echo "    root: $ROOT"
echo

echo "==> Python / uv"
uv --version
uv run python --version
echo

echo "==> Project layout"
uv run cad-llm info
echo

echo "==> Ensuring local directories"
uv run cad-llm ensure-dirs
echo

echo "==> Core imports"
uv run python -c "import cad_llm; print(f'cad_llm {cad_llm.__version__}')"
echo

echo "==> Optional: MLX (Apple Silicon)"
if uv run python -c "import mlx; import mlx_lm; print('MLX stack OK')" 2>/dev/null; then
  :
else
  echo "    MLX not installed — run: uv sync"
fi
echo

echo "==> Optional: CAD kernel"
if uv run python -c "import cadquery; import build123d; print('CAD stack OK')" 2>/dev/null; then
  :
else
  echo "    CAD stack not installed — run: uv sync"
fi
echo

echo "Done."
