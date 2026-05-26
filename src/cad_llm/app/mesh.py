"""Load exported meshes from project outputs for preview."""

from __future__ import annotations

import tempfile
from pathlib import Path

MESH_SUFFIXES = (".stl", ".step", ".stp", ".glb", ".obj")
_STEP_SUFFIXES = {".step", ".stp"}


def latest_mesh(outputs_dir: Path) -> Path | None:
    if not outputs_dir.is_dir():
        return None
    files = [p for p in outputs_dir.rglob("*") if p.suffix.lower() in MESH_SUFFIXES and p.is_file()]
    if not files:
        return None
    stl_files = [p for p in files if p.suffix.lower() == ".stl"]
    if stl_files:
        return max(stl_files, key=lambda p: p.stat().st_mtime)
    return max(files, key=lambda p: p.stat().st_mtime)


def _load_step_via_cadquery(path: Path):
    import cadquery as cq

    shape = cq.importers.importStep(str(path))
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = Path(tmp.name)
    try:
        cq.exporters.export(shape, str(stl_path))
        return _load_trimesh_file(stl_path)
    finally:
        stl_path.unlink(missing_ok=True)


def _load_trimesh_file(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if hasattr(g, "faces")]
        if not geometries:
            msg = f"No mesh geometry in {path.name}"
            raise ValueError(msg)
        return trimesh.util.concatenate(geometries)
    return loaded


def load_trimesh(path: Path):
    suffix = path.suffix.lower()
    if suffix in _STEP_SUFFIXES:
        try:
            return _load_trimesh_file(path)
        except Exception:
            return _load_step_via_cadquery(path)
    return _load_trimesh_file(path)
