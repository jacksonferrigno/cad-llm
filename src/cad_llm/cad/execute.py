import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cadquery as cq
import trimesh


@dataclass
class ExecutionResult:
    success: bool
    error: str | None
    has_geometry: bool
    is_watertight: bool
    library: str | None


def _export_workplane(workplane: cq.Workplane, path: Path) -> None:
    from cadquery import exporters

    exporters.export(workplane, str(path))


def _export_build123d(part: Any, path: Path) -> None:
    from build123d import export_stl

    export_stl(part, str(path))


def _mesh_is_watertight(path: Path) -> bool:
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return bool(mesh.is_watertight)


def _check_watertight(obj: Any) -> tuple[bool, str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        stl_path = Path(tmp) / "part.stl"
        if isinstance(obj, cq.Workplane):
            _export_workplane(obj, stl_path)
            return _mesh_is_watertight(stl_path), "cadquery"

        module = type(obj).__module__
        if module.startswith("build123d"):
            _export_build123d(obj, stl_path)
            return _mesh_is_watertight(stl_path), "build123d"

    return False, None


def _collect_geometry(
    namespace: dict[str, Any],
    captured: list[Any],
) -> list[tuple[Any, str]]:
    objects: list[tuple[Any, str]] = []

    for obj in captured:
        if isinstance(obj, cq.Workplane):
            objects.append((obj, "cadquery"))
        elif type(obj).__module__.startswith("build123d"):
            objects.append((obj, "build123d"))

    for value in namespace.values():
        if isinstance(value, cq.Workplane):
            objects.append((value, "cadquery"))
        elif type(value).__module__.startswith("build123d") and type(value).__name__ in {
            "Part",
            "Compound",
            "Shape",
        }:
            objects.append((value, "build123d"))

    deduped: list[tuple[Any, str]] = []
    seen: set[int] = set()
    for obj, library in objects:
        obj_id = id(obj)
        if obj_id not in seen:
            seen.add(obj_id)
            deduped.append((obj, library))
    return deduped


def execute_cad_code(code: str) -> ExecutionResult:
    import build123d

    captured: list[Any] = []

    def capture(obj: Any, *_args: Any, **_kwargs: Any) -> None:
        captured.append(obj)

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "cadquery": cq,
        "cq": cq,
        "build123d": build123d,
        "bd": build123d,
        "show_object": capture,
        "show": capture,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as exc:
        return ExecutionResult(
            success=False,
            error=str(exc),
            has_geometry=False,
            is_watertight=False,
            library=None,
        )

    geometry = _collect_geometry(namespace, captured)
    if not geometry:
        return ExecutionResult(
            success=True,
            error=None,
            has_geometry=False,
            is_watertight=False,
            library=None,
        )

    obj, library = geometry[-1]
    try:
        is_watertight, detected_library = _check_watertight(obj)
    except Exception as exc:
        return ExecutionResult(
            success=True,
            error=str(exc),
            has_geometry=True,
            is_watertight=False,
            library=library,
        )

    return ExecutionResult(
        success=True,
        error=None,
        has_geometry=True,
        is_watertight=is_watertight,
        library=detected_library or library,
    )
