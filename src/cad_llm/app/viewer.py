"""Matplotlib 3D preview embedded in the desktop shell."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cad_llm.app import theme
from cad_llm.app.mesh import load_trimesh

# Viewer-specific colors (muted — not full neon)
_MESH_BASE = (0.52, 0.62, 0.72)
_GRID = "#2a2a34"
_AXIS_X = "#c45c5c"
_AXIS_Y = "#6ba368"
_AXIS_Z = "#6a8fc4"
_EDGE = (0.35, 0.45, 0.55, 0.18)


class CadViewerPanel(ctk.CTkFrame):
    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, fg_color=theme.PANEL, corner_radius=0, width=400, **kwargs)
        self.pack_propagate(False)

        self._view_center: np.ndarray | None = None
        self._view_half: float = 1.0

        header = ctk.CTkFrame(self, fg_color=theme.PANEL, corner_radius=0)
        header.pack(fill="x", padx=12, pady=(12, 0))
        ctk.CTkLabel(
            header,
            text="CAD PREVIEW",
            font=theme.FONT_TITLE,
            text_color=theme.ACCENT,
            anchor="w",
        ).pack(side="left")
        self._filename = ctk.CTkLabel(
            header,
            text="",
            font=theme.FONT_UI,
            text_color=theme.MUTED,
            anchor="e",
        )
        self._filename.pack(side="right", fill="x", expand=True)

        ctk.CTkLabel(
            self,
            text="drag to orbit  ·  scroll to zoom",
            font=theme.FONT_UI,
            text_color=theme.MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(2, 0))

        self._plot_host = tk.Frame(self, bg=theme.BG, highlightthickness=0)
        self._plot_host.pack(fill="both", expand=True, padx=12, pady=12)

        self._figure = Figure(figsize=(4.5, 4.5), facecolor=theme.BG)
        self._figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._axes = self._figure.add_subplot(111, projection="3d", facecolor=theme.BG)
        self._mpl_canvas = FigureCanvasTkAgg(self._figure, master=self._plot_host)
        widget = self._mpl_canvas.get_tk_widget()
        widget.configure(bg=theme.BG, highlightthickness=0)
        widget.pack(fill="both", expand=True)
        self._mpl_canvas.mpl_connect("scroll_event", self._on_scroll)
        self._show_empty()

    def _style_axes(self) -> None:
        self._axes.set_facecolor(theme.BG)
        self._axes.set_axis_off()
        self._axes.grid(False)

    def _apply_view_limits(self) -> None:
        if self._view_center is None:
            return
        c = self._view_center
        h = self._view_half
        self._axes.set_xlim(c[0] - h, c[0] + h)
        self._axes.set_ylim(c[1] - h, c[1] + h)
        self._axes.set_zlim(c[2] - h, c[2] + h)
        self._axes.set_box_aspect([1, 1, 1])

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes is not self._axes or self._view_center is None:
            return
        factor = 0.88 if event.button == "up" else 1.12
        self._view_half = max(self._view_half * factor, 1e-4)
        self._apply_view_limits()
        self._mpl_canvas.draw_idle()

    def _face_colors(self, mesh: Any) -> np.ndarray:
        normals = mesh.face_normals
        light = np.array([0.35, 0.25, 0.9], dtype=float)
        light /= np.linalg.norm(light)
        shade = 0.38 + 0.62 * np.clip(normals @ light, 0.0, 1.0)
        rgb = np.outer(shade, _MESH_BASE)
        return np.column_stack([rgb, np.full(len(shade), 0.96)])

    def _draw_floor_grid(self, center: np.ndarray, half: float, z_floor: float) -> None:
        steps = 10
        xs = np.linspace(center[0] - half, center[0] + half, steps)
        ys = np.linspace(center[1] - half, center[1] + half, steps)
        for x in xs:
            self._axes.plot(
                [x, x],
                [center[1] - half, center[1] + half],
                [z_floor, z_floor],
                color=_GRID,
                linewidth=0.45,
                alpha=0.55,
            )
        for y in ys:
            self._axes.plot(
                [center[0] - half, center[0] + half],
                [y, y],
                [z_floor, z_floor],
                color=_GRID,
                linewidth=0.45,
                alpha=0.55,
            )

    def _draw_axis_gizmo(self, origin: np.ndarray, half: float) -> None:
        length = half * 0.42
        o = origin
        axes = [
            (np.array([length, 0, 0]), _AXIS_X, "X"),
            (np.array([0, length, 0]), _AXIS_Y, "Y"),
            (np.array([0, 0, length]), _AXIS_Z, "Z"),
        ]
        for delta, color, label in axes:
            end = o + delta
            self._axes.plot(
                [o[0], end[0]],
                [o[1], end[1]],
                [o[2], end[2]],
                color=color,
                linewidth=1.4,
                alpha=0.85,
            )
            self._axes.text(
                end[0],
                end[1],
                end[2],
                label,
                color=color,
                fontsize=8,
                ha="center",
                va="center",
            )

    def _show_empty(self) -> None:
        self._axes.clear()
        self._style_axes()
        self._view_center = None
        self._filename.configure(text="waiting for export")
        self._mpl_canvas.draw_idle()

    def clear(self) -> None:
        self._show_empty()

    def show_mesh_path(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self._filename.configure(text="no export yet")
            self.clear()
            return
        try:
            mesh = load_trimesh(path)
        except Exception as exc:  # noqa: BLE001
            self._filename.configure(text=path.name)
            self._axes.clear()
            self._style_axes()
            self._axes.text2D(
                0.5,
                0.5,
                f"preview unavailable\n{exc}",
                transform=self._axes.transAxes,
                ha="center",
                va="center",
                color=theme.MUTED,
                fontsize=9,
            )
            self._mpl_canvas.draw_idle()
            return

        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        span = float(np.max(bounds[1] - bounds[0]))
        half = max(span * 0.72, 1e-3)
        z_floor = float(bounds[0][2])

        self._axes.clear()
        self._style_axes()

        self._draw_floor_grid(center, half, z_floor)
        self._draw_axis_gizmo(center, half)

        verts = mesh.vertices[mesh.faces]
        collection = Poly3DCollection(
            verts,
            facecolors=self._face_colors(mesh),
            edgecolors=_EDGE,
            linewidths=0.12,
        )
        self._axes.add_collection3d(collection)

        self._view_center = center
        self._view_half = half
        self._apply_view_limits()
        self._axes.view_init(elev=24, azim=-52)
        self._mpl_canvas.draw_idle()
        self._filename.configure(text=path.name)
