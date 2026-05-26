"""Inline activity indicator — shimmer grid + single status byline."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import customtkinter as ctk

from cad_llm.app import theme

_GRID_N = 3
_CELL = 6
_GAP = 2
_CANVAS_PAD = 3
_CANVAS_SIZE = _GRID_N * _CELL + (_GRID_N - 1) * _GAP + _CANVAS_PAD * 2
_TICK_MS = 110
_DIM = "#26262c"
_LEVELS = ["#2d2d34", "#4a4a52", "#8a7a52", theme.ACCENT]


class ActivityBlock(ctk.CTkFrame):
    """Compact status row: animated grid + one updating byline."""

    def __init__(self, master: Any) -> None:
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=theme.BORDER,
        )
        self._tick = 0
        self._after_id: str | None = None
        self._embed_index: str | None = None

        self._canvas = tk.Canvas(
            self,
            width=_CANVAS_SIZE,
            height=_CANVAS_SIZE,
            bg=theme.SURFACE,
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack(side="left", padx=(10, 10), pady=6)
        self._cells: list[int] = []
        for row in range(_GRID_N):
            for col in range(_GRID_N):
                x0 = _CANVAS_PAD + col * (_CELL + _GAP)
                y0 = _CANVAS_PAD + row * (_CELL + _GAP)
                rect = self._canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + _CELL,
                    y0 + _CELL,
                    fill=_DIM,
                    outline="",
                )
                self._cells.append(rect)

        self._label = ctk.CTkLabel(
            self,
            text="thinking…",
            font=theme.FONT_UI,
            text_color=theme.MUTED,
        )
        self._label.pack(side="left", padx=(0, 12), pady=6)

        self._schedule_tick()

    def set_status(self, text: str) -> None:
        if text:
            self._label.configure(text=text)

    def stop(self) -> None:
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:  # noqa: BLE001
                pass
            self._after_id = None
        for rect in self._cells:
            self._canvas.itemconfigure(rect, fill=_DIM)

    def _schedule_tick(self) -> None:
        self._after_id = self.after(_TICK_MS, self._animate)

    def _animate(self) -> None:
        self._tick = (self._tick + 1) % 24
        n_levels = len(_LEVELS)
        for i, rect in enumerate(self._cells):
            row = i // _GRID_N
            col = i % _GRID_N
            phase_offset = (row + col) * 2
            value = (self._tick + phase_offset) % (n_levels * 2)
            idx = value if value < n_levels else (n_levels * 2 - 1 - value)
            self._canvas.itemconfigure(rect, fill=_LEVELS[idx])
        self._schedule_tick()
