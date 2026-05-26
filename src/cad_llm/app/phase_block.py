"""Inline phase indicator for the transcript.

Renders a compact pill for grouped agent activity. While running, a 3x3 grid
shimmers inside the pill. When finished, the animation stops and detail lines
can be revealed on click.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from cad_llm.app import theme


_PHASE_LABELS: dict[str, tuple[str, str]] = {
    "research": ("researching", "researched"),
    "implement": ("implementing", "implemented"),
    "chat": ("thinking", "responded"),
}

_GRID_N = 3
_CELL = 6
_GAP = 2
_CANVAS_PAD = 3
_CANVAS_SIZE = _GRID_N * _CELL + (_GRID_N - 1) * _GAP + _CANVAS_PAD * 2
_TICK_MS = 110
_DIM = "#26262c"
_LEVELS = ["#2d2d34", "#4a4a52", "#8a7a52", theme.ACCENT]


class PhaseBlock(ctk.CTkFrame):
    """Compact, clickable pill representing one agent activity block."""

    def __init__(
        self,
        master: Any,
        phase: str,
        *,
        on_toggle: Callable[["PhaseBlock"], None],
    ) -> None:
        super().__init__(
            master,
            fg_color=theme.SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=theme.BORDER,
        )
        self.phase = phase
        self._on_toggle = on_toggle
        self._expanded = False
        self._done = False
        self._step_count = 0
        self._error_count = 0
        self._tick = 0
        self._after_id: str | None = None

        self._caret = ctk.CTkLabel(
            self,
            text="▸",
            font=theme.FONT_LABEL,
            text_color=theme.SUBTLE,
            width=14,
        )
        self._caret.pack(side="left", padx=(10, 4), pady=6)

        self._canvas = tk.Canvas(
            self,
            width=_CANVAS_SIZE,
            height=_CANVAS_SIZE,
            bg=theme.SURFACE,
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack(side="left", padx=(0, 10), pady=6)
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

        active_label, _ = _PHASE_LABELS.get(phase, (phase, phase))
        self._label = ctk.CTkLabel(
            self,
            text=f"{active_label}…",
            font=theme.FONT_UI,
            text_color=theme.TEXT,
        )
        self._label.pack(side="left", pady=6)

        self._meta = ctk.CTkLabel(
            self,
            text="",
            font=theme.FONT_LABEL,
            text_color=theme.MUTED,
        )
        self._meta.pack(side="left", padx=(10, 12), pady=6)

        for widget in (self, self._caret, self._canvas, self._label, self._meta):
            widget.bind("<Button-1>", self._handle_click)
            widget.configure(cursor="hand2")

        self._schedule_tick()

    def _handle_click(self, _event: object = None) -> None:
        if not self._done or self._step_count == 0:
            return
        self._expanded = not self._expanded
        self._caret.configure(text="▾" if self._expanded else "▸")
        self._on_toggle(self)

    def _schedule_tick(self) -> None:
        self._after_id = self.after(_TICK_MS, self._animate)

    def _animate(self) -> None:
        if self._done:
            return
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

    def record_step(self, *, is_error: bool) -> None:
        self._step_count += 1
        if is_error:
            self._error_count += 1

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:  # noqa: BLE001
                pass
            self._after_id = None
        for rect in self._cells:
            self._canvas.itemconfigure(rect, fill=_DIM)
        center_idx = (_GRID_N // 2) * _GRID_N + (_GRID_N // 2)
        self._canvas.itemconfigure(self._cells[center_idx], fill=theme.ACCENT_DIM)
        _, done_label = _PHASE_LABELS.get(self.phase, (self.phase, self.phase))
        self._label.configure(text=done_label, text_color=theme.MUTED)
        bits = []
        if self._step_count:
            noun = "step" if self._step_count == 1 else "steps"
            bits.append(f"{self._step_count} {noun}")
        if self._error_count:
            bits.append(f"⚠ {self._error_count}")
        self._meta.configure(text="   ·   ".join(bits) if bits else "")
        if self._step_count > 0:
            self._caret.configure(text="▸", text_color=theme.MUTED)
        else:
            self._caret.configure(text=" ", text_color=theme.SUBTLE)
