"""Native desktop shell — terminal chat + CAD preview."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

import customtkinter as ctk

from cad_llm.agent.session import AgentSession
from cad_llm.agent.steps import AgentStep
from cad_llm.app.formatting import format_assistant_reply, format_step
from cad_llm.app.mesh import latest_mesh
from cad_llm.app.theme import (
    ACCENT,
    AGENT,
    BG,
    BORDER,
    DIVIDER,
    ERROR,
    FONT_LABEL,
    FONT_MONO,
    FONT_TITLE,
    FONT_UI,
    MUTED,
    OK,
    PANEL,
    SIDEBAR,
    SUBTLE,
    SURFACE,
    TEXT,
    USER,
)
from cad_llm.app.viewer import CadViewerPanel
from cad_llm.config import settings
from cad_llm.inference.generate import CadGenerator
from cad_llm.tools.workspace import create_chat, create_project, list_projects
from cad_llm.tools.workspace.project import ChatLayout, ProjectLayout


class TerminalText(ctk.CTkTextbox):
    """Monospace transcript with tag colors."""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(
            master,
            font=FONT_MONO,
            fg_color=BG,
            text_color=TEXT,
            wrap="word",
            activate_scrollbars=True,
            **kwargs,
        )
        self._tag_colors()

    def _tag_colors(self) -> None:
        self.tag_config("user", foreground=USER)
        self.tag_config("agent", foreground=AGENT)
        self.tag_config("tool_out", foreground=ACCENT)
        self.tag_config("tool_in", foreground=OK)
        self.tag_config("error", foreground=ERROR)
        self.tag_config("setup", foreground=MUTED)
        self.tag_config("nudge", foreground=MUTED)
        self.tag_config("done", foreground=TEXT)
        self.tag_config("muted", foreground=MUTED)
        self.tag_config("system", foreground=MUTED)

    def append_line(self, text: str, tag: str = "muted") -> None:
        if not text:
            return
        self.configure(state="normal")
        self.insert("end", text + "\n", tag)
        self.see("end")
        self.configure(state="disabled")


class CadDesktopApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("cad-llm")
        self.geometry("1280x780")
        self.minsize(960, 600)
        self.configure(fg_color=BG)

        self._projects_root = settings.resolve(settings.projects_dir)
        self._project: ProjectLayout | None = None
        self._chat: ChatLayout | None = None
        self._session: AgentSession | None = None
        self._preloaded: CadGenerator | None = None
        self._busy = False
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()

        self._build_layout()
        self._refresh_projects()
        threading.Thread(target=self._preload_model, daemon=True).start()
        self.after(100, self._poll_events)

    def _preload_model(self) -> None:
        try:
            gen = CadGenerator()
            gen.load()
            self._events.put(("model_ready", gen))
        except Exception as exc:  # noqa: BLE001
            self._events.put(("model_error", str(exc)))

    def _build_layout(self) -> None:
        root = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        root.pack(fill="both", expand=True)

        # Sidebar
        sidebar = ctk.CTkFrame(root, width=220, fg_color=SIDEBAR, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand = ctk.CTkFrame(sidebar, fg_color=SIDEBAR, corner_radius=0, height=44)
        brand.pack(fill="x", padx=18, pady=(18, 8))
        brand.pack_propagate(False)
        ctk.CTkLabel(
            brand,
            text="cad-llm",
            font=("SF Pro Text", 14, "bold"),
            text_color=TEXT,
            anchor="w",
        ).pack(side="left")

        ctk.CTkLabel(
            sidebar,
            text="projects",
            font=FONT_LABEL,
            text_color=MUTED,
            anchor="w",
        ).pack(fill="x", padx=18, pady=(8, 4))

        self._project_list = ctk.CTkScrollableFrame(
            sidebar, fg_color=SIDEBAR, corner_radius=0
        )
        self._project_list.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        ctk.CTkButton(
            sidebar,
            text="+  new project",
            font=FONT_UI,
            fg_color="transparent",
            hover_color=BORDER,
            text_color=MUTED,
            anchor="w",
            height=30,
            command=self._new_project_dialog,
        ).pack(fill="x", padx=10, pady=(0, 14))

        # Vertical divider
        ctk.CTkFrame(root, width=1, fg_color=DIVIDER, corner_radius=0).pack(
            side="left", fill="y"
        )

        # Center: terminal
        center = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        center.pack(side="left", fill="both", expand=True)

        header = ctk.CTkFrame(center, fg_color=BG, height=44, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        self._header_label = ctk.CTkLabel(
            header,
            text="select a project to begin",
            font=FONT_UI,
            text_color=MUTED,
            anchor="w",
        )
        self._header_label.pack(side="left", fill="both", padx=20, pady=10)

        ctk.CTkFrame(center, height=1, fg_color=DIVIDER, corner_radius=0).pack(fill="x")

        transcript_wrap = ctk.CTkFrame(center, fg_color=BG, corner_radius=0)
        transcript_wrap.pack(fill="both", expand=True)
        self._transcript = TerminalText(transcript_wrap, corner_radius=0)
        self._transcript.pack(fill="both", expand=True, padx=12, pady=(8, 8))

        ctk.CTkFrame(center, height=1, fg_color=DIVIDER, corner_radius=0).pack(fill="x")

        input_row = ctk.CTkFrame(center, fg_color=BG, corner_radius=0)
        input_row.pack(fill="x", padx=12, pady=12)

        prompt_box = ctk.CTkFrame(
            input_row,
            fg_color=SURFACE,
            corner_radius=8,
            border_width=1,
            border_color=BORDER,
        )
        prompt_box.pack(fill="x")

        ctk.CTkLabel(
            prompt_box,
            text="›",
            font=FONT_MONO,
            text_color=ACCENT,
            width=24,
        ).pack(side="left", padx=(10, 0), pady=8)

        self._prompt = ctk.CTkEntry(
            prompt_box,
            placeholder_text="ask cad-llm…",
            font=FONT_MONO,
            fg_color=SURFACE,
            border_width=0,
            text_color=TEXT,
            placeholder_text_color=SUBTLE,
        )
        self._prompt.pack(side="left", fill="x", expand=True, padx=(4, 8), pady=8)
        self._prompt.bind("<Return>", self._on_submit)

        self._send_btn = ctk.CTkButton(
            prompt_box,
            text="send  ⏎",
            width=80,
            height=28,
            corner_radius=6,
            fg_color=BORDER,
            hover_color=SUBTLE,
            text_color=TEXT,
            font=FONT_LABEL,
            command=self._on_submit,
        )
        self._send_btn.pack(side="right", padx=8, pady=8)

        # Vertical divider before viewer
        ctk.CTkFrame(root, width=1, fg_color=DIVIDER, corner_radius=0).pack(
            side="right", fill="y"
        )

        # Right: CAD viewer
        self._viewer = CadViewerPanel(root)
        self._viewer.pack(side="right", fill="both")

    def _refresh_projects(self) -> None:
        for child in self._project_list.winfo_children():
            child.destroy()

        projects = list_projects(self._projects_root)
        if not projects:
            ctk.CTkLabel(
                self._project_list,
                text="No projects yet",
                font=FONT_UI,
                text_color=MUTED,
            ).pack(anchor="w", padx=8, pady=4)
            return

        for project in projects:
            meta_path = project.meta_dir / "project.json"
            name = project.project_id
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("name") or project.project_id
            btn = ctk.CTkButton(
                self._project_list,
                text=name,
                anchor="w",
                font=FONT_UI,
                fg_color="transparent",
                hover_color=BORDER,
                text_color=TEXT,
                command=lambda p=project: self._select_project(p),
            )
            btn.pack(fill="x", pady=2)

    def _select_project(self, project: ProjectLayout) -> None:
        if self._busy:
            return
        self._project = project
        self._chat = create_chat(project, title="Desktop session")
        self._session = None
        meta_path = project.meta_dir / "project.json"
        name = project.project_id
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = meta.get("name") or project.project_id
        self._header_label.configure(
            text=f"{name}   ·   {project.project_id}",
            text_color=TEXT,
        )
        self._transcript.configure(state="normal")
        self._transcript.delete("1.0", "end")
        self._transcript.configure(state="disabled")
        self._transcript.append_line(f"project  {project.root}", "system")
        if self._preloaded is not None:
            self._transcript.append_line("Model ready.", "system")
        else:
            self._transcript.append_line("Loading model…", "system")
        self._viewer.show_mesh_path(latest_mesh(project.outputs_dir))

    def _new_project_dialog(self) -> None:
        dialog = ctk.CTkInputDialog(text="Project name:", title="New project")
        name = dialog.get_input()
        if not name or not name.strip():
            return
        slug = name.strip().lower().replace(" ", "-")[:32]
        project = create_project(self._projects_root, name=name.strip(), project_id=slug)
        self._refresh_projects()
        self._select_project(project)

    def _on_submit(self, _event: object = None) -> None:
        if self._busy or self._project is None or self._chat is None:
            return
        text = self._prompt.get().strip()
        if not text:
            return
        self._prompt.delete(0, "end")
        self._transcript.append_line(f"you › {text}", "user")
        self._set_busy(True)
        threading.Thread(target=self._run_agent, args=(text,), daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self._send_btn.configure(state=state)
        self._prompt.configure(state=state)

    def _run_agent(self, prompt: str) -> None:
        assert self._project is not None
        assert self._chat is not None

        def on_step(step: AgentStep) -> None:
            self._events.put(("step", step))

        try:
            if self._session is None:
                if self._preloaded is None:
                    self._events.put(("status", "Loading model…"))
                self._session = AgentSession.create(
                    self._project,
                    self._chat,
                    generator=self._preloaded,
                )
                self._preloaded = None
                self._events.put(("status", "Ready."))

            result = self._session.run_turn(prompt, on_step=on_step)
            mesh = latest_mesh(self._project.outputs_dir)
            self._events.put(("mesh", mesh))
        except Exception as exc:  # noqa: BLE001
            self._events.put(("error", str(exc)))
        finally:
            self._events.put(("idle", None))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self._transcript.append_line(str(payload), "system")
            elif kind == "model_ready":
                self._preloaded = payload
                self._transcript.append_line("Model ready.", "system")
            elif kind == "model_error":
                self._transcript.append_line(f"Model load failed: {payload}", "error")
            elif kind == "step":
                step = payload
                if step.kind == "assistant":
                    text = format_assistant_reply(step.content)
                    if text:
                        self._transcript.append_line(f"agent › {text}", "done")
                else:
                    tag, line = format_step(step)
                    self._transcript.append_line(line, tag)
            elif kind == "mesh":
                self._viewer.show_mesh_path(payload)
            elif kind == "error":
                self._transcript.append_line(f"error  {payload}", "error")
            elif kind == "idle":
                self._set_busy(False)
        self.after(100, self._poll_events)


def run() -> None:
    app = CadDesktopApp()
    app.mainloop()
