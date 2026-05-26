from cad_llm.app.formatting import format_assistant_reply
from cad_llm.app.mesh import latest_mesh


def test_format_assistant_reply_strips_code_and_truncates() -> None:
    raw = (
        "Build succeeded.\n\n"
        "```python\nimport cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)\n```\n\n"
        "More details here."
    )
    out = format_assistant_reply(raw)
    assert "```" not in out
    assert "import cadquery" not in out
    assert "Build succeeded" in out


def test_latest_mesh_prefers_stl(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    step = outputs / "model.step"
    stl = outputs / "model.stl"
    step.write_text("step")
    stl.write_text("stl")
    # Make STL newer
    import os
    import time

    os.utime(step, (1, 1))
    os.utime(stl, (time.time(), time.time()))

    assert latest_mesh(outputs) == stl
