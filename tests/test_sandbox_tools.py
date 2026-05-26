from pathlib import Path

import pytest

from cad_llm.tools.sandbox import get_sandbox_tools, run_python


def test_get_sandbox_tools_exports_run_python_sandbox() -> None:
    tools = get_sandbox_tools()
    assert len(tools) == 1
    assert tools[0].name == "run_python_sandbox"


def test_run_python_valid_script(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text('print("sandbox ok")\n')

    result = run_python(tmp_path, entrypoint="hello.py")

    assert "exit_code=0" in result
    assert "stdout:\nsandbox ok" in result


def test_run_python_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text('print("nope")\n')

    with pytest.raises(ValueError, match="Path escapes project root"):
        run_python(tmp_path, entrypoint=str(outside))


def test_run_python_missing_file_message(tmp_path: Path) -> None:
    result = run_python(tmp_path, entrypoint="missing.py")

    assert result == f"error: entrypoint is not a file: {tmp_path / 'missing.py'}"


def test_run_python_timeout(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(5)\n")

    result = run_python(tmp_path, entrypoint="slow.py", timeout=1)

    assert result == "error: timed out after 1s"
