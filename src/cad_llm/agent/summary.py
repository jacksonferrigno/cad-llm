"""Human-readable one-line summaries for agent tool I/O."""

from __future__ import annotations


def summarize_tool_call(name: str, arguments: dict[str, object]) -> str:
    if name == "write_file":
        return f"write {arguments.get('path', '?')}"
    if name == "read_file":
        path = arguments.get("path", "?")
        if arguments.get("start_line") or arguments.get("end_line"):
            return f"read {path} (lines)"
        return f"read {path}"
    if name == "grep":
        pattern = str(arguments.get("pattern", "?"))
        if len(pattern) > 36:
            pattern = pattern[:33] + "..."
        target = arguments.get("path")
        if target:
            return f"grep {pattern!r} in {target}"
        return f"grep {pattern!r}"
    if name == "search_replace":
        return f"patch {arguments.get('path', '?')}"
    if name == "delete_file":
        return f"delete {arguments.get('path', '?')}"
    if name == "run_python_sandbox":
        return f"run {arguments.get('entrypoint', 'src/main.py')}"
    if name == "load_skill":
        return f"skill {arguments.get('name', '?')}"
    if name == "search_cadquery_docs":
        query = str(arguments.get("query", "?"))
        if len(query) > 48:
            query = query[:45] + "..."
        return f"docs {query!r}"
    return name


def summarize_tool_result(name: str, output: str) -> str:
    if name == "write_file":
        if output.startswith("error:"):
            return output.removeprefix("error: ").strip()[:100]
        return f"saved {output}"

    if name == "run_python_sandbox":
        first = output.splitlines()[0] if output else output
        if first.startswith("exit_code=0"):
            return "ok"
        for line in output.splitlines():
            stripped = line.strip()
            if "Error" in stripped or stripped.startswith("error:"):
                return stripped[:100]
        return first[:100]

    if name == "read_file":
        lines = output.count("\n") + (1 if output else 0)
        return f"{lines} lines"

    if name == "grep":
        matches = len([ln for ln in output.splitlines() if ln.strip()])
        return f"{matches} matches" if matches else "no matches"

    if name == "search_replace":
        return output

    if name == "search_cadquery_docs":
        hits = output.count("[1]")
        return f"{hits or 'no'} hits" if hits else "no hits"

    if name == "load_skill":
        return "loaded"

    if output.startswith("error:"):
        return output.removeprefix("error: ").strip()[:100]

    if len(output) > 100:
        return output[:97] + "..."
    return output or "ok"
