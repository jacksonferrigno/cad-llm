import pytest

from cad_llm.tools.skills.loader import list_skills, load_skill


def test_list_skills_includes_builtins() -> None:
    names = list_skills()
    assert "cad-generation" in names
    assert "cad-debug" in names


def test_load_skill_cad_generation() -> None:
    content = load_skill("cad-generation")
    assert "# CAD generation" in content
    assert "search_cadquery_docs" in content


def test_load_skill_cad_debug() -> None:
    content = load_skill("cad-debug")
    assert "# CAD debug" in content
    assert "read_file" in content


@pytest.mark.parametrize(
    "name",
    ["Bad Name!", "", "../etc/passwd", "UPPERCASE"],
)
def test_load_skill_invalid_name_raises(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid skill name"):
        load_skill(name)


def test_load_skill_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Unknown skill"):
        load_skill("nonexistent-skill")
