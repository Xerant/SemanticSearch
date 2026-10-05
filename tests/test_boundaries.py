"""Architectural rules, enforced by scanning imports:

1. The engine never imports domain code (domain knowledge arrives only via packs).
2. `core` imports only the standard library and itself, so every other layer can
   depend on it without pulling in third-party packages.
"""

import ast
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "semanticsearch"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module)
    return mods


def test_engine_never_imports_domains():
    offenders = {
        str(p.relative_to(PKG)): m
        for p in PKG.rglob("*.py")
        for m in imported_modules(p)
        if m == "domains" or m.startswith("domains.")
    }
    assert offenders == {}


def test_core_is_stdlib_only():
    offenders = {}
    for p in (PKG / "core").rglob("*.py"):
        for m in imported_modules(p):
            top = m.split(".")[0]
            if m.startswith("semanticsearch.core") or top == "__future__":
                continue
            if top not in sys.stdlib_module_names:
                offenders[str(p.relative_to(PKG))] = m
    assert offenders == {}
