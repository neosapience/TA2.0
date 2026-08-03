import ast
from pathlib import Path


FORBIDDEN_IMPORTS = {"pymongo", "redis", "pytorch_lightning"}


def test_runtime_has_no_training_or_external_service_imports():
    root = Path(__file__).parents[1] / "talking_avatar"
    imported = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint(FORBIDDEN_IMPORTS)
