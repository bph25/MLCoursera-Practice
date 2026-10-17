"""Validate blank practice notebooks without solving or executing their exercises."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    checked_cells = checked_links = cleared_cells = 0
    errors = []
    for item in manifest["assignments"] + manifest["labs"]:
        path = root / item["notebook"]
        nb = json.loads(path.read_text())
        assert nb["nbformat"] == 4 and isinstance(nb["cells"], list)
        tasks = 0
        for index, cell in enumerate(nb["cells"]):
            text = "".join(cell["source"])
            label = f"{item['id']} cell {index}"
            if cell["cell_type"] == "code":
                checked_cells += 1
                if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                    errors.append(f"{label}: saved output/count")
                else:
                    cleared_cells += 1
                # IPython magics are not Python statements; the rest must compile even when blank.
                cleaned = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("%", "!")))
                try:
                    tree = ast.parse(cleaned)
                except SyntaxError as error:
                    errors.append(f"{label}: {error}")
                    continue
                if "exercise" in cell.get("metadata", {}).get("tags", []):
                    tasks += 1
                    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                    if functions:
                        for node in functions:
                            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
                            if len(body) != 1 or not isinstance(body[0], ast.Raise):
                                errors.append(f"{label}: implementation left in {node.name}")
                            elif not isinstance(body[0].exc, ast.Call) or not isinstance(body[0].exc.func, ast.Name) or body[0].exc.func.id != "NotImplementedError":
                                errors.append(f"{label}: missing TODO exception")
                    elif "NotImplementedError" not in text:
                        errors.append(f"{label}: missing model/open exercise placeholder")
            if cell["cell_type"] == "markdown":
                if re.search(r"</?details\b|Click for (?:more )?hints|Click for code|Last Resort", text, re.I):
                    errors.append(f"{label}: hidden solution/hint remains")
                # HTML images and Markdown links/images must be local or explicitly external.
                links = re.findall(r'''(?:src|href)=["']([^"']+)["']''', text)
                links += re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", text)
                for link in links:
                    if re.match(r"(?:[a-z]+:|#|//)", link, re.I):
                        continue
                    target = path.parent / unquote(link.split("#")[0].split("?")[0])
                    checked_links += 1
                    if not target.exists():
                        errors.append(f"{label}: missing link {link}")
        if tasks != item["exercise_cells"]:
            errors.append(f"{item['id']}: task count differs from manifest")
        for asset in item.get("assets", []):
            if not (path.parent / asset).is_file():
                errors.append(f"{item['id']}: missing support file {asset}")
    notebooks = list(root.rglob("*.ipynb"))
    if len(notebooks) != 17:
        errors.append(f"Expected 17 notebooks, found {len(notebooks)}")
    for path in root.rglob("*"):
        if any(name in path.parts for name in ("archive", ".ipynb_checkpoints", "__pycache__")):
            errors.append(f"Unexpected archive/cache: {path.relative_to(root)}")
        if path.suffix in {".h5", ".pyc"}:
            errors.append(f"Unexpected trained model/bytecode: {path.relative_to(root)}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text())
            except SyntaxError as error:
                errors.append(f"Invalid helper: {path.relative_to(root)}: {error}")
    # Check original files only if the source repo is accessible; the pack otherwise stands alone.
    source = Path(manifest["source"])
    hashes_match = None
    if source.is_dir():
        hashes_match = all(hashlib.sha256((source / p).read_bytes()).hexdigest() == digest
                           for p, digest in manifest["original_notebook_hashes"].items())
        if not hashes_match:
            errors.append("Source notebooks changed since generation")
    result = dict(notebooks=len(notebooks), code_cells=checked_cells, cleared_code_cells=cleared_cells,
                  exercise_cells=sum(x["exercise_cells"] for x in manifest["assignments"] + manifest["labs"]),
                  local_links_checked=checked_links, original_notebooks_unchanged=hashes_match,
                  errors=errors, passed=not errors)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    result = verify(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve())
    sys.exit(0 if result["passed"] else 1)
