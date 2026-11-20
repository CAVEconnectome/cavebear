"""Run the example scripts and write them to docs/tutorial as notebooks and Markdown.

Run this script again when an example changes:

    uv run --group docs python scripts/build_tutorial.py [--no-run] [name ...]

With no names, the script builds every example in ``EXAMPLES``. Each example is a
percent-format script (``# %%`` cells). The script runs it as a notebook, from the
``examples`` directory, and writes ``docs/tutorial/<name>.ipynb`` with the outputs and
``docs/tutorial/<name>.md`` for the site. With ``--no-run``, the script does not run
the examples. It writes the Markdown again from the notebooks in ``docs/tutorial``.

Each output goes in a ``<div class="nb-output">``, which
``docs/stylesheets/notebook.css`` styles.
"""

import base64
import re
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient
from nbconvert.preprocessors import CoalesceStreamsPreprocessor

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
OUT_DIR = ROOT / "docs" / "tutorial"
SOURCE_URL = "https://github.com/CAVEconnectome/cavebear/blob/main/examples"
EXAMPLES = ["demo", "partitionwise_join"]

# Admonitions to put at the top of a page, after the download links.
NOTES = {
    "partitionwise_join": """\
!!! danger "Work in progress"

    This API is a work in progress. We are still exploring how to balance an
    efficient query plan with an API that mirrors the lazy expressions of Polars.""",
}


def links(name: str) -> str:
    return (
        f"[Download the notebook]({name}.ipynb) · "
        f"[View the source script]({SOURCE_URL}/{name}.py)"
    )


def with_links(markdown: str, name: str) -> str:
    """Put the download links, and the note of the page, after its first heading."""
    header = ["", links(name)]
    if name in NOTES:
        header += ["", NOTES[name]]
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i + 1 : i + 1] = header
            break
    else:
        lines[:0] = [*header[1:], ""]
    return "\n".join(lines) + "\n"


def fence(text: str, lang: str) -> str:
    """A fenced code block, with a fence longer than any backtick run in ``text``."""
    ticks = "`" * max([3, *(len(run) + 1 for run in re.findall(r"`+", text))])
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


# Polars puts this style in each table. notebook.css styles the tables instead.
POLARS_STYLE = re.compile(r"<style>.*?</style>\s*", re.DOTALL)


def render_output(output: dict, name: str, key: str) -> str | None:
    """One output of a code cell as Markdown or HTML, or ``None`` to leave it out."""
    if output["output_type"] == "stream":
        return fence(output["text"], "text")
    if output["output_type"] == "error":
        return fence("\n".join(output["traceback"]), "text")
    data = output.get("data", {})
    if "text/html" in data:
        html = POLARS_STYLE.sub("", data["text/html"]).replace(' border="1"', "")
        return f'<div class="nb-html">{html}</div>'
    if "image/png" in data:
        image = OUT_DIR / f"{name}_files" / f"{key}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(base64.b64decode(data["image/png"]))
        return f"![]({image.parent.name}/{image.name})"
    if "text/plain" in data:
        return fence(data["text/plain"], "text")
    return None


def render(nb: nbformat.NotebookNode, name: str) -> str:
    """The notebook as a Markdown page."""
    blocks = []
    for i, cell in enumerate(nb.cells):
        source = cell.source.strip()
        if cell.cell_type == "markdown":
            if source:
                blocks.append(source)
            continue
        if source:
            blocks.append(fence(source, "python"))
        for j, output in enumerate(cell.get("outputs", [])):
            body = render_output(output, name, f"cell{i}_output{j}")
            if body is not None:
                blocks.append(f'<div class="nb-output" markdown>\n\n{body}\n\n</div>')
    return with_links("\n\n".join(blocks), name)


def build(name: str, run: bool = True) -> None:
    notebook = OUT_DIR / f"{name}.ipynb"
    if run:
        source = EXAMPLES_DIR / f"{name}.py"
        nb = jupytext.read(source)
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        print(f"running {source.relative_to(ROOT)}", flush=True)
        NotebookClient(
            nb,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(EXAMPLES_DIR)}},
        ).execute()
        # Join the prints of a cell into one output, as Jupyter shows them.
        nb, _ = CoalesceStreamsPreprocessor().preprocess(nb, {})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, notebook)
    else:
        nb = nbformat.read(notebook, as_version=4)
    (OUT_DIR / f"{name}.md").write_text(render(nb, name))
    print(f"wrote {(OUT_DIR / name).relative_to(ROOT)}.md", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    run = "--no-run" not in args
    for name in [a for a in args if a != "--no-run"] or EXAMPLES:
        build(name, run=run)
