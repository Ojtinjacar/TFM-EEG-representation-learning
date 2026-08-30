"""Builds a notebook from a script, so that the notebook is a build artefact.

A notebook edited by hand is a notebook that cannot be reviewed in a diff, cannot be
regenerated once its outputs go stale, and records no trace of what produced it. Writing the
script instead keeps the source in plain Python: the cells are readable in a pull request,
the whole thing is rebuilt with one command, and a cell that stops working is fixed where it
was written rather than inside a JSON blob.

Only ``nbformat`` is needed, and it is already pinned in ``environment.yml`` next to
``nbconvert``, which is what executes the result.

Usage from a build script::

    nb = Notebook()
    nb.markdown("# Title", cell_id="s0-cover")
    nb.code("import numpy as np", cell_id="s0-setup")
    nb.write("notebook/name.ipynb")
"""

from __future__ import annotations

import os
import textwrap

import nbformat

KERNEL_NAME = "python3"
DISPLAY_NAME = "Python 3"


class NotebookError(RuntimeError):
    """Raised when a notebook cannot be built as requested."""


class Notebook:
    """Accumulates cells and writes them out as a notebook.

    Attributes are private on purpose: a half-written notebook should not be readable, and
    the only way in is through the two adders.
    """

    def __init__(self):
        self._cells = []
        self._ids = set()

    def markdown(self, source, cell_id=None):
        """Appends a markdown cell.

        Args:
            source (str): Cell body. Common leading whitespace is stripped, so it can be
                written as an indented triple-quoted string.
            cell_id (str | None): Stable identifier, by convention ``s<section>-<slug>``.

        Raises:
            NotebookError: If the identifier repeats one already used.
        """
        self._append(nbformat.v4.new_markdown_cell, source, cell_id)

    def code(self, source, cell_id=None):
        """Appends a code cell.

        Args:
            source (str): Cell body, dedented like in :meth:`markdown`.
            cell_id (str | None): Stable identifier.

        Raises:
            NotebookError: If the identifier repeats one already used.
        """
        self._append(nbformat.v4.new_code_cell, source, cell_id)

    def _append(self, factory, source, cell_id):
        """Builds one cell and records its identifier.

        Args:
            factory (callable): ``nbformat`` constructor for the cell kind.
            source (str): Cell body.
            cell_id (str | None): Stable identifier.

        Raises:
            NotebookError: If the identifier repeats one already used.
        """
        if cell_id is not None:
            if cell_id in self._ids:
                raise NotebookError(f"Cell id {cell_id!r} is used twice.")
            self._ids.add(cell_id)

        cell = factory(textwrap.dedent(source).strip())
        if cell_id is not None:
            cell["id"] = cell_id
        self._cells.append(cell)

    def write(self, path, kernel=KERNEL_NAME):
        """Validates the notebook and writes it to disk.

        Args:
            path (str): Destination ``.ipynb``.
            kernel (str): Kernel name recorded in the metadata.

        Returns:
            str: The path written.

        Raises:
            NotebookError: If no cell was added.
        """
        if not self._cells:
            raise NotebookError("A notebook with no cells is not worth writing.")

        notebook = nbformat.v4.new_notebook(cells=self._cells)
        notebook.metadata.kernelspec = {
            "display_name": DISPLAY_NAME, "language": "python", "name": kernel,
        }
        notebook.metadata.language_info = {"name": "python"}
        nbformat.validate(notebook)

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            nbformat.write(notebook, fh)

        n_code = sum(1 for c in self._cells if c["cell_type"] == "code")
        print(f"[INFO] {path}: {len(self._cells)} cells ({n_code} code, "
              f"{len(self._cells) - n_code} markdown)", flush=True)
        return path
