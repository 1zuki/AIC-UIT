import json
from pathlib import Path
from textwrap import dedent

import nbformat

BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "zero_shot_order.py"
NOTEBOOK_PATH = BASE_DIR / "solution.ipynb"

source = dedent(SOURCE_PATH.read_text(encoding="utf-8"))
notebook = nbformat.v4.new_notebook()
notebook.cells = [
    nbformat.v4.new_code_cell(
        source=source,
        metadata={},
    )
]
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
    },
}
NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
