# interlock
Training a model to solve a jigsaw puzzle

## Setup
Install [uv](https://docs.astral.sh/uv/) once:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, from the repo root:
```bash
uv sync
```
This creates `.venv/` with the right Python and all dependencies. Run things with `uv run`, e.g. `uv run python scripts/preprocess.py` or `uv run pytest`. In VS Code, select `.venv/bin/python` as the interpreter.

Add a dependency with `uv add <package>` (or `uv add --dev <package>` for dev-only tools) and commit both `pyproject.toml` and `uv.lock`.

scripts/visualize.py shows what happens when interlock/data_collection/cutting.py runs. Run with `uv run python scripts/visualize.py --image path/to/photo.jpg`
