# Development Setup

This project uses [uv](https://github.com/Astral-sh/uv#installation) for managing Python environments and dependencies. Please ensure you have it installed before proceeding.

Once uv is installed, you can set up your development environment by running:

```bash
uv sync
```

This will create a virtual environment and install all necessary dependencies as specified in the `pyproject.toml` file.

To run a file run :

```bash
uv run <file>
```

To check for linting formatting and type issues, run:

```bash
# Lint
uv run ruff check

# Format
uv run ruff format

# Type Check
uv run ty check
```
