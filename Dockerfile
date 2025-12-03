FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked --no-install-project --no-editable

COPY . /app/

CMD ["/app/.venv/bin/python3", "/app/main.py"]