ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/talking-avatar/.venv/bin:${PATH}"

WORKDIR /opt/talking-avatar
COPY . .
RUN uv venv --system-site-packages .venv \
    && uv pip install --python .venv/bin/python --no-deps -r requirements.lock

ENTRYPOINT ["python", "inference.py"]
