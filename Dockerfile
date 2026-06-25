FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 10001 agentctl

COPY pyproject.toml README.md ./
COPY agent_ctl ./agent_ctl

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[server,anthropic,openai]"

USER agentctl

EXPOSE 8400

CMD ["sh", "-c", ": \"${AGENT_CTL_API_TOKEN:?set AGENT_CTL_API_TOKEN}\"; exec agent-ctl --config /config/agent_ctl.yaml serve --host 0.0.0.0 --port 8400 --api-token \"$AGENT_CTL_API_TOKEN\""]
