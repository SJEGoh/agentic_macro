# The sleeve runs as its own container, next to the executor's but not inside it: a bug in
# message parsing or a model call that hangs must not be able to touch the loop that talks
# to IB.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# procps for the healthcheck's pgrep; build tools for chromadb's native deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentic_macro/ ./agentic_macro/
COPY run_bot.py .

# Worldviews and the news store live on a volume, not in the image.
VOLUME ["/app/db"]
ENV AGENTIC_DB_PATH=/app/db/worldviews.db \
    AGENTIC_MEMORY_PATH=/app/db/chroma

# On start: report the configuration, build the news store if it is empty, then poll.
CMD ["python", "run_bot.py"]
