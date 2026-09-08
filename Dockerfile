# The sleeve runs as its own container, next to the executor's but not inside it: a bug in
# message parsing or a model call that hangs must not be able to touch the loop that talks
# to IB.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentic_macro/ ./agentic_macro/
COPY run_bot.py .

# Worldviews live on a volume, not in the image. Losing this database would leave positions
# at the broker that no worldview claims — and the next netted book would close them.
VOLUME ["/app/db"]
ENV AGENTIC_DB_PATH=/app/db/worldviews.db

CMD ["python", "run_bot.py"]
