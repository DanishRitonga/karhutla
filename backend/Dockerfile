# Dockerfile backend Karhutla Dashboard.
# Kompatibel untuk deploy ke HuggingFace Spaces (Docker SDK, port 7860),
# atau ke platform lain (Railway/Render/VPS) dengan PORT env var apa saja.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	bash \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh /app/prepare.sh

ENV PORT=7860
EXPOSE 7860

CMD ["/app/start.sh"]
