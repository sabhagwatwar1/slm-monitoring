
import json
import time
import requests

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest,
    CONTENT_TYPE_LATEST
)

app = FastAPI(title="SLM Metrics Exporter")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"

# Request metrics
REQUESTS = Counter(
    "slm_requests_total",
    "Total inference requests"
)

ERRORS = Counter(
    "slm_request_errors_total",
    "Failed inference requests"
)

LATENCY = Histogram(
    "slm_request_latency_seconds",
    "End-to-end inference latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)
)

TTFT = Histogram(
    "slm_ttft_seconds",
    "Time to first generated text",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)

TPOT = Histogram(
    "slm_tpot_seconds",
    "Average generation time per output token",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
)

PROMPT_TOKENS = Counter(
    "slm_prompt_tokens_total",
    "Total prompt tokens processed"
)

COMPLETION_TOKENS = Counter(
    "slm_completion_tokens_total",
    "Total completion tokens generated"
)

PROMPT_TPS = Gauge(
    "slm_prompt_tokens_per_second",
    "Prompt processing throughput"
)

COMPLETION_TPS = Gauge(
    "slm_completion_tokens_per_second",
    "Completion generation throughput"
)

ACTIVE = Gauge(
    "slm_active_requests",
    "Currently active inference requests"
)

QUEUE = Gauge(
    "slm_queue_depth",
    "Requests waiting for an inference slot"
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SLM Metrics Exporter",
        "model": MODEL,
        "endpoints": {
            "metrics": "/metrics",
            "generate": "POST /generate",
            "docs": "/docs"
        }
    }

@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/generate")
def generate(payload: dict):
    prompt = payload.get("prompt")

    if not prompt:
        raise HTTPException(400, "Prompt is required")

    start = time.perf_counter()
    first_token_time = None

    result = {}

    REQUESTS.inc()
    ACTIVE.inc()

    try:
        with requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=180
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if chunk.get("response"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                if chunk.get("done"):
                    result = chunk

        end = time.perf_counter()

        LATENCY.observe(end - start)

        if first_token_time is not None:
            TTFT.observe(first_token_time - start)

        prompt_count = result.get("prompt_eval_count", 0)
        output_count = result.get("eval_count", 0)

        prompt_duration = (
            result.get("prompt_eval_duration", 0) / 1e9
        )

        output_duration = (
            result.get("eval_duration", 0) / 1e9
        )

        PROMPT_TOKENS.inc(prompt_count)
        COMPLETION_TOKENS.inc(output_count)

        if prompt_duration > 0:
            PROMPT_TPS.set(prompt_count / prompt_duration)

        if output_duration > 0 and output_count > 0:
            COMPLETION_TPS.set(output_count / output_duration)
            TPOT.observe(output_duration / output_count)

        return {
            "model": MODEL,
            "response": "Generation completed",
            "latency_seconds": round(end - start, 4),
            "ttft_seconds": (
                round(first_token_time - start, 4)
                if first_token_time else None
            ),
            "prompt_tokens": prompt_count,
            "completion_tokens": output_count
        }

    except requests.RequestException as exc:
        ERRORS.inc()
        raise HTTPException(502, str(exc))

    finally:
        ACTIVE.dec()
