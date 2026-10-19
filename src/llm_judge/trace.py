"""Durable traces and a content-addressed decisions cache. -- PI/OpenAI"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import httpx

TRANSIENT = {408, 429, 500, 502, 503, 504, 520, 524, 529}
FIRST_PAIR = ContextVar("first_judged_pair", default=False)


def digest(value):
    # PI/OpenAI: Question/option insertion order is an experimental condition, not cosmetic JSON.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def records(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        # PI/OpenAI: Preserve a torn final write before restoring the append-only boundary.
        with path.with_suffix(path.suffix + ".partial").open("ab") as stream:
            stream.write(data[end:])
        with path.open("r+b") as stream:
            stream.truncate(end)
        data = data[:end]
    return [json.loads(line) for line in data.splitlines()]


class Trace:
    def __init__(self, output: Path | str):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.run_id = uuid.uuid4().hex
        self.first_id = None
        self.label_count = 0
        self.decide = None
        self.served_path = output / "served_models.json"
        self.served = json.loads(self.served_path.read_text()) if self.served_path.exists() else {}
        self.first_keys = set()
        for name in ("events.jsonl", "samples.jsonl", "decisions_cache.jsonl"):
            records(output / name)
        with (output / "run.md").open("a") as stream:
            stream.write("\n# Judge run\n\nRead every message, schema, exposed reasoning, reply and order variant of the first pair before accepting scores.\n")

    def __enter__(self):
        self.status("running")
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.event("failed", error=repr(error))
            self.markdown("Failed operation", {"error": repr(error), "read": "events.jsonl"})
        self.status("failed" if error is not None else "success")
        return False

    def event(self, kind, **data):
        value = {"time": time.time(), "run_id": self.run_id, "kind": kind, **data}
        append(self.output / "events.jsonl", value)
        if data.get("key") in self.first_keys or FIRST_PAIR.get():
            self.markdown("First pair: raw event", value)

    def status(self, status, **data):
        value = {"run_id": self.run_id, "status": status, **data}
        pending = self.output / "status.json.tmp"
        pending.write_text(json.dumps(value, ensure_ascii=False))
        pending.replace(self.output / "status.json")

    def model(self, backend, served):
        if backend in self.served and self.served[backend] != served:
            raise ValueError(f"{backend}: mixed served models; use a new output/cache directory")
        self.served[backend] = served
        pending = self.served_path.with_suffix(".tmp")
        pending.write_text(json.dumps(self.served))
        pending.replace(self.served_path)

    def markdown(self, title, value):
        with (self.output / "run.md").open("a") as stream:
            stream.write(f"\n## {title}\n\n```json\n{json.dumps(value, ensure_ascii=False, indent=2)}\n```\n")
            stream.flush()
            os.fsync(stream.fileno())

    def sample(self, value):
        append(self.output / "samples.jsonl", value)
        if value["metadata"]["pair"]["id"] == self.first_id:
            self.markdown("First pair: complete sample (including cache hits and failures)", value)


class Decisions:
    def __init__(self, trace, validate, *, parallel=4, transport=None, sleep=asyncio.sleep):
        if parallel <= 0:
            raise ValueError("parallel must be positive")
        self.trace, self.validate = trace, validate
        self.parallel = asyncio.Semaphore(parallel)
        self.transport, self.sleep = transport, sleep
        self.cache = {}
        for row in records(trace.output / "decisions_cache.jsonl"):
            if row["key"] != digest(row["request"]):
                raise ValueError("cache key mismatch")
            validate(row["body"], row["request"])
            self.cache[row["key"]] = row
        self.pending = {}

    async def __call__(self, request, *, refresh=False):
        body, charge = await self.charged(request, refresh=refresh)
        return body

    async def charged(self, request, *, refresh=False):
        """Return the untouched provider body and this caller's new charge; cache/dedup cost zero. -- PI/OpenAI"""
        key = digest(request)
        if refresh:
            body = await self._fetch(key, request)
            return body, body["usage"]["cost"]
        if key in self.cache:
            row = self.cache[key]
            self.trace.model("jev", row["body"]["model"])
            self.trace.event("decision_cache_hit", **row)
            return row["body"], 0.0
        owner = key not in self.pending
        if owner:
            self.pending[key] = asyncio.create_task(self._fetch(key, request))
        # PI/OpenAI: Concurrent duplicates share a request, but only its initiating caller incurs the charge.
        body = await self.pending[key]
        return body, body["usage"]["cost"] if owner else 0.0

    async def _fetch(self, key, request):
        async with self.parallel:
            headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                       "HTTP-Referer": "https://github.com/wassname/llm-judge",
                       "X-OpenRouter-Title": "judge-example/decisions"}
            async with httpx.AsyncClient(headers=headers, transport=self.transport, timeout=180) as client:
                for attempt in range(6):
                    self.trace.event("decision_request", key=key, attempt=attempt, request=request)
                    try:
                        response = await client.post("https://openrouter.ai/api/alpha/decisions", json=request)
                    except httpx.TransportError as error:
                        self.trace.event("decision_transport_error", key=key, attempt=attempt, error=repr(error))
                        if attempt == 5:
                            raise
                    else:
                        self.trace.event("decision_response", key=key, attempt=attempt,
                                         status=response.status_code, response_text=response.text)
                        if response.status_code not in TRANSIENT or attempt == 5:
                            response.raise_for_status()
                            body = response.json()
                            self.validate(body, request)
                            self.trace.model("jev", body["model"])
                            row = {"key": key, "request": request, "body": body}
                            append(self.trace.output / "decisions_cache.jsonl", row)
                            self.cache[key] = row
                            return body
                    await self.sleep(min(30, 2 ** attempt) + random.random())
        raise AssertionError("unreachable retry exhaustion")
