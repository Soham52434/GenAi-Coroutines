# GenAI Coroutines

**High-performance async batch processing for Datalab Marker API (OCR) and OpenAI Responses API.**

Built in Rust with Python bindings, `genai-coroutines` eliminates GIL bottlenecks and processes hundreds of concurrent API requests with production-grade reliability — smart retries, precise rate limiting, structured output parsing, and cost tracking.

## Features

- **Concurrent Processing** — Semaphore-based concurrency control for both APIs.
- **Smart Retry Logic** — Exponential backoff with jitter. Auto-retries on 429/5xx; fails fast on 400/401.
- **Zero GIL** — Native Rust async runtime bypasses Python's Global Interpreter Lock.
- **Structured Output** — JSON Schema enforcement (OpenAI) and page-level structured extraction (Datalab).
- **Cost & Usage Tracking** — Token-level usage (OpenAI) and cost breakdown in cents (Datalab) returned with every result.
- **Parsing Helpers** — Built-in `parse_responses()` and `parse_documents()` to extract clean text/HTML.
- **Order Preservation** — Results always match input order.

---

## Quick Start

```python
import asyncio
from genai_coroutines import (
    DocumentConfig, DocumentProcessor, parse_documents,
    ResponsesRequest, ResponsesProcessor, parse_responses,
)
```

---

## Datalab OCR — `DocumentConfig` & `DocumentProcessor`

Process PDFs and images into structured text, HTML, or Markdown using the Datalab Marker API.

### All Parameters — `DocumentConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | **Required** | Your Datalab API key. |
| `api_url` | `str` | `"https://www.datalab.to/api/v1/marker"` | API endpoint URL. Override for custom/self-hosted endpoints. |
| `output_format` | `str` | `"json"` | `"json"` · `"html"` · `"markdown"` · `"chunks"` |
| `mode` | `str` | `"accurate"` | `"fast"` · `"balanced"` · `"accurate"` |
| **Concurrency & Retry** ||||
| `max_concurrent_requests` | `int` | `10` | Maximum parallel API calls. Controls semaphore permits. |
| `poll_interval_secs` | `int` | `2` | Seconds between status polling requests. |
| `max_poll_attempts` | `int` | `60` | Max polling attempts before timeout. |
| `max_retries` | `int` | `5` | Retry attempts on rate-limit/server errors. |
| `base_retry_delay_secs` | `int` | `5` | Base delay for exponential backoff (seconds). |
| `jitter_percent` | `int` | `200` | Random jitter range (±%) applied to backoff. |
| **Structured Extraction** ||||
| `page_schema` | `str` (JSON) | `None` | JSON schema string for structured data extraction per page. See [Structured Extraction](#structured-extraction-datalab) below. |
| **Page Control** ||||
| `paginate` | `bool` | `False` | Return output separated by page. |
| `page_range` | `str` | `None` | Specific pages to process (e.g., `"0-5"`, `"0,2,4"`). |
| `max_pages` | `int` | `None` | Maximum number of pages to process. |
| `disable_image_extraction` | `bool` | `False` | Skip image extraction from documents. |
| **Advanced** ||||
| `extras` | `str` (JSON) | `None` | Additional API parameters as a JSON string. |
| `webhook_url` | `str` | `None` | URL to receive webhook callback on completion. |

### Structured Extraction (Datalab)

Use `page_schema` to extract structured fields from each page. Pass a **JSON schema string** describing the fields you want:

```python
import json

schema = json.dumps({
    "type": "object",
    "properties": {
        "patient_name": {"type": "string", "description": "Full name of the patient"},
        "diagnosis": {"type": "string", "description": "Primary diagnosis"},
        "date": {"type": "string", "description": "Date of visit (YYYY-MM-DD)"}
    },
    "required": ["patient_name", "diagnosis"]
})

config = DocumentConfig(
    api_key="YOUR_DATALAB_KEY",
    output_format="json",
    mode="accurate",
    page_schema=schema  # Structured extraction
)
```

> **Note**: The `page_schema` value is validated as valid JSON at initialization time. If the string is not valid JSON, a `ValueError` is raised immediately.

### Usage Example

```python
import asyncio
from genai_coroutines import DocumentConfig, DocumentProcessor, parse_documents

async def main():
    config = DocumentConfig(
        api_key="YOUR_DATALAB_KEY",
        mode="accurate",
        max_concurrent_requests=10,
        page_range="0-5",        # Only process first 6 pages
        max_pages=10              # Safety cap
    )
    processor = DocumentProcessor(config)

    # Load documents
    files = ["report1.pdf", "report2.pdf", "scan.png"]
    batch = []
    for f in files:
        with open(f, "rb") as fh:
            batch.append(fh.read())

    # Process
    results = await processor.process_multiparts(batch)

    # Parse consolidated HTML from each document
    html_list = parse_documents(results)
    for i, html in enumerate(html_list):
        print(f"Doc {i}: {len(html)} chars of HTML")

    # Access cost breakdown (native Python dict)
    for r in results:
        if r["success"] and r.get("cost_breakdown"):
            cost = r["cost_breakdown"]  # Already a dict, no json.loads needed
            print(f"Cost: {cost['final_cost_cents']} cents")

asyncio.run(main())
```

### OCR Output Structure

Each item in the returned list:

```python
{
    "index": 0,                      # Matches input order
    "success": True,
    "json_response": "{...}",        # Raw JSON string from Datalab API
    "cost_breakdown": {              # Cost tracking (when available)
        "final_cost_cents": 15,
        "list_cost_cents": 15
    },
    "processing_time_secs": 4.5,
    "error": None                    # Error message if success=False
}
```

---

## OpenAI Responses API — `ResponsesRequest` & `ResponsesProcessor`

Batch process chat completions with structured output, reasoning models, tools, and multi-turn conversations.

### All Parameters — `ResponsesRequest`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | **Required** | Your OpenAI API key. |
| `system_prompt` | `str` | **Required** | System instructions for the model. |
| `user_prompts` | `list[str]` | **Required** | List of user prompts to process as a batch. |
| `model` | `str` | **Required** | Model ID: `"gpt-4o"`, `"gpt-4o-mini"`, `"o3-mini"`, etc. |
| `response_format` | `dict` | **Required** | Output format. See [Structured Output](#structured-output-openai) below. |
| `timeout_secs` | `int` | `60` | Per-request timeout in seconds. |
| **Concurrency & Retry** ||||
| `max_concurrent_requests` | `int` | `10` | Maximum parallel API calls. Controls semaphore permits. |
| `max_retries` | `int` | `5` | Retry attempts on rate-limit/server errors. |
| `retry_delay_min_ms` | `int` | `1000` | Minimum backoff delay in milliseconds. |
| `retry_delay_max_ms` | `int` | `60000` | Maximum backoff delay in milliseconds. |
| **Sampling** ||||
| `temperature` | `float` | `None` | Sampling temperature (0.0–2.0). |
| `top_p` | `float` | `None` | Nucleus sampling threshold. |
| `max_output_tokens` | `int` | `None` | Maximum tokens in the response. |
| **Reasoning (o-series models)** ||||
| `reasoning_effort` | `str` | `None` | `"low"` · `"medium"` · `"high"` — Controls thinking depth for o3-mini, o1, etc. |
| `reasoning_summary` | `str` | `None` | `"auto"` · `"concise"` · `"detailed"` — Controls reasoning summary output. |
| **Tools & Function Calling** ||||
| `tools` | `list[dict]` | `None` | Tool/function definitions for function calling. |
| `tool_choice` | `dict` | `None` | Tool selection strategy (`"auto"`, `"required"`, or specific tool). |
| `parallel_tool_calls` | `bool` | `None` | Allow model to call multiple tools in parallel. |
| **Multi-Turn** ||||
| `previous_response_id` | `str` | `None` | ID of a previous response to continue a conversation. |
| `include` | `list[str]` | `None` | Additional data to include (e.g., `["file_search_call.results"]`). |
| **Other** ||||
| `store` | `bool` | `None` | Whether to store the response for later retrieval. |
| `truncation` | `str` | `None` | Truncation strategy for context window overflow. |
| `metadata` | `dict` | `None` | Custom metadata to attach to the request. |
| `service_tier` | `str` | `None` | Service tier (`"auto"`, `"default"`). |
| `stream` | `bool` | `None` | Enable streaming (advanced). |

### Structured Output (OpenAI)

The `response_format` parameter controls how the model formats its output. Three modes are supported:

#### 1. JSON Schema (Strict)

Force the model to output valid JSON matching your exact schema:

```python
request = ResponsesRequest(
    api_key="YOUR_KEY",
    model="gpt-4o-mini",
    system_prompt="Extract patient info from the text.",
    user_prompts=["Patient John Doe, age 45, diagnosed with hypertension on 2024-01-15."],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "patient_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "diagnosis": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["name", "age", "diagnosis", "date"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
```

**Output** (guaranteed valid JSON matching schema):
```json
{"name": "John Doe", "age": 45, "diagnosis": "hypertension", "date": "2024-01-15"}
```

#### 2. JSON Object (Flexible)

Force valid JSON output without a specific schema:

```python
response_format={"type": "json_object"}
```

#### 3. Plain Text

No format enforcement:

```python
response_format={"type": "text"}
```

### Reasoning Models (o3-mini, o1)

For reasoning models, control the depth of thinking and summary:

```python
request = ResponsesRequest(
    api_key="YOUR_KEY",
    model="o3-mini",
    system_prompt="Solve this step by step.",
    user_prompts=["What is 1234 * 5678?"],
    response_format={"type": "text"},
    reasoning_effort="high",         # low | medium | high
    reasoning_summary="detailed"     # auto | concise | detailed
)
```

### Function Calling / Tools

Define tools the model can call:

```python
request = ResponsesRequest(
    api_key="YOUR_KEY",
    model="gpt-4o",
    system_prompt="You are a helpful assistant with access to tools.",
    user_prompts=["What's the weather in San Francisco?"],
    response_format={"type": "text"},
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)
```

### Multi-Turn Conversations

Continue a conversation by referencing a previous response:

```python
# First turn
results1 = await processor.process_batch(request1)
response_id = json.loads(results1["results"][0]["raw_response"])["id"]

# Second turn
request2 = ResponsesRequest(
    api_key="YOUR_KEY",
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    user_prompts=["Can you elaborate on that?"],
    response_format={"type": "text"},
    previous_response_id=response_id  # Continue the conversation
)
results2 = await processor.process_batch(request2)
```

### Usage Example

```python
import asyncio, json
from genai_coroutines import ResponsesRequest, ResponsesProcessor, parse_responses

async def main():
    request = ResponsesRequest(
        api_key="YOUR_OPENAI_KEY",
        model="gpt-4o-mini",
        system_prompt="You are an expert data extractor.",
        user_prompts=[
            "Extract: John Doe, 45, hypertension",
            "Extract: Jane Smith, 32, diabetes",
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "patient",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "condition": {"type": "string"}
                    },
                    "required": ["name", "age", "condition"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        max_concurrent_requests=20,
        max_retries=3
    )

    processor = ResponsesProcessor()
    results = await processor.process_batch(request)

    # Parse clean text
    texts = parse_responses(results)
    for text in texts:
        data = json.loads(text)
        print(f"{data['name']} — {data['condition']}")

    # Access token usage (native Python dict — no json.loads needed)
    for r in results["results"]:
        if r["success"] and r.get("usage"):
            usage = r["usage"]  # Already a dict
            print(f"Tokens: {usage['total_tokens']}")

asyncio.run(main())
```

### OpenAI Output Structure

```python
{
    "total_success": 2,
    "total_errors": 0,
    "results": [
        {
            "success": True,
            "raw_response": "{...}",      # Full OpenAI JSON response (string)
            "usage": {                     # Token usage (native Python dict)
                "input_tokens": 42,
                "output_tokens": 18,
                "total_tokens": 60,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}
            }
        },
        {
            "success": False,
            "error": "401 Unauthorized: Invalid API key",
            "error_type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
            "is_retriable": False,
            "attempts": 1
        }
    ]
}
```

---

## Helper Functions

### `parse_responses(results) → list[str]`

Extracts clean assistant message text from OpenAI batch results.

```python
from genai_coroutines import parse_responses

texts = parse_responses(results)
# ["John Doe, 45, hypertension...", "Jane Smith, 32, diabetes..."]
```

- **Input**: The dict returned by `ResponsesProcessor.process_batch()`.
- **Output**: List of strings — **one per input prompt** (same length as `user_prompts`).
- **Behavior**:
  - Aggregates all text from `output[].content[].text` for each response.
  - If the model called tools instead of producing text, returns the **full raw JSON** so you can inspect tool calls.
  - Returns `""` for failed prompts.

### `parse_documents(results) → list[str]`

Extracts content from all OCR results, auto-detecting the output format.

```python
from genai_coroutines import parse_documents

contents = parse_documents(results)
# ["<p>Page 1 content</p>\n<p>Page 2 content</p>", ...]
```

- **Input**: The list returned by `DocumentProcessor.process_multiparts()`.
- **Output**: List of strings — **one per input document** (same length as input batch).
- **Format handling**:
  - `output_format="json"`: Extracts and concatenates `json.children[].html`.
  - `output_format="html"`: Returns raw HTML string.
  - `output_format="markdown"`: Returns markdown string.
  - `paginate=True`: Concatenates content from paginated output.
  - `page_schema` (structured extraction): Returns the full JSON string.
  - Fallback: Returns raw JSON string so nothing is ever lost.
- Returns `""` for failed documents.

---

## Error Handling

Errors are classified internally:

| Category | HTTP Codes | Behavior |
|---|---|---|
| **Retriable** | 429, 500, 502, 503, 504, timeouts | Auto-retry with exponential backoff + jitter |
| **Fatal** | 400, 401, 403, 404 | Fail immediately, no retry |

- Failed items have `"success": False` with an `"error"` message.
- The batch **never crashes** due to individual failures — all other items continue processing.
- For OpenAI errors, `is_retriable`, `error_type`, `param`, `code`, and `attempts` are included.

---

## Logging

All Rust-level logs are bridged to Python's `logging` module under the `genai_coroutines` logger:

```python
import logging

# See all logs
logging.basicConfig(level=logging.INFO)

# Per-module control
logging.getLogger("genai_coroutines.ocr").setLevel(logging.DEBUG)
logging.getLogger("genai_coroutines.responses").setLevel(logging.WARNING)
```

Sample log output:
```
INFO  [chandra] batch_start | files=10 concurrency=10
WARN  [chandra] rate_limit | index=3 attempt=2/5
INFO  [chandra] task_success | index=3 time=12.50s attempts=2
INFO  [chandra] batch_done | ok=10/10 errors=0
```

---

## Performance Tuning

| Scenario | `max_concurrent_requests` | Retry Config |
|---|---|---|
| **Small batch (<50)** | 5–10 | Defaults |
| **High volume (1k+)** | 30–50 | Increase `retry_delay_max_ms` |
| **Rate-limited API** | 3–5 | Increase `jitter_percent` and `base_retry_delay_secs` |
| **Reasoning models** | 5–10 | Increase `timeout_secs` to 120+ |

> `max_concurrent_requests` controls the semaphore. Even with high concurrency, the retry logic will back off automatically on rate limits.
