"""
genai_coroutines - High-performance async batch processing for GenAI APIs
"""
import json

# The Rust extension is compiled as genai_coroutines._internal (module-name = "genai_coroutines._internal").
# Submodules are registered in sys.modules by lib.rs at import time.
from genai_coroutines._internal import ocr, responses

# Re-export classes for convenience
DocumentConfig = ocr.DocumentConfig
DocumentProcessor = ocr.DocumentProcessor
ResponsesRequest = responses.ResponsesRequest
ResponsesProcessor = responses.ResponsesProcessor

__version__ = "1.0.0"

__all__ = [
    "DocumentConfig",
    "DocumentProcessor",
    "ResponsesRequest",
    "ResponsesProcessor",
    "__version__",
    "parse_responses",
    "parse_documents",
]


def parse_responses(results):
    """
    Extract clean text from OpenAI Responses API results.

    Returns exactly ONE string per input prompt (1:1 mapping).
    For each prompt, aggregates all text from message content blocks.
    For tool-call-only responses, returns the full raw JSON so you can
    inspect tool calls. Failed prompts return "".

    Args:
        results (dict): Results from ResponsesProcessor.process_batch()

    Returns:
        list[str]: List of assistant message texts (same length as input prompts)

    Example:
        >>> results = await processor.process_batch(request)
        >>> texts = parse_responses(results)
        >>> len(texts) == len(request.user_prompts)
        True
        >>> print(texts[0])
        '{"name": "John Doe", "age": 45}'
    """
    parsed = []
    for result in results.get("results", []):
        if not result.get("success"):
            parsed.append("")
            continue

        try:
            response_json = json.loads(result["raw_response"])
            output_items = response_json.get("output", [])

            text_parts = []
            has_tool_calls = False

            for output_item in output_items:
                item_type = output_item.get("type", "")

                if item_type == "message":
                    for content in output_item.get("content", []):
                        if content.get("type") == "output_text":
                            text = content.get("text", "")
                            if text:
                                text_parts.append(text)

                elif item_type == "function_call":
                    has_tool_calls = True

            if text_parts:
                parsed.append("\n".join(text_parts))
            elif has_tool_calls:
                # Model called tools instead of producing text —
                # return full raw JSON so the caller can inspect tool calls
                parsed.append(result["raw_response"])
            else:
                parsed.append("")

        except (json.JSONDecodeError, KeyError, TypeError):
            parsed.append("")

    return parsed


def parse_documents(results):
    """
    Extract content from Datalab OCR results.

    Handles all output formats automatically:
    - json:        Extracts and concatenates HTML from json.children[].html
    - html:        Returns the raw HTML string directly
    - markdown:    Returns the markdown string directly
    - paginated:   Concatenates content across pages
    - page_schema: Returns the full structured JSON string

    Returns exactly ONE string per input document (1:1 mapping).
    Failed documents return "".

    Args:
        results (list[dict]): Results from DocumentProcessor.process_multiparts()

    Returns:
        list[str]: List of content strings (same length as input documents)

    Example:
        >>> results = await processor.process_multiparts(batch)
        >>> contents = parse_documents(results)
        >>> len(contents) == len(batch)
        True
    """
    parsed = []
    for result in results:
        if not result.get("success"):
            parsed.append("")
            continue

        raw = result.get("json_response")
        if not raw:
            parsed.append("")
            continue

        try:
            response_json = json.loads(raw)

            # --- Strategy 1: JSON format with page children (most common) ---
            json_root = response_json.get("json")
            if isinstance(json_root, dict):
                children = json_root.get("children", [])
                if children:
                    html_parts = []
                    for page in children:
                        page_html = page.get("html", "")
                        if page_html:
                            html_parts.append(page_html)
                    if html_parts:
                        parsed.append("\n".join(html_parts))
                        continue

            # --- Strategy 2: Direct HTML output ---
            html_val = response_json.get("html")
            if html_val and isinstance(html_val, str):
                parsed.append(html_val)
                continue

            # --- Strategy 3: Markdown output ---
            md_val = response_json.get("markdown")
            if md_val and isinstance(md_val, str):
                parsed.append(md_val)
                continue

            # --- Strategy 4: Paginated output (list of pages) ---
            pages = response_json.get("pages")
            if isinstance(pages, list):
                page_contents = []
                for page in pages:
                    if isinstance(page, dict):
                        content = (
                            page.get("html")
                            or page.get("markdown")
                            or page.get("text", "")
                        )
                        if content:
                            page_contents.append(content)
                    elif isinstance(page, str):
                        page_contents.append(page)
                if page_contents:
                    parsed.append("\n".join(page_contents))
                    continue

            # --- Strategy 5: page_schema structured extraction ---
            if response_json.get("page_schema") or response_json.get("schema"):
                parsed.append(raw)
                continue

            # Fallback: return raw JSON string so nothing is lost
            parsed.append(raw)

        except (json.JSONDecodeError, KeyError, TypeError):
            parsed.append("")

    return parsed