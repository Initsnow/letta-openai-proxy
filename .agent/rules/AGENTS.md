# AGENTS.md

This file provides guidance to Antigravity (and other AI coding agents) when working with this repository.

## Project Overview

**letta-openai-proxy** — Makes Letta agents available through an OpenAI-compatible API. Built on Hayhooks (web framework) + Haystack pipeline, exposing Letta agents as OpenAI models (`/v1/models`, `/v1/chat/completions`).

## Key Files

| File | Role |
|------|------|
| `app.py` | FastAPI entry point; patches Hayhooks router for `/models` and `/chat/completions` |
| `pipelines/letta_proxy/pipeline_wrapper.py` | Haystack pipeline + `LettaChatGenerator` component |
| `cli_client.py` | Interactive CLI REPL client (for manual testing) |
| `logging_config.py` | structlog setup (JSON or pretty, controlled by env vars) |
| `tests/test_tool_calling.py` | Unit tests for tool-call flow |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LETTA_BASE_URL` | `http://letta:8283` | Letta server URL |
| `LETTA_API_TOKEN` | _(empty)_ | Bearer token for Letta |
| `LETTA_PASSTHROUGH_TOOLS` | `true` | Forward OpenAI tools to Letta |
| `LETTA_CHAT_DEBUG_TOOL_STATEMENTS` | `false` | Log full tool call arguments |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HAYSTACK_LOGGING_USE_JSON` | `false` | JSON vs pretty log output |
| `HAYSTACK_CONTENT_TRACING` | `false` | Enable Haystack content tracing |

## Development Commands

```pwsh
# Install / sync deps (Windows)
uv sync

# Run server
uv run python app.py

# Run CLI client
uv run python cli_client.py

# Run tests
uv run pytest tests/ -v

# Add a production dependency (needed by the server at runtime)
uv add <package>

# Add a dev-only dependency (tests, linters — not included in production)
uv add --dev <package>
```

## Architecture & Response Flow

```
Client
  │
  ▼ GET /v1/models
app.py: get_models_override()
  └─► Letta.agents.list()  →  returns agents (filtered: no "sleeptime" suffix)

  ▼ POST /v1/chat/completions
app.py: chat_completions_override()
  └─► PipelineWrapper.run_chat_completion()
        └─► LettaChatGenerator.run()
              └─► client.agents.messages.create(streaming=True/False)
                    ├─ ReasoningMessage   → <think>\n- HH:MM:SS reasoning…
                    ├─ ToolCallMessage    → <think>\n- HH:MM:SS Calling tool…
                    ├─ ToolReturnMessage  → status, N chars returned
                    └─ AssistantMessage   → </think> + text content
```

### Streaming Think-block State (thread-safe)

`_process_streaming_chunk(chunk, stream_state)` receives a **per-request** `stream_state` dict (`{"think_block_open": bool}`) instead of storing state on `self`, making concurrent requests safe.

### Letta Client Caching

`LettaChatGenerator` caches the `Letta` client in `self._client` and reuses it across requests. Invalidate by setting `self._client = None`.

### Tool Call Flow (client-side tools)

1. `PipelineWrapper.run_chat_completion()` detects trailing `role=tool` messages
2. Wraps them as `{"type": "tool_result", ...}` list
3. `LettaChatGenerator._process_tool_input()` converts to `ApprovalCreateParam`
4. Sent to Letta via `client.agents.messages.create()`

## Server

- Default port: **1416** (Hayhooks setting)
- 5-minute timeout, 3 retries per Letta request
- Agents with names ending in `sleeptime` are excluded from the model list

## Known Constraints

- `generation_kwargs` (temperature, max_tokens, etc.) are filtered out and **not** forwarded to Letta — Letta manages its own generation parameters
- `tool_choice` is accepted but currently passed through without enforcement
