# Letta OpenAI Proxy

This project provides an OpenAI-compatible API for [Letta](https://docs.letta.com) agents, allowing you to use Letta agents with any OpenAI-compatible client (e.g., Open WebUI, LibreChat, or custom scripts).


## Key Features

- **OpenAI Compatibility**: Seamlessly integrates Letta agents into the OpenAI ecosystem.
- **Dynamic Model Discovery**: Automatically lists your Letta agents as available models via the `/v1/models` endpoint.
- **Streaming Support**: Real-time token streaming for a responsive chat experience.
- **Tool Calling**: Full support for tool execution, including client-side tool execution via `ApprovalRequestMessage`.
- **Reasoning Visualization**: Displays Letta's reasoning process (CoT) within `<think>` tags, compatible with modern UI clients.
- **Image Support**: Support for multimodal inputs (images) in chat completions.
- **Structured Logging**: High-quality observability using `structlog`.
- **Haystack Integration**: Built on top of [Haystack](https://haystack.deepset.ai/) and [Hayhooks](https://github.com/deepset-ai/hayhooks) for a robust and extensible architecture.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) for Python dependency management.
- A running Letta server.

### Setup

1. **Sync dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   Copy `env_example` to `.env` and set your Letta server details:
   ```env
   LETTA_BASE_URL=http://your-letta-server:8283
   LETTA_API_TOKEN=your-letta-token (optional)
   ```

3. **Run the server**:
   ```bash
   uv run python app.py
   ```
   The proxy will be available at `http://localhost:1416`.

## Usage

### Listing Models
The proxy dynamically fetches your Letta agents and presents them as OpenAI models:
```bash
curl http://localhost:1416/v1/models
```

### Chatting with an Agent
You can use any OpenAI client. Point the base URL to `http://localhost:1416/v1` and use the Letta Agent ID or Name as the model name.

#### CLI Client
A simple CLI client is included for testing:
```bash
uv run python cli_client.py
```
Type `/models` to see available agents and start chatting.

## Credits & Thanks

Special thanks to **[wsargent](https://github.com/wsargent)** for the original implementation of the [letta-openai-proxy](https://github.com/wsargent/letta-openai-proxy), which served as the foundation for this project.
