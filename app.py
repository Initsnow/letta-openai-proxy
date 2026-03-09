import os
import time
import uuid
from typing import Any, Dict, Generator, Union

import uvicorn
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from hayhooks import BasePipelineWrapper, create_app
from hayhooks.server.pipelines import registry
from hayhooks.server.routers import openai as openai_module_to_patch
from haystack.dataclasses import StreamingChunk
from haystack.tracing.logging_tracer import LoggingTracer
from haystack.tracing.tracer import enable_tracing, tracer
from hayhooks.server.routers.openai import (
    ChatCompletion,
    ChatRequest,
    Choice,
    Message,
    ModelObject,
    ModelsResponse,
)
from hayhooks.settings import settings
from letta_client import Letta
from loguru import logger
from logging_config import configure_logging

# Configure logging immediately
configure_logging()


# Optional: Enable Haystack content tracing if DEBUG level is set or explicit env var
if os.getenv("HAYSTACK_CONTENT_TRACING", "false").lower() == "true":
    tracer.is_content_tracing_enabled = True
    enable_tracing(
        LoggingTracer(
            tags_color_strings={
                "haystack.component.input": "\x1b[1;31m",
                "haystack.component.name": "\x1b[1;34m",
            }
        )
    )

# Define the Letta server URL and token
LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://letta:8283")
LETTA_API_TOKEN = os.getenv("LETTA_API_TOKEN", "")


def fetch_letta_models():
    """Fetch available models from Letta server using the Letta client directly"""
    try:
        effective_token = LETTA_API_TOKEN if LETTA_API_TOKEN else None
        # Initialize the Letta client
        client = Letta(base_url=LETTA_BASE_URL, api_key=effective_token)

        # Get the list of agents
        agents = client.agents.list()

        # Filter out agents with names ending in "sleeptime"
        return [
            {"id": agent.id, "name": agent.name}
            for agent in agents
            if not agent.name.endswith("sleeptime")
        ]
    except Exception as e:
        logger.error(
            f"Unexpected error when fetching agents from Letta: {e}", exc_info=True
        )
        return []


async def get_models_override():
    """
    Override of the OpenAI /models endpoint to return Letta models.

    This returns a list of available Letta agents as OpenAI-compatible models.
    """
    # OPT-3: fetch_letta_models() is a blocking network call; run it in a thread
    # pool to avoid blocking the async event loop.
    letta_models = await run_in_threadpool(fetch_letta_models)

    return ModelsResponse(
        data=[
            ModelObject(
                id=model["id"],
                name=model["name"],
                object="model",
                created=int(time.time()),
                owned_by="letta",
            )
            for model in letta_models
        ],
        object="list",
    )


openai_module_to_patch.get_models = get_models_override


async def chat_completions_override(
    chat_req: ChatRequest,
) -> Union[ChatCompletion, StreamingResponse]:
    # Get the letta_proxy pipeline wrapper
    # Assuming 'letta_proxy' is the registered name of your pipeline
    pipeline_wrapper = registry.get("letta_proxy")

    if not pipeline_wrapper:
        logger.error("Pipeline 'letta_proxy' not found in registry.")
        raise HTTPException(
            status_code=500, detail="Chat backend pipeline 'letta_proxy' not found."
        )

    if not isinstance(pipeline_wrapper, BasePipelineWrapper):
        logger.error(
            f"Retrieved 'letta_proxy' is not a BasePipelineWrapper instance. Type: {type(pipeline_wrapper)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Chat backend pipeline 'letta_proxy' is of an unexpected type.",
        )

    if (
        not pipeline_wrapper._is_run_chat_completion_implemented
    ):  # Now Pylance should be happier after isinstance
        logger.error(
            f"Pipeline 'letta_proxy' (type: {type(pipeline_wrapper)}) does not implement run_chat_completion."
        )
        raise HTTPException(
            status_code=501,
            detail="Chat completions endpoint not implemented for 'letta_proxy' model.",
        )

    request_body_dump = chat_req.model_dump()
    if "agent_id" not in request_body_dump:
        request_body_dump["agent_id"] = chat_req.model
        logger.info(
            f"Injected agent_id='{chat_req.model}' into request_body_dump for letta_proxy."
        )

    try:
        result_generator = await run_in_threadpool(
            pipeline_wrapper.run_chat_completion,
            model=chat_req.model,
            messages=chat_req.messages,
            body=request_body_dump,
        )
    except ValueError as ve:
        logger.error(f"ValueError in letta_proxy.run_chat_completion: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(
            f"Exception calling letta_proxy.run_chat_completion: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Error processing chat request with letta_proxy."
        )

    resp_id = f"chatcmpl-{uuid.uuid4()}"  # OpenAI compatible ID

    # OPT-4: Compute `created` once so all chunks in this response share the
    # same timestamp, matching the OpenAI streaming specification.
    created_at = int(time.time())

    def stream_chunks() -> Generator[str, None, None]:
        try:
            for chunk_content in result_generator:
                tool_calls = None
                content = ""

                if isinstance(chunk_content, StreamingChunk):
                    content = chunk_content.content
                    if "tool_calls" in chunk_content.meta:
                        tool_calls = chunk_content.meta["tool_calls"]
                elif isinstance(chunk_content, str):
                    content = chunk_content
                else:
                    logger.warning(
                        f"letta_proxy returned non-string chunk: {type(chunk_content)}. Converting to str."
                    )
                    content = str(chunk_content)

                # Construct Message arguments
                msg_args = {"role": "assistant"}
                if content:
                    msg_args["content"] = content
                if tool_calls:
                    msg_args["tool_calls"] = tool_calls

                chunk_resp = ChatCompletion(
                    id=resp_id,
                    object="chat.completion.chunk",
                    created=created_at,
                    model=chat_req.model,
                    choices=[Choice(index=0, delta=Message(**msg_args))],  # pyright: ignore[reportArgumentType]
                )
                yield f"data: {chunk_resp.model_dump_json()}\n\n"

            final_chunk = ChatCompletion(
                id=resp_id,
                object="chat.completion.chunk",
                created=created_at,
                model=chat_req.model,
                choices=[
                    Choice(
                        index=0,
                        delta=Message(role="assistant", content=""),  # pyright: ignore[reportArgumentType]
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            # BUG-7: Do NOT leak the error message as an AI reply.
            # Log the error and close the stream gracefully.
            logger.error(f"Error during streaming from letta_proxy: {e}", exc_info=True)
            yield "data: [DONE]\n\n"

    if chat_req.stream:
        logger.info(f"Returning StreamingResponse for model {chat_req.model}")
        return StreamingResponse(stream_chunks(), media_type="text/event-stream")
    else:
        # Non-streaming: collect all chunks and return a single ChatCompletion
        logger.info(
            f"Returning non-streaming ChatCompletion for model {chat_req.model}"
        )
        full_response_content = ""
        try:
            tool_calls = []
            for chunk_content in result_generator:
                if isinstance(chunk_content, StreamingChunk):
                    # Check for tool_calls in meta
                    if "tool_calls" in chunk_content.meta:
                        tool_calls.extend(chunk_content.meta["tool_calls"])
                    chunk_content = chunk_content.content
                elif not isinstance(chunk_content, str):
                    logger.warning(
                        f"letta_proxy returned non-string chunk (non-streaming): {type(chunk_content)}. Converting to str."
                    )
                    chunk_content = str(chunk_content)
                full_response_content += chunk_content

            # Check if we have tool calls
            msg_args: Dict[str, Any] = {"role": "assistant"}
            if full_response_content:
                msg_args["content"] = full_response_content
            else:
                # If no content but tool calls, content should probably be null or empty string depending on client expectation.
                # OpenAI usually sends null content with tool calls.
                msg_args["content"] = None

            if tool_calls:
                msg_args["tool_calls"] = tool_calls
                logger.info(f"Including tool_calls in final response: {tool_calls}")

            # Use model_construct to bypass validation for "content" being None or strict literals
            message = Message.model_construct(**msg_args)  # type: ignore

            if not tool_calls and not full_response_content:
                logger.warning(
                    "Sending empty content and no tool calls. This might confuse the client."
                )

            # Use model_construct for Choice as well to allow "tool_calls" finish reason
            choice = Choice.model_construct(
                index=0,
                message=message,
                finish_reason="tool_calls" if tool_calls else "stop",  # type: ignore
            )

            final_resp = ChatCompletion(
                id=resp_id,
                object="chat.completion",
                created=int(time.time()),
                model=chat_req.model,
                choices=[choice],
            )
            logger.debug(
                f"Final ChatCompletion Response: {final_resp.model_dump_json()}"
            )
            return final_resp
        except Exception as e:
            logger.error(
                f"Error during non-streaming from letta_proxy: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail=f"Error collecting stream from letta_proxy: {e}"
            )


for route_idx, route in enumerate(openai_module_to_patch.router.routes):
    if isinstance(route, APIRoute):
        if route.path in ["/models", "/v1/models"]:
            route.endpoint = get_models_override
        elif (
            route.path in ["/chat/completions", "/v1/chat/completions"]
            or route.operation_id == "chat_completions"
        ):  # covers /{pipeline_name}/chat
            route.endpoint = chat_completions_override

hayhooks = create_app()

if __name__ == "__main__":
    import os as _os

    _log_level = _os.getenv("LOG_LEVEL", "INFO").lower()
    # log_config=None: prevent uvicorn from calling dictConfig() which would
    # override our logging setup. log_level keeps uvicorn's own output in sync.
    uvicorn.run(
        "app:hayhooks",
        host=settings.host,
        port=settings.port,
        log_config=None,
        log_level=_log_level,
    )
