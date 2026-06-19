import os
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union


from loguru import logger
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage, StreamingChunk, select_streaming_callback
from haystack.utils import Secret
from hayhooks import BasePipelineWrapper, get_last_user_message, streaming_generator
from letta_client import Letta
from letta_client.types import MessageCreateParam, ToolReturnMessage
from letta_client.types.agents import (
    AssistantMessage,
    ImageContentParam,
    LettaResponse,
    LettaStreamingResponse,
    ReasoningMessage,
    TextContentParam,
    ToolCallMessage,
    Message as LettaMessageUnion,
    ApprovalRequestMessage,
)
from letta_client.types.agents.approval_create_param import ApprovalCreateParam
from letta_client.types.agents.message_create_params import (
    ClientTool,
    Message as LettaCreateMessage,
)
from letta_client.types.agents.letta_response import Usage
from letta_client.types.agents.letta_streaming_response import LettaUsageStatistics


@component
class LettaChatGenerator:
    """
    Generates chat responses using Letta.
    """

    def __init__(
        self,
        base_url: Optional[str] = os.getenv("LETTA_BASE_URL"),
        token: Optional[Secret] = Secret.from_env_var(
            ["LETTA_API_TOKEN"], strict=False
        ),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initialize the component with a Letta client.

        :param base_url: The base URL of the Letta instance.
        :param token: The token to use as HTTP bearer authorization for Letta.
        :param generation_kwargs: A dictionary with keyword arguments to customize text generation.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        logger.info(f"Using Letta base URL: {base_url}")
        self.base_url = base_url
        self.token = token
        # Don't allow any OpenAI generation kwargs for now.
        self.generation_kwargs = {}
        self.streaming_callback = streaming_callback
        self.request_options: Dict[str, Any] = {"timeout": 300, "max_retries": 3}
        self.passthrough_tools = (
            os.getenv("LETTA_PASSTHROUGH_TOOLS", "true").lower() == "true"
        )
        # OPT-1: Cache the Letta client to avoid recreating it on every request.
        self._client: Optional[Letta] = None
        self._client_tools_by_agent: Dict[str, List[ClientTool]] = {}
        self._client_tools_by_call_id: Dict[str, List[ClientTool]] = {}

    @component.output_types(replies=List[ChatMessage], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: Union[str, List],
        agent_id: str,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Send a query to Letta and return the response.

        :param prompt: The string prompt to use for text generation.
        :param agent_id: The id of the Letta agent to use for text generation.
        :param streaming_callback: An optional callable for handling streaming responses.
        :param tools: A list of tools available to the model.
        :param tool_choice: The tool choice for the model.
        :param kwargs: Additional keyword arguments (filtered for OpenAI compatibility).
        :returns:
            A list of strings containing the generated responses and a list of dictionaries containing the metadata for each response.
        """

        if kwargs:
            logger.warning(f"Received unexpected kwargs: {kwargs}")

        try:
            # OPT-1: Reuse the cached client; create one only if needed.
            if self._client is None:
                token_value = None if self.token is None else self.token.resolve_value()
                logger.info(f"Creating Letta client at {self.base_url}")
                self._client = Letta(base_url=self.base_url, api_key=token_value)
            client = self._client
            logger.info(f"Connecting to Letta at {self.base_url} with agent {agent_id}")
        except Exception as e:
            logger.exception(f"Failed to create Letta client: {str(e)}")
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        f"Failed to create Letta client: {str(e)}"
                    )
                ]
            }

        if not agent_id:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        f"No Letta agent ID available for {agent_id}!"
                    )
                ]
            }

        try:
            # Check for tool results in the prompt/messages
            approval_message = self._process_tool_input(prompt)
            if approval_message:
                # If we have an approval message (tool results), we use that instead of a new user message
                messages: List[LettaCreateMessage] = [approval_message]
                # OPT-2: ApprovalCreateParam is a TypedDict, use bracket notation not attribute
                _approvals = approval_message.get("approvals") or []
                num_approvals = len(list(_approvals))
                logger.debug(
                    f"Created approval message with {num_approvals} tool results"
                )
            else:
                message = self._message_from_user(prompt)
                messages: List[LettaCreateMessage] = [message]
                logger.debug(f"Created message: {message}")
        except Exception as e:
            logger.exception(f"Failed to create message from prompt: {str(e)}")
            return {
                "replies": [
                    ChatMessage.from_assistant(f"Failed to create message: {str(e)}")
                ]
            }

        supplied_client_tools = self._convert_tools_to_client_tools(tools) if tools else None
        client_tools = self._resolve_client_tools(
            agent_id=agent_id,
            supplied_client_tools=supplied_client_tools,
            approval_message=approval_message,
        )
        streaming_callback = select_streaming_callback(
            self.streaming_callback, streaming_callback, requires_async=False
        )

        completions: List[ChatMessage] = []
        if streaming_callback is not None:
            try:
                logger.info(f"Creating stream for agent_id: {agent_id}")
                logger.debug(f"Request timeout: {self.request_options.get('timeout')}")

                stream_completion: Iterator[LettaStreamingResponse] = (
                    client.agents.messages.create(
                        agent_id=agent_id,
                        messages=messages,
                        client_tools=client_tools,
                        streaming=True,
                        timeout=self.request_options.get("timeout"),
                    )
                )

                chunks = []
                # BUG-1: Use local stream_state dict instead of self.think_block_open
                # to be thread-safe across concurrent requests.
                stream_state = {
                    "think_block_open": False,
                    "client_tools": client_tools,
                    "client_tool_names": self._client_tool_names(client_tools),
                    "tool_call_index": 0,
                }
                last_chunk = None
                # Sometimes the response will time out while streaming, so we need a try / catch
                try:
                    for chunk in stream_completion:
                        last_chunk = chunk

                        chunk_delta: Optional[StreamingChunk] = (
                            self._process_streaming_chunk(chunk, stream_state)
                        )
                        if chunk_delta:
                            chunks.append(chunk_delta)
                            streaming_callback(chunk_delta)

                    # BUG-3: Guard against last_chunk being None (empty stream)
                    if last_chunk is None:
                        logger.warning(
                            "Stream produced no chunks for agent %s", agent_id
                        )
                        completions = [ChatMessage.from_assistant("")]
                    else:
                        completions = [
                            self._create_message_from_chunks(
                                agent_id, last_chunk, chunks
                            )
                        ]
                except Exception as e:
                    logger.exception(
                        f"An error occurred while processing a streaming response: {str(e)}"
                    )
                    completions = [
                        ChatMessage.from_assistant(
                            f"An error occurred while streaming response: {str(e)}"
                        )
                    ]
            except Exception as e:
                logger.exception(
                    f"Failed to create Letta stream for agent {agent_id}: {str(e)}"
                )
                completions = [
                    ChatMessage.from_assistant(f"Failed to create stream: {str(e)}")
                ]

        else:
            try:
                completion: LettaResponse = client.agents.messages.create(
                    agent_id=agent_id,
                    messages=messages,
                    client_tools=client_tools,
                    streaming=False,
                    timeout=self.request_options.get("timeout"),
                )
                completions = [self._build_message(agent_id, completion, client_tools)]
            except Exception as e:
                logger.exception(
                    f"An error occurred while processing a response: {str(e)}"
                )
                completions = [
                    ChatMessage.from_assistant(
                        f"An error occurred while waiting for response: {str(e)}"
                    )
                ]

        # logger.debug(f"run: completions={completions}")

        return {"replies": completions}

    def _convert_tools_to_client_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[ClientTool]:
        """
        Convert OpenAI dictionary-style tools to Letta client_tools format.
        OpenAI format: {"type": "function", "function": {"name": "...", "description": "...", "parameters": ...}}
        Letta client_tools format expected by SDK: [{"name": "...", "description": "...", "parameters": ...}, ...]
        """
        if not self.passthrough_tools:
            logger.debug(
                "LETTA_PASSTHROUGH_TOOLS is false, ignoring client provided tools."
            )
            return []

        client_tools: List[ClientTool] = []
        for tool in tools:
            if tool.get("type") == "function":
                function_def = tool.get("function", {})
                client_tool: ClientTool = {
                    "name": function_def.get("name", ""),
                    "description": function_def.get("description"),
                    "parameters": function_def.get("parameters"),
                }
                client_tools.append(client_tool)
        return client_tools

    @staticmethod
    def _copy_client_tools(
        client_tools: Optional[List[ClientTool]],
    ) -> Optional[List[ClientTool]]:
        if not client_tools:
            return None

        return [dict(tool) for tool in client_tools]  # type: ignore[list-item]

    def _resolve_client_tools(
        self,
        agent_id: str,
        supplied_client_tools: Optional[List[ClientTool]],
        approval_message: Optional[ApprovalCreateParam],
    ) -> Optional[List[ClientTool]]:
        if supplied_client_tools:
            copied_tools = self._copy_client_tools(supplied_client_tools)
            if copied_tools:
                self._client_tools_by_agent[agent_id] = copied_tools
            return supplied_client_tools

        if not approval_message:
            return None

        for approval in approval_message.get("approvals", []):
            tool_call_id = approval.get("tool_call_id")
            if tool_call_id and tool_call_id in self._client_tools_by_call_id:
                return self._copy_client_tools(
                    self._client_tools_by_call_id[tool_call_id]
                )

        return self._copy_client_tools(self._client_tools_by_agent.get(agent_id))

    def _remember_client_tool_call(
        self, tool_call_id: str, client_tools: Optional[List[ClientTool]]
    ) -> None:
        copied_tools = self._copy_client_tools(client_tools)
        if copied_tools:
            self._client_tools_by_call_id[tool_call_id] = copied_tools

    @staticmethod
    def _client_tool_names(client_tools: Optional[List[ClientTool]]) -> set[str]:
        if not client_tools:
            return set()

        return {tool["name"] for tool in client_tools if tool.get("name")}

    @staticmethod
    def _tool_call_payload(tool_call: Any, index: int) -> Dict[str, Any]:
        return {
            "index": index,
            "id": tool_call.tool_call_id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        }

    def _process_tool_input(
        self, prompt: Union[str, List]
    ) -> Optional[ApprovalCreateParam]:
        """
        Check if the input messages contain tool results (role='tool') and convert them to an ApprovalMessage.
        This handles the client-side tool execution flow.
        """
        if (
            isinstance(prompt, list)
            and len(prompt) > 0
            and isinstance(prompt[0], dict)
            and prompt[0].get("type") == "tool_result"
        ):
            # Construct approval message
            approvals = []
            for item in prompt:
                approvals.append(
                    {
                        "type": "tool",
                        "tool_call_id": item["tool_call_id"],
                        "tool_return": item["content"],
                        "status": "success",
                    }
                )
            return ApprovalCreateParam(type="approval", approvals=approvals)
        return None

    @staticmethod
    def _message_from_user(prompt: Union[str, List]) -> MessageCreateParam:
        if isinstance(prompt, str):
            return MessageCreateParam(
                role="user", content=[TextContentParam(type="text", text=prompt)]
            )

        content_parts = []
        for part in prompt:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    content_parts.append(
                        TextContentParam(type="text", text=part.get("text", ""))
                    )
                elif part_type == "image_url":
                    # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        # Base64 data URI
                        # Format: data:image/jpeg;base64,...
                        try:
                            header, data = image_url.split(",", 1)
                            media_type = header.split(":", 1)[1].split(";", 1)[0]
                            content_parts.append(
                                ImageContentParam(
                                    type="image",
                                    source={
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Failed to parse base64 image URL: {e}")
                    else:
                        # Standard URL
                        content_parts.append(
                            ImageContentParam(
                                type="image", source={"type": "url", "url": image_url}
                            )
                        )
                elif part_type == "image":
                    # Letta format pass-through (if source structure matches)
                    # We assume it matches Letta's expected dict structure or is convertible
                    # Since ImageContent expects specific fields, let's try to adapt or pass generic dict if needed
                    # But Letta client likely expects objects.
                    # The user provided: {"type": "image", "source": {...}}
                    if "source" in part:
                        content_parts.append(
                            ImageContentParam(type="image", source=part["source"])
                        )
                else:
                    logger.warning(f"Unknown content part type: {part_type}")

        return MessageCreateParam(role="user", content=content_parts)

    def _create_message_from_chunks(
        self, agent_id, completion_chunk, streamed_chunks: List[StreamingChunk]
    ) -> ChatMessage:
        """
        Creates a single ChatMessage from the streamed chunks. Some data is retrieved from the completion chunk.
        """
        # logger.debug(f"_create_message_from_chunks: completion_chunk={completion_chunk}, streamed_chunks={streamed_chunks}")

        # "".join([chunk.content for chunk in streamed_chunks])
        complete_response = ChatMessage.from_assistant("")
        finish_reason = "stop"  # streamed_chunks[-1].meta["finish_reason"]

        usage_dict = {}
        if isinstance(completion_chunk, LettaUsageStatistics):
            usage_dict = {
                "completion_tokens": completion_chunk.completion_tokens,
                "prompt_tokens": completion_chunk.prompt_tokens,
                "total_tokens": completion_chunk.total_tokens,
            }

        complete_response.meta.update(
            {
                "model": agent_id,
                "index": 0,
                "finish_reason": finish_reason,
                "completion_start_time": streamed_chunks[0].meta.get("received_at")
                if streamed_chunks
                else None,
                "usage": usage_dict,
            }
        )
        return complete_response

    def _debug_tool_statements(self) -> bool:
        """
        Returns True if the environment variable LETTA_CHAT_DEBUG_TOOL_STATEMENTS is set to True.
        """
        return os.getenv("LETTA_CHAT_DEBUG_TOOL_STATEMENTS", "False").lower() == "true"

    def _process_streaming_chunk(
        self, chunk: LettaStreamingResponse, stream_state: Dict[str, Any]
    ) -> Optional[StreamingChunk]:
        """
        Process a streaming chunk based on its type and invoke the streaming callback.

        :param chunk: The streaming chunk to process.
        :param stream_state: A per-request mutable dict holding streaming state
            (e.g. ``think_block_open``).  Using a dict instead of instance attributes
            makes this method safe for concurrent requests (BUG-1).
        """
        content_prefix = ""

        # Check if we need to open a think block
        is_think_chunk = isinstance(
            chunk, (ReasoningMessage, ToolCallMessage, ToolReturnMessage)
        )
        if is_think_chunk and not stream_state["think_block_open"]:
            content_prefix = "<think>"
            stream_state["think_block_open"] = True

        # Check if we need to close a think block
        is_assistant_chunk = isinstance(chunk, AssistantMessage)
        if is_assistant_chunk and stream_state["think_block_open"]:
            content_prefix = "</think>"
            stream_state["think_block_open"] = False

        if isinstance(chunk, ReasoningMessage):
            reasoning_chunk: ReasoningMessage = chunk
            now = datetime.now()
            meta_dict = {"type": "assistant", "received_at": now.isoformat()}
            display_time = now.astimezone().time().isoformat("seconds")
            reasoning = (
                reasoning_chunk.reasoning.strip().removeprefix('"').removesuffix('"')
            )
            content = f"\n- {display_time} {reasoning}"
            return StreamingChunk(content=content_prefix + content, meta=meta_dict)

        if isinstance(chunk, ToolCallMessage):
            tool_call = chunk.tool_call

            now = datetime.now()
            display_time = now.astimezone().time().isoformat("seconds")
            meta_dict = {"type": "assistant", "received_at": now.isoformat()}
            tool_name = tool_call.name

            call_statement = f"Calling server tool {tool_name}"

            arguments: str = tool_call.arguments or "{}"

            no_heartbeat_requested = """"request_heartbeat": false""" in arguments
            if no_heartbeat_requested:
                call_statement = call_statement + " *without heartbeat*"

            if self._debug_tool_statements():  # BUG-4: fixed typo
                call_statement = call_statement + " with arguments: " + arguments

            content = f"\n- {display_time} {call_statement}..."
            return StreamingChunk(content=content_prefix + content, meta=meta_dict)

        if isinstance(chunk, ApprovalRequestMessage):
            tool_call = chunk.tool_call

            now = datetime.now()
            display_time = now.astimezone().time().isoformat("seconds")
            meta_dict = {"type": "assistant", "received_at": now.isoformat()}
            tool_name = tool_call.name

            client_tool_names: set[str] = stream_state.get("client_tool_names", set())
            is_client_tool = tool_name in client_tool_names

            arguments: str = tool_call.arguments or "{}"

            if is_client_tool:
                tool_call_index = stream_state.get("tool_call_index", 0)
                tool_call_payload = self._tool_call_payload(tool_call, tool_call_index)
                stream_state["tool_call_index"] = tool_call_index + 1
                self._remember_client_tool_call(
                    tool_call.tool_call_id, stream_state.get("client_tools")
                )
                logger.debug(
                    f"constructed client tool_call_payload from ApprovalRequestMessage: {tool_call_payload}"
                )

                # OpenAI clients execute client tools when they receive tool_calls.
                meta_dict["tool_calls"] = [tool_call_payload]  # type: ignore[assignment]
                meta_dict["finish_reason"] = "tool_calls"
                return StreamingChunk(content="", meta=meta_dict)

            call_statement = f"Server tool {tool_name} requires approval in Letta"
            no_heartbeat_requested = """"request_heartbeat": false""" in arguments
            if no_heartbeat_requested:
                call_statement = call_statement + " *without heartbeat*"

            if self._debug_tool_statements():
                call_statement = call_statement + " with arguments: " + arguments

            content = f"\n- {display_time} {call_statement}..."
            return StreamingChunk(content=content_prefix + content, meta=meta_dict)

        if isinstance(chunk, ToolReturnMessage):
            tool_return_message: ToolReturnMessage = chunk
            now = datetime.now()
            meta_dict = {"type": "assistant", "received_at": now.isoformat()}
            content = f" {tool_return_message.status}, returned {len(tool_return_message.tool_return)} characters."
            return StreamingChunk(content=content_prefix + content, meta=meta_dict)

        if isinstance(chunk, AssistantMessage):
            now = datetime.now()
            meta_dict = {"type": "assistant", "received_at": now.isoformat()}

            if isinstance(chunk.content, list):
                # Handle list of content parts (LettaAssistantMessageContentUnion)
                text_content = "".join(
                    [part.text for part in chunk.content if hasattr(part, "text")]
                )
            else:
                text_content = chunk.content or ""

            content = content_prefix + text_content
            return StreamingChunk(content=content, meta=meta_dict)

        return None

    def _build_message(
        self,
        agent_id: str,
        response: LettaResponse,
        client_tools: Optional[List[ClientTool]] = None,
    ):
        """
        Converts the response from Letta to a ChatMessage.

        :param response:
            The response returned by Letta.
        :returns:
            The ChatMessage.
        """
        # logger.debug(f"_build_message: response={response}")

        messages: List[LettaMessageUnion] = response.messages
        usage: Usage = response.usage
        usage_dict = {
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
        }

        # BUG-5: Accumulate text from ALL AssistantMessages (not just the last one)
        # and use an incrementing index for tool_calls.
        chat_message = None
        assistant_content_parts: List[str] = []
        tool_call_index = 0
        client_tool_names = self._client_tool_names(client_tools)

        for message in messages:
            if isinstance(message, AssistantMessage):
                if isinstance(message.content, list):
                    content_str = "".join(
                        [part.text for part in message.content if hasattr(part, "text")]
                    )
                else:
                    content_str = message.content or ""
                if content_str:
                    assistant_content_parts.append(content_str)

            elif isinstance(message, ApprovalRequestMessage):
                tool_call = message.tool_call
                is_client_tool = tool_call.name in client_tool_names

                if not is_client_tool:
                    assistant_content_parts.append(
                        f"Server tool {tool_call.name} requires approval in Letta."
                    )
                    continue

                # BUG-5: Use incrementing index so multiple tool calls are correctly indexed
                tool_call_payload = self._tool_call_payload(tool_call, tool_call_index)
                tool_call_index += 1
                self._remember_client_tool_call(tool_call.tool_call_id, client_tools)

                if not chat_message:
                    chat_message = ChatMessage.from_assistant("")

                if "tool_calls" not in chat_message.meta:
                    chat_message.meta["tool_calls"] = []

                chat_message.meta["tool_calls"].append(tool_call_payload)

        # Merge all accumulated assistant text into the final message
        if assistant_content_parts:
            combined_content = "".join(assistant_content_parts)
            if chat_message:
                chat_message._content = [combined_content]  # type: ignore[assignment]
            else:
                chat_message = ChatMessage.from_assistant(combined_content)

        if not chat_message:
            chat_message = ChatMessage.from_assistant("No message found")

        chat_message.meta.update(
            {
                "model": agent_id,
                "index": 0,
                "finish_reason": "tool_calls"
                if "tool_calls" in chat_message.meta
                else "stop",
                "usage": usage_dict,
            }
        )
        return chat_message


class PipelineWrapper(BasePipelineWrapper):
    skip_mcp = True

    def setup(self) -> None:
        self.pipeline = Pipeline()

        letta_chat_generator = LettaChatGenerator()
        self.pipeline.add_component("llm", letta_chat_generator)

    def run_api(self, prompt: str = "", agent_id: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        result = self.pipeline.run({"llm": {"prompt": prompt, "agent_id": agent_id}})
        return result["llm"]["replies"][0]

    def run_chat_completion(
        self, model: str, messages: List[dict], body: dict
    ) -> Union[str, Generator]:
        # The body argument contains the full request body, which may be used to extract more
        # information like the temperature or the max_tokens (see the OpenAI API reference for more information).
        logger.debug(
            f"Running pipeline with model: {model}, body keys: {list(body.keys())}"
        )

        # Filter out OpenAI-specific parameters that might conflict with Letta
        filtered_body = {}
        for key, value in body.items():
            if key not in [
                "stream",
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "user",
                "n",
                "stop",
            ]:
                filtered_body[key] = value

        # Check if agent_id is in the nested body structure
        if (
            "body" in filtered_body
            and isinstance(filtered_body["body"], dict)
            and "agent_id" in filtered_body["body"]
        ):
            agent_id = filtered_body["body"]["agent_id"]
        else:
            agent_id = filtered_body.get("agent_id")
            if not agent_id:
                raise ValueError("No agent_id provided in the request body")
        if "tools" in filtered_body:
            tools = filtered_body.pop("tools")
        else:
            tools = None

        if "tool_choice" in filtered_body:
            tool_choice = filtered_body.pop("tool_choice")
        else:
            tool_choice = None

        # Custom logic to handle "tool" role messages which hayhooks.get_last_user_message ignores.
        # If the last message is a tool message, we want to pass THAT to the pipeline.
        # We can pass it as a special list to prompt.

        last_message = messages[-1] if messages else None
        prompt = None

        if last_message and last_message.get("role") == "tool":
            # Collect all trailing tool messages
            tool_messages = []
            for msg in reversed(messages):
                if msg.get("role") == "tool":
                    tool_messages.insert(0, msg)
                else:
                    break

            # Pass these as a special prompt structure that LettaChatGenerator will recognize
            # We wrap it in a list of dicts which _message_from_user might try to parse,
            # so we should use a distinct type or key.
            # But wait, `LettaChatGenerator.run` takes `prompt`.
            # We can change `_message_from_user` or `_process_tool_input` to handle this.
            prompt = [
                {
                    "type": "tool_result",
                    "content": m["content"],
                    "tool_call_id": m["tool_call_id"],
                }
                for m in tool_messages
            ]
        else:
            # Fallback to standard behavior
            prompt = get_last_user_message(messages)  # type: ignore[arg-type]

        return streaming_generator(
            pipeline=self.pipeline,
            pipeline_run_args={
                "llm": {
                    "prompt": prompt,
                    "agent_id": agent_id,
                    "tools": tools,
                    "tool_choice": tool_choice,
                }
            },
        )
