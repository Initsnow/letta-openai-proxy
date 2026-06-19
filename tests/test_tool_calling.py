import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Union

# Set up mocks for dependencies BEFORE importing pipeline_wrapper
mock_hayhooks = MagicMock()
sys.modules["hayhooks"] = mock_hayhooks
sys.modules["haystack"] = MagicMock()
sys.modules["haystack.dataclasses"] = MagicMock()
sys.modules["haystack.utils"] = MagicMock()
sys.modules["structlog"] = MagicMock()


# Mock haystack component decorator to pass through
def mock_component_decorator(cls=None, **kwargs):
    if cls:
        return cls

    def wrapper(c):
        return c

    return wrapper


# output_types needs to be a decorator too
def mock_output_types(**kwargs):
    def wrapper(f):
        return f

    return wrapper


mock_haystack_component = MagicMock(side_effect=mock_component_decorator)
mock_haystack_component.output_types = mock_output_types
sys.modules["haystack"].component = mock_haystack_component
sys.modules["haystack.component"] = mock_haystack_component  # just in case


# Mock letta_client and its types
# Mock letta_client and its types with real classes for isinstance checks
mock_letta_client = MagicMock()
sys.modules["letta_client"] = mock_letta_client
# RequestOptions is now in letta_client, so we mock it there
mock_letta_client.RequestOptions = MagicMock()


# Define dummy classes for types used in isinstance checks
class MockLettaStreamingResponse:
    pass


class MockAssistantMessage:
    pass


class MockReasoningMessage:
    def __init__(self, reasoning=""):
        self.reasoning = reasoning


class MockToolCallMessage:
    def __init__(self, tool_call=None):
        self.tool_call = tool_call


class MockToolReturnMessage:
    def __init__(self, status="success", tool_return=""):
        self.status = status
        self.tool_return = tool_return


class MockLettaUsageStatistics:
    pass


class MockLettaResponse:
    pass


# Register them in sys.modules so import works
mock_streaming_response_mod = MagicMock()
mock_streaming_response_mod.LettaStreamingResponse = MockLettaStreamingResponse
# Added LettaUsageStatistics to streaming response module
mock_streaming_response_mod.LettaUsageStatistics = MockLettaUsageStatistics

# Mock letta_client.types
mock_types = MagicMock()
mock_types.ToolReturnMessage = MockToolReturnMessage
sys.modules["letta_client.types"] = mock_types

# Mock letta_client.types.agents
mock_agents_pkg = MagicMock()
mock_agents_pkg.LettaStreamingResponse = MockLettaStreamingResponse
mock_agents_pkg.AssistantMessage = MockAssistantMessage
mock_agents_pkg.ReasoningMessage = MockReasoningMessage
mock_agents_pkg.ToolCallMessage = MockToolCallMessage
mock_agents_pkg.Message = MagicMock()
mock_agents_pkg.LettaResponse = MockLettaResponse
sys.modules["letta_client.types.agents"] = mock_agents_pkg


# Mock letta_client.types.agents sub-modules imported by pipeline_wrapper
class MockApprovalRequestMessage:
    def __init__(self, tool_call=None):
        self.tool_call = tool_call


class MockApprovalCreateParam(dict):
    """Minimal TypedDict-compatible mock for ApprovalCreateParam."""

    pass


mock_approval_create_param_mod = MagicMock()
mock_approval_create_param_mod.ApprovalCreateParam = MockApprovalCreateParam
sys.modules["letta_client.types.agents.approval_create_param"] = (
    mock_approval_create_param_mod
)

mock_agents_pkg.ApprovalRequestMessage = MockApprovalRequestMessage

mock_message_create_params_mod = MagicMock()
sys.modules["letta_client.types.agents.message_create_params"] = (
    mock_message_create_params_mod
)

# letta_client.types.agents.letta_response (Usage class)
mock_agents_letta_response_mod = MagicMock()
mock_agents_letta_response_mod.Usage = MagicMock()
sys.modules["letta_client.types.agents.letta_response"] = mock_agents_letta_response_mod

# letta_client.types.agents.letta_streaming_response (LettaUsageStatistics)
sys.modules["letta_client.types.agents.letta_streaming_response"] = (
    mock_streaming_response_mod
)

mock_reasoning_message_mod = MagicMock()
mock_reasoning_message_mod.ReasoningMessage = MockReasoningMessage
sys.modules["letta_client.types.reasoning_message"] = mock_reasoning_message_mod

mock_tool_call_message_mod = MagicMock()
mock_tool_call_message_mod.ToolCallMessage = MockToolCallMessage
sys.modules["letta_client.types.tool_call_message"] = mock_tool_call_message_mod

mock_tool_return_message_mod = MagicMock()
mock_tool_return_message_mod.ToolReturnMessage = MockToolReturnMessage
sys.modules["letta_client.types.tool_return_message"] = mock_tool_return_message_mod

mock_usage_stats_mod = MagicMock()
mock_usage_stats_mod.LettaUsageStatistics = MockLettaUsageStatistics
sys.modules["letta_client.types.letta_usage_statistics"] = mock_usage_stats_mod

mock_response_mod = MagicMock()
mock_response_mod.LettaResponse = MockLettaResponse
sys.modules["letta_client.types.letta_response"] = mock_response_mod

sys.modules["letta_client.types.letta_message_union"] = MagicMock()

# Provide real or mock implementations for imported items
mock_hayhooks.BasePipelineWrapper = object  # simple base
mock_hayhooks.get_last_user_message = lambda x: "last message"
mock_hayhooks.streaming_generator = MagicMock()


# Mock StreamingChunk reasonably well since we use it
class MockStreamingChunk:
    def __init__(self, content, meta=None):
        self.content = content
        self.meta = meta or {}

    def __eq__(self, other):
        return self.content == other.content and self.meta == other.meta

    def __repr__(self):
        return f"StreamingChunk(content={self.content}, meta={self.meta})"


class MockChatMessage:
    def __init__(self, content):
        self._content = [content]
        self.meta = {}

    @classmethod
    def from_assistant(cls, content):
        return cls(content)

    @property
    def content(self):
        return "".join(str(part) for part in self._content)


sys.modules["haystack.dataclasses"].StreamingChunk = MockStreamingChunk
sys.modules["haystack.dataclasses"].ChatMessage = MockChatMessage

# Now we can import the module to test
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pipelines", "letta_proxy")
    )
)
# Also add project root for other imports if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We need to rely on the side effects of imports, so we might need to patch specifically inside the test
# but since we already mocked sys.modules, imports in pipeline_wrapper will get our mocks.
try:
    from pipeline_wrapper import LettaChatGenerator
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"sys.path: {sys.path}")
    raise e


class TestToolCalling(unittest.TestCase):
    def setUp(self):
        self.generator = LettaChatGenerator()
        # Enable tool debug logging for verification (optional)
        os.environ["LETTA_CHAT_DEBUG_TOOL_STATEMENTS"] = "True"

    def test_convert_tools_to_client_tools(self):
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        expected_client_tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]

        result = self.generator._convert_tools_to_client_tools(openai_tools)
        self.assertEqual(result, expected_client_tools)

    def test_process_tool_input_valid(self):
        prompt = [
            {
                "type": "tool_result",
                "tool_call_id": "call_123",
                "content": "Sunny and warm",
            }
        ]

        expected_approval = {
            "type": "approval",
            "approvals": [
                {
                    "type": "tool",
                    "tool_call_id": "call_123",
                    "tool_return": "Sunny and warm",
                    "status": "success",
                }
            ],
        }

        result = self.generator._process_tool_input(prompt)
        self.assertEqual(result, expected_approval)

    def test_process_tool_input_invalid(self):
        self.assertIsNone(self.generator._process_tool_input("some string"))
        self.assertIsNone(self.generator._process_tool_input([]))
        self.assertIsNone(
            self.generator._process_tool_input([{"type": "text", "text": "hello"}])
        )

    def test_process_streaming_chunk_server_tool_call_is_not_openai_tool_call(self):
        mock_tool_call = MagicMock()
        mock_tool_call.name = "get_weather"
        mock_tool_call.arguments = '{"location": "Paris"}'
        mock_tool_call.tool_call_id = "call_abc"

        from pipeline_wrapper import ToolCallMessage as MockToolCallMessageClass

        class RealMockToolCallMessage(MockToolCallMessageClass):
            pass

        chunk = RealMockToolCallMessage()
        chunk.tool_call = mock_tool_call

        stream_state = {
            "think_block_open": False,
            "client_tool_names": {"get_weather"},
            "tool_call_index": 0,
        }
        result = self.generator._process_streaming_chunk(chunk, stream_state)

        self.assertIsNotNone(result)
        self.assertNotIn("tool_calls", result.meta)
        self.assertIn("Calling server tool get_weather", result.content)

    def test_process_streaming_chunk_client_approval_request_becomes_tool_call(self):
        mock_tool_call = MagicMock()
        mock_tool_call.name = "get_weather"
        mock_tool_call.arguments = '{"location": "Paris"}'
        mock_tool_call.tool_call_id = "call_abc"

        from pipeline_wrapper import ApprovalRequestMessage as MockApprovalRequestClass

        class RealMockApprovalRequestMessage(MockApprovalRequestClass):
            pass

        chunk = RealMockApprovalRequestMessage()
        chunk.tool_call = mock_tool_call

        stream_state = {
            "think_block_open": False,
            "client_tool_names": {"get_weather"},
            "tool_call_index": 0,
        }
        result = self.generator._process_streaming_chunk(chunk, stream_state)

        self.assertIsNotNone(result)
        self.assertEqual(result.content, "")
        self.assertIn("tool_calls", result.meta)
        self.assertEqual(result.meta["finish_reason"], "tool_calls")
        self.assertEqual(len(result.meta["tool_calls"]), 1)
        self.assertEqual(
            result.meta["tool_calls"][0]["function"]["name"], "get_weather"
        )
        self.assertEqual(
            result.meta["tool_calls"][0]["function"]["arguments"],
            '{"location": "Paris"}',
        )
        self.assertEqual(result.meta["tool_calls"][0]["id"], "call_abc")
        self.assertEqual(result.meta["tool_calls"][0]["index"], 0)

    def test_build_message_maps_client_approval_request_to_tool_calls(self):
        mock_tool_call = MagicMock()
        mock_tool_call.name = "get_weather"
        mock_tool_call.arguments = '{"location": "Paris"}'
        mock_tool_call.tool_call_id = "call_abc"

        response = MockLettaResponse()
        response.messages = [MockApprovalRequestMessage(tool_call=mock_tool_call)]
        response.usage = MagicMock(
            completion_tokens=1,
            prompt_tokens=2,
            total_tokens=3,
        )

        result = self.generator._build_message(
            "agent-123",
            response,
            client_tools=[
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {},
                }
            ],
        )

        self.assertEqual(result.meta["finish_reason"], "tool_calls")
        self.assertEqual(len(result.meta["tool_calls"]), 1)
        self.assertEqual(
            result.meta["tool_calls"][0]["function"]["name"], "get_weather"
        )
        self.assertEqual(
            self.generator._client_tools_by_call_id["call_abc"][0]["name"],
            "get_weather",
        )

    def test_build_message_does_not_expose_server_approval_as_tool_call(self):
        mock_tool_call = MagicMock()
        mock_tool_call.name = "database_write"
        mock_tool_call.arguments = '{"value": "x"}'
        mock_tool_call.tool_call_id = "call_server"

        response = MockLettaResponse()
        response.messages = [MockApprovalRequestMessage(tool_call=mock_tool_call)]
        response.usage = MagicMock(
            completion_tokens=1,
            prompt_tokens=2,
            total_tokens=3,
        )

        result = self.generator._build_message(
            "agent-123",
            response,
            client_tools=[
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {},
                }
            ],
        )

        self.assertEqual(result.meta["finish_reason"], "stop")
        self.assertNotIn("tool_calls", result.meta)
        self.assertIn("database_write", result.content)

    def test_run_calls_create_with_tools(self):
        # Test that run() calls create with streaming=True and client_tools
        self.generator.base_url = "http://test-url"
        self.generator.token = MagicMock()
        self.generator.token.resolve_value.return_value = "token"

        # Mock Letta client instance returned by constructor
        with patch("pipeline_wrapper.Letta") as MockLetta:
            mock_client = MockLetta.return_value
            mock_client.agents.messages.create.return_value = iter([])  # empty stream

            tools = [
                {
                    "type": "function",
                    "function": {"name": "test_tool", "parameters": {}},
                }
            ]

            self.generator.run(
                prompt="Use tool",
                agent_id="agent-123",
                streaming_callback=MagicMock(),
                tools=tools,
            )

            # Verify create was called with correct args
            mock_client.agents.messages.create.assert_called_once()
            call_kwargs = mock_client.agents.messages.create.call_args[1]
            self.assertEqual(call_kwargs["agent_id"], "agent-123")
            self.assertTrue(call_kwargs["streaming"])
            self.assertIsNotNone(call_kwargs["client_tools"])
            self.assertEqual(len(call_kwargs["client_tools"]), 1)
            self.assertEqual(call_kwargs["client_tools"][0]["name"], "test_tool")

    def test_run_reuses_client_tools_when_tool_result_has_no_tools(self):
        self.generator.base_url = "http://test-url"
        self.generator.token = MagicMock()
        self.generator.token.resolve_value.return_value = "token"
        self.generator._client_tools_by_call_id["call_abc"] = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {},
            }
        ]

        with patch("pipeline_wrapper.Letta") as MockLetta:
            mock_client = MockLetta.return_value
            mock_client.agents.messages.create.return_value = MockLettaResponse()
            mock_client.agents.messages.create.return_value.messages = []
            mock_client.agents.messages.create.return_value.usage = MagicMock(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            )

            self.generator.run(
                prompt=[
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_abc",
                        "content": "Sunny",
                    }
                ],
                agent_id="agent-123",
                tools=None,
            )

            mock_client.agents.messages.create.assert_called_once()
            call_kwargs = mock_client.agents.messages.create.call_args[1]
            self.assertEqual(call_kwargs["messages"][0]["type"], "approval")
            self.assertEqual(call_kwargs["client_tools"][0]["name"], "get_weather")


if __name__ == "__main__":
    unittest.main()
