import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


class TestAIGenerator:

    def _make_generator(self):
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator, mock_client

    def _make_text_response(self, text="test response"):
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = text
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        return mock_response

    def _make_tool_use_response(self, tool_name="search_course_content", tool_input=None):
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = tool_name
        mock_tool_block.input = tool_input or {"query": "test"}
        mock_tool_block.id = "tool_123"

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_response.stop_reason = "tool_use"
        return mock_response

    def test_generate_response_returns_text(self):
        generator, mock_client = self._make_generator()
        mock_client.messages.create.return_value = self._make_text_response("Hello!")
        result = generator.generate_response(query="Hi")
        assert result == "Hello!"

    def test_generate_response_includes_conversation_history(self):
        generator, mock_client = self._make_generator()
        mock_client.messages.create.return_value = self._make_text_response()
        generator.generate_response(query="Hi", conversation_history="User: hello\nAssistant: hi")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation" in call_kwargs["system"]

    def test_generate_response_without_history(self):
        generator, mock_client = self._make_generator()
        mock_client.messages.create.return_value = self._make_text_response()
        generator.generate_response(query="Hi")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation" not in call_kwargs["system"]

    def test_generate_response_passes_tools(self):
        generator, mock_client = self._make_generator()
        mock_client.messages.create.return_value = self._make_text_response()
        tools = [{"name": "test_tool", "input_schema": {}}]
        generator.generate_response(query="Hi", tools=tools)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_handles_tool_use(self):
        generator, mock_client = self._make_generator()
        tool_response = self._make_tool_use_response()
        final_response = self._make_text_response("Final answer")
        mock_client.messages.create.side_effect = [tool_response, final_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "search results"

        result = generator.generate_response(
            query="Search for Python",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )
        assert result == "Final answer"
        mock_tool_manager.execute_tool.assert_called_once()

    def test_generate_response_tool_use_without_manager_returns_text(self):
        generator, mock_client = self._make_generator()
        # When stop_reason is tool_use but no tool_manager, falls through to text
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "I need to search"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.stop_reason = "tool_use"
        mock_client.messages.create.return_value = mock_response
        result = generator.generate_response(query="search")
        assert result == "I need to search"
