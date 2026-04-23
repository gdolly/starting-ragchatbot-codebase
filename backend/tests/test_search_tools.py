import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:

    def _make_tool(self, search_results=None):
        mock_store = MagicMock()
        if search_results is not None:
            mock_store.search.return_value = search_results
        return CourseSearchTool(mock_store), mock_store

    def test_execute_returns_formatted_results(self):
        results = SearchResults(
            documents=["doc content"],
            metadata=[{"course_title": "Python 101", "lesson_number": 1}],
            distances=[0.5],
        )
        tool, _ = self._make_tool(results)
        output = tool.execute(query="test query")
        assert "Python 101" in output
        assert "doc content" in output

    def test_execute_returns_error_message_on_error(self):
        results = SearchResults.empty("No course found matching 'xyz'")
        tool, _ = self._make_tool(results)
        output = tool.execute(query="test", course_name="xyz")
        assert "No course found" in output

    def test_execute_returns_no_content_message_on_empty_results(self):
        results = SearchResults(documents=[], metadata=[], distances=[])
        tool, _ = self._make_tool(results)
        output = tool.execute(query="nonexistent topic")
        assert "No relevant content found" in output

    def test_execute_passes_filters_to_store(self):
        results = SearchResults(documents=[], metadata=[], distances=[])
        tool, mock_store = self._make_tool(results)
        tool.execute(query="q", course_name="Python", lesson_number=2)
        mock_store.search.assert_called_once_with(query="q", course_name="Python", lesson_number=2)

    def test_execute_tracks_sources(self):
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Python 101", "lesson_number": 3}],
            distances=[0.1],
        )
        tool, _ = self._make_tool(results)
        tool.execute(query="test")
        assert "Python 101 - Lesson 3" in tool.last_sources

    def test_get_tool_definition_has_required_fields(self):
        tool, _ = self._make_tool()
        definition = tool.get_tool_definition()
        assert definition["name"] == "search_course_content"
        assert "input_schema" in definition
        assert "query" in definition["input_schema"]["properties"]


class TestToolManager:

    def test_register_and_execute_tool(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool.execute.return_value = "result"
        manager.register_tool(mock_tool)

        result = manager.execute_tool("test_tool", query="hello")
        assert result == "result"

    def test_execute_unknown_tool_returns_not_found(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent")
        assert "not found" in result

    def test_register_tool_without_name_raises(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {}
        with pytest.raises(ValueError, match="must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "t1", "description": "desc"}
        manager.register_tool(mock_tool)
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "t1"

    def test_get_last_sources_returns_sources_from_tool(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "t1"}
        mock_tool.last_sources = ["Source A"]
        manager.register_tool(mock_tool)
        assert manager.get_last_sources() == ["Source A"]

    def test_get_last_sources_returns_empty_when_no_sources(self):
        manager = ToolManager()
        assert manager.get_last_sources() == []

    def test_reset_sources_clears_tool_sources(self):
        manager = ToolManager()
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"name": "t1"}
        mock_tool.last_sources = ["Source A"]
        manager.register_tool(mock_tool)
        manager.reset_sources()
        assert mock_tool.last_sources == []
