import os
import pytest
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem


class TestRAGSystem:

    @pytest.fixture
    def rag(self, mock_config):
        with patch("rag_system.DocumentProcessor") as mock_dp, \
             patch("rag_system.VectorStore") as mock_vs, \
             patch("rag_system.AIGenerator") as mock_ai, \
             patch("rag_system.SessionManager") as mock_sm, \
             patch("rag_system.ToolManager") as mock_tm, \
             patch("rag_system.CourseSearchTool") as mock_cst:
            system = RAGSystem(mock_config)
            yield system

    def test_query_returns_response_and_sources(self, rag):
        rag.ai_generator.generate_response.return_value = "Here is the answer"
        rag.tool_manager.get_last_sources.return_value = ["Source A"]
        rag.session_manager.get_conversation_history.return_value = None

        response, sources = rag.query("What is Python?")
        assert response == "Here is the answer"
        assert sources == ["Source A"]

    def test_query_with_session_retrieves_history(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.tool_manager.get_last_sources.return_value = []
        rag.session_manager.get_conversation_history.return_value = "User: hi\nAssistant: hello"

        rag.query("follow up question", session_id="session_1")
        rag.session_manager.get_conversation_history.assert_called_once_with("session_1")

    def test_query_adds_exchange_to_session(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.tool_manager.get_last_sources.return_value = []
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("question", session_id="s1")
        rag.session_manager.add_exchange.assert_called_once()

    def test_query_resets_sources_after_retrieval(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.tool_manager.get_last_sources.return_value = ["src"]
        rag.session_manager.get_conversation_history.return_value = None

        rag.query("q")
        rag.tool_manager.reset_sources.assert_called_once()

    def test_get_course_analytics(self, rag):
        rag.vector_store.get_course_count.return_value = 3
        rag.vector_store.get_existing_course_titles.return_value = ["A", "B", "C"]

        expected = {"total_courses": 3, "course_titles": ["A", "B", "C"]}
        actual = rag.get_course_analytics()
        assert actual == expected

    def test_add_course_document_success(self, rag):
        mock_course = MagicMock()
        mock_chunks = [MagicMock(), MagicMock()]
        rag.document_processor.process_course_document.return_value = (mock_course, mock_chunks)

        course, count = rag.add_course_document("test.txt")
        assert course == mock_course
        assert count == 2
        rag.vector_store.add_course_metadata.assert_called_once_with(mock_course)
        rag.vector_store.add_course_content.assert_called_once_with(mock_chunks)

    def test_add_course_document_failure_returns_none(self, rag):
        rag.document_processor.process_course_document.side_effect = Exception("parse error")
        course, count = rag.add_course_document("bad.txt")
        assert course is None
        assert count == 0

    def test_add_course_folder_nonexistent_returns_zero(self, rag):
        courses, chunks = rag.add_course_folder("/nonexistent/path")
        assert courses == 0
        assert chunks == 0
