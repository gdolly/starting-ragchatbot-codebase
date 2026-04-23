import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.RAGSystem") as mock_rag_cls:
        mock_rag = MagicMock()
        mock_rag_cls.return_value = mock_rag
        mock_rag.session_manager.create_session.return_value = "session_1"
        mock_rag.query.return_value = ("This is the answer", ["Source A"])
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }

        from app import app
        app_client = TestClient(app, raise_server_exceptions=False)
        yield app_client, mock_rag


class TestQueryEndpoint:

    def test_post_query_success(self, client):
        app_client, mock_rag = client
        response = app_client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "This is the answer"
        assert body["sources"] == ["Source A"]
        assert body["session_id"] == "session_1"

    def test_post_query_with_session_id(self, client):
        app_client, mock_rag = client
        response = app_client.post(
            "/api/query", json={"query": "follow up", "session_id": "existing_session"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "existing_session"

    def test_post_query_missing_query_field(self, client):
        app_client, _ = client
        response = app_client.post("/api/query", json={})
        assert response.status_code == 422

    def test_post_query_empty_body(self, client):
        app_client, _ = client
        response = app_client.post("/api/query")
        assert response.status_code == 422

    def test_post_query_server_error(self, client):
        app_client, mock_rag = client
        mock_rag.query.side_effect = Exception("internal error")
        mock_rag.session_manager.create_session.return_value = "s1"
        response = app_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 500
        assert "internal error" in response.json()["detail"]


class TestCoursesEndpoint:

    def test_get_courses_success(self, client):
        app_client, _ = client
        response = app_client.get("/api/courses")
        assert response.status_code == 200
        body = response.json()
        assert body["total_courses"] == 2
        assert body["course_titles"] == ["Course A", "Course B"]

    def test_get_courses_server_error(self, client):
        app_client, mock_rag = client
        mock_rag.get_course_analytics.side_effect = Exception("db error")
        response = app_client.get("/api/courses")
        assert response.status_code == 500
        assert "db error" in response.json()["detail"]
