import pytest
from session_manager import SessionManager, Message


class TestSessionManager:

    def test_create_session_returns_unique_ids(self):
        manager = SessionManager()
        session_1 = manager.create_session()
        session_2 = manager.create_session()
        assert session_1 != session_2

    def test_create_session_initialises_empty_history(self):
        manager = SessionManager()
        session_id = manager.create_session()
        assert manager.sessions[session_id] == []

    def test_add_message_stores_message(self):
        manager = SessionManager()
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "hello")

        expected = Message(role="user", content="hello")
        actual = manager.sessions[session_id][0]
        assert actual == expected

    def test_add_message_creates_session_if_missing(self):
        manager = SessionManager()
        manager.add_message("new_session", "user", "hello")
        assert len(manager.sessions["new_session"]) == 1

    def test_add_exchange_stores_both_messages(self):
        manager = SessionManager()
        session_id = manager.create_session()
        manager.add_exchange(session_id, "question", "answer")
        assert len(manager.sessions[session_id]) == 2
        assert manager.sessions[session_id][0].role == "user"
        assert manager.sessions[session_id][1].role == "assistant"

    def test_get_conversation_history_returns_formatted_string(self):
        manager = SessionManager()
        session_id = manager.create_session()
        manager.add_exchange(session_id, "hello", "hi there")

        history = manager.get_conversation_history(session_id)
        assert "User: hello" in history
        assert "Assistant: hi there" in history

    @pytest.mark.parametrize(
        "session_id",
        [None, "nonexistent_session"],
    )
    def test_get_conversation_history_returns_none_for_invalid_session(self, session_id):
        manager = SessionManager()
        assert manager.get_conversation_history(session_id) is None

    def test_get_conversation_history_returns_none_for_empty_session(self):
        manager = SessionManager()
        session_id = manager.create_session()
        assert manager.get_conversation_history(session_id) is None

    def test_clear_session_removes_all_messages(self):
        manager = SessionManager()
        session_id = manager.create_session()
        manager.add_exchange(session_id, "q", "a")
        manager.clear_session(session_id)
        assert manager.sessions[session_id] == []

    def test_history_truncation_respects_max_history(self):
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()
        for i in range(5):
            manager.add_exchange(session_id, f"q{i}", f"a{i}")
        # max_history=2 means keep last 2*2=4 messages
        assert len(manager.sessions[session_id]) == 4
