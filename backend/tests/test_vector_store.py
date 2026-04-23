import pytest
import tempfile
import shutil
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:

    def test_from_chroma_parses_results(self):
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
            "distances": [[0.1, 0.5]],
        }
        results = SearchResults.from_chroma(chroma_results)
        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"key": "val1"}, {"key": "val2"}]
        assert results.distances == [0.1, 0.5]

    def test_from_chroma_handles_empty(self):
        chroma_results = {"documents": [], "metadatas": [], "distances": []}
        results = SearchResults.from_chroma(chroma_results)
        assert results.documents == []

    def test_empty_creates_error_result(self):
        results = SearchResults.empty("some error")
        assert results.is_empty()
        assert results.error == "some error"

    def test_is_empty_returns_false_when_documents_exist(self):
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])
        assert not results.is_empty()


class TestVectorStore:

    @pytest.fixture
    def vector_store(self):
        tmp_dir = tempfile.mkdtemp()
        store = VectorStore(chroma_path=tmp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5)
        yield store
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.fixture
    def populated_store(self, vector_store):
        course = Course(
            title="Test Course",
            course_link="http://example.com",
            instructor="Dr. Test",
            lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="http://example.com/l1")],
        )
        chunks = [
            CourseChunk(content="Python is a programming language used for many applications.", course_title="Test Course", lesson_number=1, chunk_index=0),
            CourseChunk(content="Variables in Python store data values for later use.", course_title="Test Course", lesson_number=1, chunk_index=1),
            CourseChunk(content="Functions help organize code into reusable blocks.", course_title="Test Course", lesson_number=2, chunk_index=2),
        ]
        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)
        return vector_store

    def test_add_and_get_course_count(self, populated_store):
        assert populated_store.get_course_count() == 1

    def test_get_existing_course_titles(self, populated_store):
        titles = populated_store.get_existing_course_titles()
        assert "Test Course" in titles

    def test_search_returns_results(self, populated_store):
        results = populated_store.search(query="Python programming")
        assert not results.is_empty()
        assert len(results.documents) > 0

    def test_search_with_course_filter(self, populated_store):
        results = populated_store.search(query="Python", course_name="Test Course")
        assert not results.is_empty()
        assert all(m.get("course_title") == "Test Course" for m in results.metadata)

    def test_search_with_lesson_filter(self, populated_store):
        results = populated_store.search(query="Python", lesson_number=1)
        assert all(m.get("lesson_number") == 1 for m in results.metadata)

    def test_search_nonexistent_course_returns_error(self, populated_store):
        results = populated_store.search(query="test", course_name="Nonexistent Course XYZ")
        assert results.is_empty()
        assert results.error is not None

    def test_add_course_content_with_empty_list(self, vector_store):
        vector_store.add_course_content([])
        assert vector_store.get_course_count() == 0

    def test_clear_all_data(self, populated_store):
        populated_store.clear_all_data()
        assert populated_store.get_course_count() == 0
        assert populated_store.get_existing_course_titles() == []

    def test_get_course_link(self, populated_store):
        link = populated_store.get_course_link("Test Course")
        assert link == "http://example.com"

    def test_get_course_link_nonexistent(self, populated_store):
        link = populated_store.get_course_link("No Such Course")
        assert link is None

    def test_get_lesson_link(self, populated_store):
        link = populated_store.get_lesson_link("Test Course", 1)
        assert link == "http://example.com/l1"

    def test_get_all_courses_metadata(self, populated_store):
        metadata = populated_store.get_all_courses_metadata()
        assert len(metadata) == 1
        assert metadata[0]["title"] == "Test Course"
        assert "lessons" in metadata[0]

    def test_build_filter_both_params(self, vector_store):
        result = vector_store._build_filter("Course A", 1)
        assert result == {"$and": [{"course_title": "Course A"}, {"lesson_number": 1}]}

    def test_build_filter_course_only(self, vector_store):
        result = vector_store._build_filter("Course A", None)
        assert result == {"course_title": "Course A"}

    def test_build_filter_lesson_only(self, vector_store):
        result = vector_store._build_filter(None, 1)
        assert result == {"lesson_number": 1}

    def test_build_filter_no_params(self, vector_store):
        result = vector_store._build_filter(None, None)
        assert result is None
