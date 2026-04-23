import pytest
from unittest.mock import MagicMock, patch
from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_lesson():
    return Lesson(lesson_number=1, title="Introduction", lesson_link="http://example.com/lesson1")


@pytest.fixture
def sample_course(sample_lesson):
    return Course(
        title="Test Course",
        course_link="http://example.com/course",
        instructor="Dr. Test",
        lessons=[sample_lesson],
    )


@pytest.fixture
def sample_course_chunk():
    return CourseChunk(
        content="This is test content for a course chunk.",
        course_title="Test Course",
        lesson_number=1,
        chunk_index=0,
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    return [
        CourseChunk(content=f"Chunk {i} content", course_title=sample_course.title, lesson_number=1, chunk_index=i)
        for i in range(3)
    ]


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    return config


@pytest.fixture
def course_document_content():
    return (
        "Course Title: Python Basics\n"
        "Course Link: http://example.com/python\n"
        "Course Instructor: Dr. Smith\n"
        "\n"
        "Lesson 1: Getting Started\n"
        "Lesson Link: http://example.com/python/lesson1\n"
        "Python is a versatile programming language. "
        "It is widely used in web development, data science, and automation.\n"
        "Lesson 2: Variables\n"
        "Lesson Link: http://example.com/python/lesson2\n"
        "Variables store data values. In Python, you do not need to declare variable types.\n"
    )
