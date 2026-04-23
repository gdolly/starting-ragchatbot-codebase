import os
import pytest
from document_processor import DocumentProcessor


class TestChunkText:

    def test_chunk_text_single_chunk(self):
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=0)
        text = "This is a short sentence. Another sentence here."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1

    def test_chunk_text_splits_long_text(self):
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=0)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = processor.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_with_overlap(self):
        processor = DocumentProcessor(chunk_size=60, chunk_overlap=30)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = processor.chunk_text(text)
        # With overlap, later chunks should share content with earlier ones
        assert len(chunks) >= 2

    def test_chunk_text_empty_string(self):
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=0)
        chunks = processor.chunk_text("")
        assert chunks == []

    def test_chunk_text_preserves_content(self):
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=0)
        text = "Important content here. Must be preserved exactly."
        chunks = processor.chunk_text(text)
        combined = " ".join(chunks)
        assert "Important content here" in combined
        assert "Must be preserved exactly" in combined


class TestReadFile:

    def test_read_file_returns_content(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("hello world", encoding="utf-8")
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=0)
        content = processor.read_file(str(file))
        assert content == "hello world"

    def test_read_file_handles_unicode(self, tmp_path):
        file = tmp_path / "unicode.txt"
        file.write_text("caf\u00e9 na\u00efve", encoding="utf-8")
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=0)
        content = processor.read_file(str(file))
        assert content == "caf\u00e9 na\u00efve"


class TestProcessCourseDocument:

    def test_process_extracts_course_metadata(self, tmp_path, course_document_content):
        file = tmp_path / "course.txt"
        file.write_text(course_document_content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(str(file))
        assert course.title == "Python Basics"
        assert course.course_link == "http://example.com/python"
        assert course.instructor == "Dr. Smith"

    def test_process_extracts_lessons(self, tmp_path, course_document_content):
        file = tmp_path / "course.txt"
        file.write_text(course_document_content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, _ = processor.process_course_document(str(file))
        assert len(course.lessons) == 2
        assert course.lessons[0].title == "Getting Started"
        assert course.lessons[0].lesson_number == 1
        assert course.lessons[1].title == "Variables"

    def test_process_creates_chunks(self, tmp_path, course_document_content):
        file = tmp_path / "course.txt"
        file.write_text(course_document_content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        _, chunks = processor.process_course_document(str(file))
        assert len(chunks) > 0
        assert all(c.course_title == "Python Basics" for c in chunks)

    def test_process_document_without_lessons(self, tmp_path):
        content = (
            "Course Title: Simple Course\n"
            "Course Link: http://example.com\n"
            "Course Instructor: Prof. X\n"
            "\n"
            "This is plain content without lesson markers. It should still be chunked."
        )
        file = tmp_path / "simple.txt"
        file.write_text(content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(str(file))
        assert course.title == "Simple Course"
        assert len(chunks) > 0

    def test_process_document_missing_metadata(self, tmp_path):
        content = "Just a title line\nSome body content here."
        file = tmp_path / "minimal.txt"
        file.write_text(content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, _ = processor.process_course_document(str(file))
        assert course.title == "Just a title line"

    def test_process_extracts_lesson_links(self, tmp_path, course_document_content):
        file = tmp_path / "course.txt"
        file.write_text(course_document_content, encoding="utf-8")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, _ = processor.process_course_document(str(file))
        assert course.lessons[0].lesson_link == "http://example.com/python/lesson1"
        assert course.lessons[1].lesson_link == "http://example.com/python/lesson2"
