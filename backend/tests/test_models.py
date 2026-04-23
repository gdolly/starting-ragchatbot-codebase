import pytest
from models import Lesson, Course, CourseChunk


class TestLesson:

    def test_create_lesson_with_all_fields(self):
        expected = Lesson(lesson_number=1, title="Intro", lesson_link="http://example.com")
        actual = Lesson(lesson_number=1, title="Intro", lesson_link="http://example.com")
        assert actual == expected

    def test_create_lesson_without_optional_link(self):
        lesson = Lesson(lesson_number=1, title="Intro")
        assert lesson.lesson_link is None

    @pytest.mark.parametrize(
        "lesson_number, title",
        [
            (0, "Zero-indexed Lesson"),
            (99, "Large Lesson Number"),
        ],
    )
    def test_create_lesson_with_various_numbers(self, lesson_number, title):
        lesson = Lesson(lesson_number=lesson_number, title=title)
        assert lesson.lesson_number == lesson_number
        assert lesson.title == title


class TestCourse:

    def test_create_course_with_all_fields(self, sample_lesson):
        expected = Course(
            title="Test Course",
            course_link="http://example.com/course",
            instructor="Dr. Test",
            lessons=[sample_lesson],
        )
        actual = Course(
            title="Test Course",
            course_link="http://example.com/course",
            instructor="Dr. Test",
            lessons=[sample_lesson],
        )
        assert actual == expected

    def test_create_course_with_defaults(self):
        course = Course(title="Minimal Course")
        assert course.course_link is None
        assert course.instructor is None
        assert course.lessons == []

    def test_course_lessons_list_is_mutable(self, sample_lesson):
        course = Course(title="Test")
        course.lessons.append(sample_lesson)
        assert len(course.lessons) == 1


class TestCourseChunk:

    def test_create_chunk_with_all_fields(self):
        expected = CourseChunk(content="text", course_title="Course", lesson_number=1, chunk_index=0)
        actual = CourseChunk(content="text", course_title="Course", lesson_number=1, chunk_index=0)
        assert actual == expected

    def test_create_chunk_without_lesson_number(self):
        chunk = CourseChunk(content="text", course_title="Course", chunk_index=0)
        assert chunk.lesson_number is None
