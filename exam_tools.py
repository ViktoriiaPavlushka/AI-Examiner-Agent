"""Функції (інструменти), які викликає AI-екзаменатор.

Замість бази даних використовуються JSON-файли:
    data/students.json — список студентів, які допущені до іспиту;
    data/topics.json   — список тем курсу NLP;
    data/exams.json    — журнал іспитів (створюється автоматично);
    results/*.json     — повний протокол кожного завершеного іспиту.
"""

from __future__ import annotations

import json
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
STUDENTS_FILE = DATA_DIR / "students.json"
TOPICS_FILE = DATA_DIR / "topics.json"
EXAMS_FILE = DATA_DIR / "exams.json"

TOPICS_PER_EXAM = (2, 3)  # іспит містить від 2 до 3 тем
NO_MORE_TOPICS = "NO_MORE_TOPICS"

_lock = threading.Lock()  # Streamlit обслуговує кількох користувачів у потоках


class Message(TypedDict):
    role: Literal["system", "user", "tool_call"]
    content: str
    datetime: str


class ExamError(Exception):
    """Помилка, яку бот має пояснити студенту (наприклад, студента не знайдено)."""


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _normalize_name(name: str) -> set[str]:
    return {part for part in name.lower().replace("’", "'").split() if part}


def load_students() -> list[dict]:
    return _load_json(STUDENTS_FILE, [])


def load_topics() -> list[str]:
    return _load_json(TOPICS_FILE, [])


def load_exams() -> list[dict]:
    return _load_json(EXAMS_FILE, [])


def find_student(email: str, name: str) -> dict:
    """Шукає студента за email і перевіряє, що ім'я збігається."""
    email = email.strip().lower()
    student = next(
        (s for s in load_students() if s["email"].lower() == email), None
    )
    if student is None:
        raise ExamError(
            f"Студента з email '{email}' не знайдено у списку допущених до іспиту."
        )
    # Порядок слів не важливий: "Вікторія Павлюх" == "Павлюх Вікторія".
    # Достатньо, щоб усі слова, які ввів студент, були в імені зі списку.
    given, expected = _normalize_name(name), _normalize_name(student["name"])
    if not given or not given <= expected:
        raise ExamError(
            "Email знайдено, але ім'я не збігається з даними у списку студентів."
        )
    return student


class ExamSession:
    """Стан одного іспиту. Методи — це інструменти, доступні агенту."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.exam_id: str | None = None
        self.student: dict | None = None
        self.topics: list[str] = []
        self.next_topic_index = 0
        self.current_topic: str | None = None
        self.topic_grades: dict[str, dict] = {}
        self.finished = False
        self.final_score: float | None = None
        self.feedback: str = ""

    # ---------- інструменти з завдання ----------

    def start_exam(self, email: str, name: str) -> list[str]:
        """Реєструє початок іспиту та повертає 2-3 випадкові теми."""
        if self.exam_id is not None:
            raise ExamError("Іспит уже розпочато. Продовжуйте з поточними темами.")

        student = find_student(email, name)
        all_topics = load_topics()
        k = min(self.rng.randint(*TOPICS_PER_EXAM), len(all_topics))
        topics = self.rng.sample(all_topics, k)

        with _lock:
            exams = load_exams()
            exam_id = f"{datetime.now():%Y%m%d-%H%M%S}-{len(exams) + 1}"
            exams.append(
                {
                    "exam_id": exam_id,
                    "email": student["email"],
                    "name": student["name"],
                    "group": student.get("group", ""),
                    "topics": topics,
                    "status": "in_progress",
                    "started_at": now_iso(),
                }
            )
            _save_json(EXAMS_FILE, exams)

        self.exam_id = exam_id
        self.student = student
        self.topics = topics
        return topics

    def get_next_topic(self) -> str:
        """Повертає наступну тему для обговорення або NO_MORE_TOPICS."""
        if self.exam_id is None:
            raise ExamError("Іспит ще не розпочато: спочатку викличте start_exam.")
        if self.next_topic_index >= len(self.topics):
            self.current_topic = None
            return NO_MORE_TOPICS
        self.current_topic = self.topics[self.next_topic_index]
        self.next_topic_index += 1
        return self.current_topic

    def end_exam(
        self,
        email: str,
        score: float,
        history: list[Message],
        feedback: str = "",
    ) -> None:
        """Записує результат іспиту: оцінку 0..10 і повну історію чату."""
        if self.exam_id is None or self.student is None:
            raise ExamError("Іспит ще не розпочато.")
        if self.finished:
            raise ExamError("Іспит уже завершено.")
        if email.strip().lower() != self.student["email"].lower():
            raise ExamError("Email не збігається з email студента, що складає іспит.")
        ungraded = [t for t in self.topics if t not in self.topic_grades]
        if self.next_topic_index < len(self.topics) or ungraded:
            raise ExamError(
                "Ще не всі теми обговорено/оцінено: " + "; ".join(ungraded)
            )

        score = round(max(0.0, min(10.0, float(score))), 1)
        finished_at = now_iso()

        with _lock:
            exams = load_exams()
            for exam in exams:
                if exam["exam_id"] == self.exam_id:
                    exam.update(
                        status="finished",
                        finished_at=finished_at,
                        score=score,
                        topic_grades=self.topic_grades,
                        feedback=feedback,
                    )
            _save_json(EXAMS_FILE, exams)

        _save_json(
            RESULTS_DIR / f"{self.exam_id}.json",
            {
                "exam_id": self.exam_id,
                "student": self.student,
                "topics": self.topics,
                "topic_grades": self.topic_grades,
                "score": score,
                "feedback": feedback,
                "finished_at": finished_at,
                "history": history,
            },
        )
        print(f"[end_exam] {self.student['name']} <{email}>: {score}/10")

        self.finished = True
        self.final_score = score
        self.feedback = feedback

    def save_history(self, history: list[Message]) -> None:
        """Оновлює історію в протоколі (дописує фінальний відгук бота)."""
        path = RESULTS_DIR / f"{self.exam_id}.json"
        record = _load_json(path, None)
        if record is not None:
            record["history"] = history
            _save_json(path, record)

    # ---------- додатковий інструмент ----------

    def grade_topic(self, topic: str, score: float, comment: str) -> str:
        """Фіксує проміжну оцінку за тему (0..10) з коротким коментарем."""
        if topic not in self.topics:
            # LLM іноді трохи перефразовує назву теми — шукаємо найближчу.
            matches = [t for t in self.topics if topic.lower() in t.lower()]
            if len(matches) != 1:
                raise ExamError(f"Тема '{topic}' не входить до цього іспиту.")
            topic = matches[0]
        if topic not in self.topics[: self.next_topic_index]:
            raise ExamError(f"Тему '{topic}' ще не було видано через get_next_topic.")
        score = round(max(0.0, min(10.0, float(score))), 1)
        self.topic_grades[topic] = {"score": score, "comment": comment}
        return f"Оцінку {score}/10 за тему '{topic}' збережено."
