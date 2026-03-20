import queue
import threading
import time

import streamlit as st

from services.grader import grade_quiz
from services.ingest import extract_text
from services.quiz_generator_ollama import GenerationStoppedError
from agents.orchestrator import run_quiz_pipeline

st.set_page_config(
    page_title="QuizMaster",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- Language ----------------------
LANGS = {
    "Русский": {
        "code": "Russian",
        "panel": "Панель",
        "format": "Формат",
        "model": "Модель",
        "upload_types": "Что можно загрузить",
        "modern_ui": "Современный AI-интерфейс для генерации тестов по вашим материалам.",
        "hero_badge": "✨ AI-powered • Русский/English интерфейс • Интерактивный тест",
        "hero_subtitle": "Загрузите учебный материал, сгенерируйте современный тест и сразу проверьте свои знания в более удобном и визуально приятном интерфейсе.",
        "chars_in_material": "Символов в материале",
        "questions_in_quiz": "Вопросов в тесте",
        "mode": "Режим",
        "step1": "1. Загрузка",
        "step2": "2. Генерация",
        "step3": "3. Тест",
        "step4": "4. Результаты",
        "upload_step": "Шаг 1",
        "generate_step": "Шаг 2",
        "quiz_step": "Шаг 3",
        "results_step": "Шаг 4",
        "upload_material": "Загрузка материала",
        "upload_file": "Загрузите файл",
        "upload_help": "Поддерживаются форматы PDF, DOCX и TXT",
        "paste_text": "Или вставьте текст вручную",
        "paste_placeholder": "Вставьте сюда конспект, лекцию, статью или любой учебный материал...",
        "hint": "Подсказка",
        "hint_text": "Лучше всего работают структурированные материалы: лекции, главы учебников, методички, статьи и конспекты.",
        "file_success": "Текст успешно извлечён из файла:",
        "file_error": "Не удалось извлечь текст:",
        "preview_material": "Предпросмотр материала",
        "show_text": "Показать текст",
        "chars": "Символов",
        "ready_processing": "Материал готов к обработке",
        "generate_quiz": "Генерация теста",
        "generate_btn": "⚡ Сгенерировать тест",
        "stop_btn": "⛔ Остановить генерацию",
        "reset_btn": "🔄 Сбросить ответы",
        "state_reset": "Состояние теста сброшено.",
        "loading_title": "🧠 Идёт генерация теста...",
        "loading_subtitle": "Модель анализирует текст, выделяет ключевые фрагменты и формирует вопросы на выбранном языке. При необходимости вы можете остановить процесс.",
        "quiz_running": "Прохождение теста",
        "choose_one": "Выберите один вариант ответа:",
        "question": "Вопрос",
        "submit_answers": "✅ Отправить ответы",
        "results": "Результаты",
        "correct_answers": "Правильных ответов",
        "total_questions": "Всего вопросов",
        "accuracy": "Точность",
        "answer_review": "Проверка ответов",
        "correct": "Верно",
        "incorrect": "Неверно",
        "your_answer": "Ваш ответ:",
        "correct_answer": "Правильный ответ:",
        "no_answer": "Нет ответа",
        "source_chunks": "Фрагменты источника:",
        "ready_block_title": "👋 Готово к работе",
        "ready_block_text": "Загрузите файл или вставьте текст, чтобы перейти к генерации теста.",
        "generation_success": "Тест успешно сгенерирован.",
        "generation_stopped": "Генерация остановлена пользователем.",
        "generation_error": "Ошибка при генерации теста:",
        "generation_no_result": "Генерация завершилась без результата.",
        "model_failed": "Модель не смогла сгенерировать тест.",
        "question_count": "Количество вопросов",
        "lang_label": "Язык интерфейса и теста",
        "russian_only_mode": "Русский / English",
    },
    "English": {
        "code": "English",
        "panel": "Panel",
        "format": "Format",
        "model": "Model",
        "upload_types": "What you can upload",
        "modern_ui": "Modern AI interface for generating quizzes from your materials.",
        "hero_badge": "✨ AI-powered • Russian/English interface • Interactive quiz",
        "hero_subtitle": "Upload study material, generate a modern quiz, and check your knowledge in a cleaner and more interactive interface.",
        "chars_in_material": "Characters in material",
        "questions_in_quiz": "Questions in quiz",
        "mode": "Mode",
        "step1": "1. Upload",
        "step2": "2. Generate",
        "step3": "3. Quiz",
        "step4": "4. Results",
        "upload_step": "Step 1",
        "generate_step": "Step 2",
        "quiz_step": "Step 3",
        "results_step": "Step 4",
        "upload_material": "Upload material",
        "upload_file": "Upload file",
        "upload_help": "Supported formats: PDF, DOCX and TXT",
        "paste_text": "Or paste text manually",
        "paste_placeholder": "Paste lecture notes, an article, or any study material here...",
        "hint": "Hint",
        "hint_text": "Structured materials work best: lectures, textbook chapters, guides, articles, and notes.",
        "file_success": "Text extracted successfully from file:",
        "file_error": "Failed to extract text:",
        "preview_material": "Material preview",
        "show_text": "Show text",
        "chars": "Characters",
        "ready_processing": "Material is ready for processing",
        "generate_quiz": "Quiz generation",
        "generate_btn": "⚡ Generate quiz",
        "stop_btn": "⛔ Stop generation",
        "reset_btn": "🔄 Reset answers",
        "state_reset": "Quiz state has been reset.",
        "loading_title": "🧠 Quiz generation in progress...",
        "loading_subtitle": "The model is analyzing the text, selecting key fragments, and generating questions in the selected language. You can stop the process if needed.",
        "quiz_running": "Take the quiz",
        "choose_one": "Choose one answer:",
        "question": "Question",
        "submit_answers": "✅ Submit answers",
        "results": "Results",
        "correct_answers": "Correct answers",
        "total_questions": "Total questions",
        "accuracy": "Accuracy",
        "answer_review": "Answer review",
        "correct": "Correct",
        "incorrect": "Incorrect",
        "your_answer": "Your answer:",
        "correct_answer": "Correct answer:",
        "no_answer": "No answer",
        "source_chunks": "Source chunks:",
        "ready_block_title": "👋 Ready to go",
        "ready_block_text": "Upload a file or paste text to start generating a quiz.",
        "generation_success": "Quiz generated successfully.",
        "generation_stopped": "Generation stopped by user.",
        "generation_error": "Quiz generation error:",
        "generation_no_result": "Generation finished without a result.",
        "model_failed": "The model could not generate a quiz.",
        "question_count": "Number of questions",
        "lang_label": "Interface and quiz language",
        "russian_only_mode": "Russian / English",
    },
}

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "Русский"

T = LANGS[st.session_state.ui_language]

# ---------------------- Premium CSS ----------------------
st.markdown(
    """
<style>
:root {
    --bg: #0b1020;
    --card: rgba(21, 28, 52, 0.78);
    --card-2: rgba(17, 23, 43, 0.92);
    --border: rgba(255,255,255,0.08);
    --soft-border: rgba(255,255,255,0.05);
    --text: #f5f7ff;
    --muted: #a9b1d6;
    --accent: #6d7cff;
    --accent-2: #8b5cf6;
    --success-bg: rgba(34,197,94,0.12);
    --success-border: rgba(34,197,94,0.35);
    --error-bg: rgba(239,68,68,0.12);
    --error-border: rgba(239,68,68,0.35);
    --warning-bg: rgba(245,158,11,0.12);
    --warning-border: rgba(245,158,11,0.35);
    --shadow: 0 10px 35px rgba(0,0,0,0.28);
}
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(109,124,255,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(139,92,246,0.14), transparent 22%),
        linear-gradient(180deg, #070b17 0%, #0b1020 45%, #0b1020 100%);
    color: var(--text);
}
.block-container {
    max-width: 1220px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 28, 0.95);
    border-right: 1px solid var(--soft-border);
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
.hero {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(135deg, rgba(109,124,255,0.16), rgba(139,92,246,0.10)),
        rgba(18, 24, 45, 0.82);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 1.7rem 1.7rem 1.5rem 1.7rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
    animation: fadeUp 0.55s ease-out;
}
.hero::before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -70px;
    top: -80px;
    background: radial-gradient(circle, rgba(109,124,255,0.32), transparent 62%);
    filter: blur(10px);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    background: rgba(109,124,255,0.16);
    border: 1px solid rgba(109,124,255,0.26);
    color: #dbe1ff;
    font-size: 0.86rem;
    font-weight: 600;
    margin-bottom: 0.95rem;
}
.hero-title {
    font-size: 2.35rem;
    line-height: 1.1;
    font-weight: 850;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.03em;
    color: var(--text);
}
.hero-subtitle {
    font-size: 1.04rem;
    line-height: 1.6;
    color: var(--muted);
    max-width: 760px;
    margin-bottom: 1rem;
}
.hero-stats {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    margin-top: 0.7rem;
}
.hero-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--soft-border);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    min-width: 170px;
    backdrop-filter: blur(10px);
}
.hero-stat-label {
    color: var(--muted);
    font-size: 0.8rem;
    margin-bottom: 0.2rem;
}
.hero-stat-value {
    color: var(--text);
    font-size: 1.05rem;
    font-weight: 700;
}
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.15rem 1.15rem 1rem 1.15rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
    animation: fadeUp 0.45s ease-out;
}
.quiz-card {
    background: var(--card-2);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.05rem 1.05rem 0.95rem 1.05rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: fadeUp 0.45s ease-out;
}
.quiz-card:hover {
    transform: translateY(-2px);
    border-color: rgba(109,124,255,0.22);
    box-shadow: 0 14px 42px rgba(0,0,0,0.34);
}
.step-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 0.2rem 0 1rem 0;
}
.step-pill {
    padding: 0.48rem 0.9rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.035);
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
.step-pill.active {
    background: linear-gradient(135deg, rgba(109,124,255,0.22), rgba(139,92,246,0.18));
    color: #eef1ff;
    border-color: rgba(109,124,255,0.35);
    box-shadow: 0 0 0 1px rgba(109,124,255,0.10) inset;
}
.step-badge {
    display: inline-block;
    padding: 0.34rem 0.78rem;
    border-radius: 999px;
    background: rgba(109,124,255,0.14);
    border: 1px solid rgba(109,124,255,0.22);
    color: #d8dfff;
    font-weight: 700;
    font-size: 0.84rem;
    margin-bottom: 0.75rem;
}
.small-muted {
    color: var(--muted);
    font-size: 0.93rem;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--soft-border);
    border-radius: 20px;
    padding: 0.75rem;
}
.status-box {
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
}
.status-box.success {
    background: var(--success-bg);
    border-color: var(--success-border);
}
.status-box.error {
    background: var(--error-bg);
    border-color: var(--error-border);
}
.status-box.warning {
    background: var(--warning-bg);
    border-color: var(--warning-border);
}
.result-good {
    padding: 0.95rem 1rem;
    border-radius: 18px;
    background: var(--success-bg);
    border: 1px solid var(--success-border);
    margin-bottom: 0.7rem;
}
.result-bad {
    padding: 0.95rem 1rem;
    border-radius: 18px;
    background: var(--error-bg);
    border: 1px solid var(--error-border);
    margin-bottom: 0.7rem;
}
.loading-box {
    background: linear-gradient(135deg, rgba(109,124,255,0.13), rgba(139,92,246,0.12));
    border: 1px solid rgba(109,124,255,0.24);
    border-radius: 22px;
    padding: 1rem 1.1rem;
    margin-top: 0.7rem;
    animation: pulseGlow 2s infinite;
}
.loading-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.35rem;
    color: var(--text);
}
.loading-subtitle {
    color: var(--muted);
    font-size: 0.94rem;
}
.answer-selected {
    background: linear-gradient(135deg, rgba(109,124,255,0.22), rgba(139,92,246,0.16));
    border: 1px solid rgba(109,124,255,0.48);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    color: #eef1ff;
    font-weight: 600;
    line-height: 1.45;
    box-shadow: 0 8px 24px rgba(109,124,255,0.12);
    word-break: break-word;
    white-space: normal;
}
.answer-unselected {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    color: #e8ebff;
    line-height: 1.45;
    word-break: break-word;
    white-space: normal;
}
.answer-label {
    font-size: 0.96rem;
    line-height: 1.5;
    margin: 0;
    white-space: normal;
    word-break: break-word;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--soft-border);
    border-radius: 20px;
    padding: 0.85rem;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}
.stButton > button {
    width: 100%;
    border-radius: 16px;
    min-height: 48px;
    border: 1px solid var(--soft-border);
    background: rgba(255,255,255,0.04);
    color: var(--text);
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: none;
}
.stButton > button:hover {
    border-color: rgba(109,124,255,0.45);
    background: rgba(109,124,255,0.09);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: white;
    border: none;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.05);
    transform: translateY(-1px);
}
.stTextArea textarea,
.stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    color: var(--text) !important;
    border-radius: 16px !important;
}
.stTextArea label,
.stFileUploader label,
.stRadio label,
.stSlider label,
.stSelectbox label {
    color: var(--text) !important;
}
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border-radius: 18px;
    border: 1px solid var(--soft-border);
}
div[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    border: 1px dashed rgba(255,255,255,0.14);
}
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}
hr {
    border: none;
    border-top: 1px solid var(--soft-border);
    margin: 1rem 0 !important;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 rgba(109,124,255,0.0); }
    50% { box-shadow: 0 0 26px rgba(109,124,255,0.12); }
    100% { box-shadow: 0 0 0 rgba(109,124,255,0.0); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- Session state ----------------------
defaults = {
    "text": "",
    "file_bytes": None,
    "file_name": None,
    "quiz": [],
    "answers": {},
    "submitted": False,
    "is_generating": False,
    "generation_thread": None,
    "generation_queue": None,
    "generation_stop_event": None,
    "generation_message": "",
    "generation_status": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _reset_quiz_state():
    st.session_state.quiz = []
    st.session_state.answers = {}
    st.session_state.submitted = False


def _clear_generation_state():
    st.session_state.is_generating = False
    st.session_state.generation_thread = None
    st.session_state.generation_queue = None
    st.session_state.generation_stop_event = None


def _set_active_text(text: str, file_bytes=None, file_name=None):
    st.session_state.text = text
    st.session_state.file_bytes = file_bytes
    st.session_state.file_name = file_name


def _start_generation(text: str, n_questions: int, language_code: str, file_bytes=None):
    _reset_quiz_state()
    st.session_state.generation_status = None
    st.session_state.generation_message = ""

    result_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    _file_bytes = file_bytes

    def worker():
        try:
            quiz = run_quiz_pipeline(
                text=text,
                n_questions=n_questions,
                language=language_code,
                stop_event=stop_event,
                file_bytes=_file_bytes,
            )

            if stop_event.is_set():
                result_queue.put(("stopped", T["generation_stopped"]))
                return

            if not quiz:
                result_queue.put(("error", T["model_failed"]))
                return

            result_queue.put(("success", quiz))

        except GenerationStoppedError:
            result_queue.put(("stopped", T["generation_stopped"]))
        except Exception as e:
            result_queue.put(("error", str(e)))

    thread = threading.Thread(target=worker, daemon=True)
    st.session_state.generation_queue = result_queue
    st.session_state.generation_stop_event = stop_event
    st.session_state.generation_thread = thread
    st.session_state.is_generating = True
    thread.start()


def _poll_generation():
    if not st.session_state.is_generating:
        return

    q = st.session_state.generation_queue
    thread = st.session_state.generation_thread

    if q is not None and not q.empty():
        status, payload = q.get()

        if status == "success":
            st.session_state.quiz = payload
            st.session_state.generation_status = "success"
            st.session_state.generation_message = T["generation_success"]
        elif status == "stopped":
            st.session_state.generation_status = "stopped"
            st.session_state.generation_message = payload
        else:
            st.session_state.generation_status = "error"
            st.session_state.generation_message = f"{T['generation_error']} {payload}"

        _clear_generation_state()
        return

    if thread is not None and not thread.is_alive():
        st.session_state.generation_status = "error"
        st.session_state.generation_message = T["generation_no_result"]
        _clear_generation_state()


def _render_step_pills(active_step: int):
    labels = [T["step1"], T["step2"], T["step3"], T["step4"]]
    html = "<div class='step-row'>"
    for i, label in enumerate(labels, start=1):
        cls = "step-pill active" if i == active_step else "step-pill"
        html += f"<div class='{cls}'>{label}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_answer_cards(question_id: int, options: list[str]):
    selected = st.session_state.answers.get(question_id)
    letters = ["A", "B", "C", "D"]

    for idx, option in enumerate(options):
        is_selected = selected == option
        letter = letters[idx] if idx < len(letters) else str(idx + 1)

        col1, col2, col3 = st.columns([1.2, 1.2, 12], vertical_alignment="center")

        with col1:
            st.markdown(
                f"""
                <div style="
                    width: 38px;
                    height: 38px;
                    border-radius: 999px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 700;
                    background: {'rgba(109,124,255,0.24)' if is_selected else 'rgba(255,255,255,0.06)'};
                    border: 1px solid {'rgba(109,124,255,0.45)' if is_selected else 'rgba(255,255,255,0.08)'};
                    color: #eef1ff;
                    margin-top: 4px;
                ">
                    {letter}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if st.button(
                "✓" if is_selected else "○",
                key=f"answer_btn_{question_id}_{idx}",
                use_container_width=True,
            ):
                st.session_state.answers[question_id] = option
                st.rerun()

        with col3:
            css_class = "answer-selected" if is_selected else "answer-unselected"
            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="answer-label">{option}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


_poll_generation()
T = LANGS[st.session_state.ui_language]

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.title("🧠 QuizMaster")
    st.markdown(f"### {T['panel']}")

    lang_choice = st.selectbox(
        T["lang_label"],
        options=list(LANGS.keys()),
        index=list(LANGS.keys()).index(st.session_state.ui_language),
    )
    if lang_choice != st.session_state.ui_language:
        st.session_state.ui_language = lang_choice
        st.rerun()

    T = LANGS[st.session_state.ui_language]

    st.markdown(f"**{T['format']}:** {T['russian_only_mode']}")
    st.markdown("**{0}:** Ollama / Qwen".format(T["model"]))
    st.divider()

    n_questions = st.slider(T["question_count"], min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown(f"### {T['upload_types']}")
    st.markdown("• PDF\n• DOCX\n• TXT\n• pasted text")
    st.divider()
    st.markdown(
        f"<div class='small-muted'>{T['modern_ui']}</div>",
        unsafe_allow_html=True,
    )

# ---------------------- Hero ----------------------
text_len = len(st.session_state.text) if st.session_state.text else 0
quiz_len = len(st.session_state.quiz) if st.session_state.quiz else 0

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge">{T['hero_badge']}</div>
        <div class="hero-title">QuizMaster</div>
        <div class="hero-subtitle">{T['hero_subtitle']}</div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-label">{T['chars_in_material']}</div>
                <div class="hero-stat-value">{text_len}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">{T['questions_in_quiz']}</div>
                <div class="hero-stat-value">{quiz_len}</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-label">{T['mode']}</div>
                <div class="hero-stat-value">{st.session_state.ui_language}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Status message ----------------------
if st.session_state.generation_status == "success":
    st.markdown(
        f"<div class='status-box success'>✅ {st.session_state.generation_message}</div>",
        unsafe_allow_html=True,
    )
elif st.session_state.generation_status == "stopped":
    st.markdown(
        f"<div class='status-box warning'>⛔ {st.session_state.generation_message}</div>",
        unsafe_allow_html=True,
    )
elif st.session_state.generation_status == "error":
    st.markdown(
        f"<div class='status-box error'>❌ {st.session_state.generation_message}</div>",
        unsafe_allow_html=True,
    )

# ---------------------- Step detect ----------------------
current_step = 1
if st.session_state.text.strip():
    current_step = 2
if st.session_state.quiz:
    current_step = 3
if st.session_state.submitted:
    current_step = 4

_render_step_pills(current_step)

# ---------------------- Step 1 ----------------------
st.markdown(f"<div class='step-badge'>{T['upload_step']}</div>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader(T["upload_material"])

col_upload, col_info = st.columns([2.2, 1])

with col_upload:
    uploaded = st.file_uploader(
        T["upload_file"],
        type=["pdf", "docx", "txt"],
        help=T["upload_help"],
    )

    manual_text = st.text_area(
        T["paste_text"],
        height=220,
        placeholder=T["paste_placeholder"],
    )

with col_info:
    st.markdown(
        f"""
<div class='metric-card'>
    <div style='font-weight:700; margin-bottom:0.45rem;'>📄 {T['hint']}</div>
    <div class='small-muted'>{T['hint_text']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

text = ""
file_bytes = None
active_file_name = None

# Priority: uploaded file > pasted text
if uploaded is not None:
    try:
        raw_bytes = uploaded.getvalue()

        # Reuse cached extraction ONLY if both filename and file bytes are identical.
        same_name = st.session_state.file_name == uploaded.name
        same_bytes = st.session_state.file_bytes == raw_bytes

        if same_name and same_bytes and st.session_state.text:
            text = st.session_state.text
            file_bytes = st.session_state.file_bytes
            active_file_name = st.session_state.file_name
        else:
            text = extract_text(uploaded.name, raw_bytes)
            file_bytes = raw_bytes
            active_file_name = uploaded.name

        st.success(f"{T['file_success']} {uploaded.name}")
    except Exception as e:
        st.error(f"{T['file_error']} {e}")

elif manual_text.strip():
    text = manual_text.strip()
    file_bytes = None
    active_file_name = None

if text.strip():
    _set_active_text(text=text, file_bytes=file_bytes, file_name=active_file_name)

st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.text.strip():
    progress_value = 0.30
    if st.session_state.is_generating:
        progress_value = 0.52
    elif st.session_state.quiz:
        progress_value = 0.76
    if st.session_state.submitted:
        progress_value = 1.0

    st.progress(progress_value)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader(T["preview_material"])

    preview_col1, preview_col2 = st.columns([3, 1])

    with preview_col1:
        with st.expander(T["show_text"], expanded=False):
            preview_text = st.session_state.text[:2500]
            if len(st.session_state.text) > 2500:
                preview_text += "..."
            st.write(preview_text)

    with preview_col2:
        st.metric(T["chars"], f"{len(st.session_state.text)}")
        st.caption(T["ready_processing"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='step-badge'>{T['generate_step']}</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader(T["generate_quiz"])

    c1, c2, c3 = st.columns(3)

    with c1:
        generate_clicked = st.button(
            T["generate_btn"],
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_generating,
        )

    with c2:
        stop_clicked = st.button(
            T["stop_btn"],
            use_container_width=True,
            disabled=not st.session_state.is_generating,
        )

    with c3:
        reset_clicked = st.button(
            T["reset_btn"],
            use_container_width=True,
            disabled=st.session_state.is_generating,
        )

    if generate_clicked:
        _start_generation(
            st.session_state.text,
            n_questions,
            T["code"],
            file_bytes=st.session_state.file_bytes,
        )
        st.rerun()

    if stop_clicked and st.session_state.generation_stop_event is not None:
        st.session_state.generation_stop_event.set()
        st.rerun()

    if reset_clicked:
        _reset_quiz_state()
        st.session_state.generation_status = None
        st.session_state.generation_message = ""
        st.info(T["state_reset"])

    if st.session_state.is_generating:
        st.markdown(
            f"""
            <div class='loading-box'>
                <div class='loading-title'>{T['loading_title']}</div>
                <div class='loading-subtitle'>{T['loading_subtitle']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.quiz:
        st.markdown(f"<div class='step-badge'>{T['quiz_step']}</div>", unsafe_allow_html=True)
        st.subheader(T["quiz_running"])

        for q in st.session_state.quiz:
            qid = q["id"]

            st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
            st.markdown(f"### {T['question']} {qid}")
            st.write(q["question"])
            st.write(f"**{T['choose_one']}**")
            render_answer_cards(qid, q["choices"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        submit_col1, _ = st.columns([1, 3])
        with submit_col1:
            if st.button(T["submit_answers"], type="primary", use_container_width=True):
                st.session_state.submitted = True

        if st.session_state.submitted:
            result = grade_quiz(st.session_state.quiz, st.session_state.answers)

            st.markdown(f"<div class='step-badge'>{T['results_step']}</div>", unsafe_allow_html=True)
            st.subheader(T["results"])

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(T["correct_answers"], result["correct"])
            with m2:
                st.metric(T["total_questions"], result["total"])
            with m3:
                st.metric(T["accuracy"], f"{result['score_pct']:.1f}%")

            st.divider()
            st.subheader(T["answer_review"])

            for d in result["details"]:
                if d["is_correct"]:
                    st.markdown(
                        f"""
                        <div class='result-good'>
                            <b>✅ {T['question']} {d['id']} — {T['correct']}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='result-bad'>
                            <b>❌ {T['question']} {d['id']} — {T['incorrect']}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write(f"**{T['your_answer']}**", d["chosen"] if d["chosen"] else T["no_answer"])
                st.write(f"**{T['correct_answer']}**", d["correct"])

                if d.get("explanation"):
                    st.caption(d["explanation"])

                src = d.get("source_chunks")
                if isinstance(src, list) and src:
                    st.caption(f"{T['source_chunks']} {src}")

                st.markdown("---")

else:
    st.markdown(
        f"""
        <div class='section-card'>
            <div style='font-size:1.05rem; font-weight:700; margin-bottom:0.35rem;'>{T['ready_block_title']}</div>
            <div class='small-muted'>{T['ready_block_text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.is_generating:
    time.sleep(0.5)
    st.rerun()