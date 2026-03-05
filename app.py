import streamlit as st
from services.ingest import extract_text
from services.quiz_generator_ollama import generate_quiz_ollama
from services.grader import grade_quiz

st.set_page_config(page_title="QuizMaster", layout="wide")
st.title("QuizMaster")

# -------- Session State --------
if "text" not in st.session_state:
    st.session_state.text = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "quiz_language" not in st.session_state:
    st.session_state.quiz_language = "English"


# -------- Step D: Caching wrapper --------
@st.cache_data(show_spinner=False)
def cached_generate_quiz(text: str, n_questions: int, language: str):
    return generate_quiz_ollama(text, n_questions=n_questions, language=language)


def _reset_quiz_state(clear_cache: bool = False):
    st.session_state.quiz = []
    st.session_state.answers = {}
    st.session_state.submitted = False
    if clear_cache:
        cached_generate_quiz.clear()


# -------- 1) Upload content --------
st.header("1. Upload content")

uploaded = st.file_uploader("Upload a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
manual_text = st.text_area("Or paste your text here", height=180, placeholder="Paste educational content here...")

text = ""

if uploaded is not None:
    try:
        text = extract_text(uploaded.name, uploaded.getvalue())
        st.success(f"Text successfully extracted from: {uploaded.name}")
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
elif manual_text.strip():
    text = manual_text.strip()

if text.strip():
    st.session_state.text = text

    with st.expander("Preview extracted text"):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))
        st.caption(f"Total characters: {len(text)}")

    st.divider()

    # -------- 2) Generate quiz --------
    st.header("2. Generate quiz")

    # Language selector with on_change reset
    st.selectbox(
        "Quiz language",
        ["English", "Russian"],
        index=["English", "Russian"].index(st.session_state.quiz_language),
        key="quiz_language",
        on_change=_reset_quiz_state,  # reset quiz when language changes
    )

    n_questions = st.slider("Number of questions", min_value=3, max_value=15, value=5)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generate quiz", type="primary"):
            try:
                # Ensure we regenerate a new quiz for the selected language
                _reset_quiz_state(clear_cache=False)
                with st.spinner("Generating quiz using local Qwen 2.5 model..."):
                    st.session_state.quiz = cached_generate_quiz(
                        st.session_state.text,
                        n_questions,
                        st.session_state.quiz_language,
                    )
                st.success("Quiz generated successfully.")
            except Exception as e:
                st.error(f"Quiz generation failed: {e}")

    with col2:
        if st.button("Reset"):
            _reset_quiz_state(clear_cache=False)
            st.info("Session reset.")

    with col3:
        if st.button("Force regenerate"):
            _reset_quiz_state(clear_cache=True)
            st.info("Cache cleared. Click Generate quiz again.")

    # -------- 3) Solve the quiz --------
    if st.session_state.quiz:
        st.divider()
        st.header("3. Solve the quiz")

        for q in st.session_state.quiz:
            qid = q["id"]
            choice = st.radio(
                f"Question {qid}: {q['question']}",
                options=q["choices"],
                index=None,
                key=f"q_{qid}",
            )
            if choice is not None:
                st.session_state.answers[qid] = choice

        st.divider()

        if st.button("Submit answers"):
            st.session_state.submitted = True

        if st.session_state.submitted:
            result = grade_quiz(st.session_state.quiz, st.session_state.answers)

            st.header("Results")
            st.metric("Score", f"{result['correct']} / {result['total']}")
            st.caption(f"Accuracy: {result['score_pct']:.1f}%")

            st.divider()
            st.header("Answer review")

            for d in result["details"]:
                if d["is_correct"]:
                    st.success(f"Question {d['id']} — Correct")
                else:
                    st.error(f"Question {d['id']} — Incorrect")

                st.write("Your answer:", d["chosen"])
                st.write("Correct answer:", d["correct"])

                if d.get("explanation"):
                    st.caption(d["explanation"])

                src = d.get("source_chunks")
                if isinstance(src, list) and src:
                    st.caption(f"Source chunks: {src}")

else:
    st.info("Please upload a file or paste text to continue.")