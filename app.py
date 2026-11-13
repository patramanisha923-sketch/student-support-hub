import io
import hashlib
import hmac
import re
import secrets
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "student_dropout_model_9_features.pkl"
FAQ_DATABASE_PATH = APP_DIR / "student_life_faqs.sqlite3"
MAX_PDF_UPLOAD_MB = 500
PASSWORD_HASH_ITERATIONS = 600_000


class PDFProcessingError(Exception):
    pass


class PDFPasswordError(PDFProcessingError):
    pass


def extract_pdf_text(uploaded_pdf, password="", use_ocr=True):
    try:
        import fitz
    except ImportError as error:
        raise PDFProcessingError(
            "PDF processing support is missing. Install the project packages "
            "from requirements.txt and restart the app."
        ) from error

    pdf_bytes = uploaded_pdf.getvalue()
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise PDFProcessingError(
            "This file does not look like a PDF. Choose a valid .pdf file."
        )

    tesseract = None
    ocr_pages = 0
    ocr_unavailable = False
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.FileDataError as error:
        raise PDFProcessingError(
            "The PDF is damaged or uses a format this PDF reader cannot open."
        ) from error

    with document:
        if document.needs_pass and not document.authenticate(password):
            raise PDFPasswordError(
                "This PDF needs a password. Enter the correct password and "
                "generate the quiz again."
            )

        page_texts = []
        for page in document:
            page_text = page.get_text("text", sort=True)
            if len(page_text.strip()) < 40:
                ocr_pages += 1
                if use_ocr:
                    if tesseract is None and not ocr_unavailable:
                        try:
                            import pytesseract
                            from PIL import Image

                            pytesseract.get_tesseract_version()
                            tesseract = (pytesseract, Image)
                        except ImportError as error:
                            raise PDFProcessingError(
                                "OCR support is missing. Install the project "
                                "packages from requirements.txt and restart "
                                "the app."
                            ) from error
                        except pytesseract.TesseractNotFoundError:
                            ocr_unavailable = True

                    if tesseract is not None:
                        pytesseract, image_module = tesseract
                        pixmap = page.get_pixmap(dpi=150, alpha=False)
                        with image_module.open(
                            io.BytesIO(pixmap.tobytes("png"))
                        ) as page_image:
                            try:
                                page_text = pytesseract.image_to_string(
                                    page_image,
                                    lang="eng",
                                )
                            except pytesseract.TesseractError as error:
                                raise PDFProcessingError(
                                    "OCR could not read this scanned page. "
                                    "Check that Tesseract and its English "
                                    "language data are installed."
                                ) from error
            if page_text.strip():
                page_texts.append(page_text)

    return "\n".join(page_texts), ocr_pages, ocr_unavailable

STOP_WORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "could",
    "does",
    "each",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def initialize_faq_database():
    with closing(connect_app_database()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_life_faqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                email TEXT PRIMARY KEY,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS student_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_email TEXT NOT NULL REFERENCES app_users(email),
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_student_notes_owner_updated "
            "ON student_notes (owner_email, updated_at DESC)"
        )
        connection.commit()


def connect_app_database():
    connection = sqlite3.connect(FAQ_DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def normalize_email(email):
    normalized_email = email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized_email):
        raise ValueError("Enter a valid email address.")
    return normalized_email


def create_account(email, password):
    normalized_email = normalize_email(email)
    if len(password) < 8:
        raise ValueError("Use a password with at least 8 characters.")

    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    try:
        with closing(connect_app_database()) as connection:
            connection.execute(
                """
                INSERT INTO app_users (email, password_salt, password_hash)
                VALUES (?, ?, ?)
                """,
                (normalized_email, salt.hex(), password_hash.hex()),
            )
            connection.commit()
    except sqlite3.IntegrityError as error:
        raise ValueError("An account already exists for this email.") from error
    return normalized_email


def verify_account(email, password):
    try:
        normalized_email = normalize_email(email)
    except ValueError:
        return None

    with closing(connect_app_database()) as connection:
        row = connection.execute(
            """
            SELECT password_salt, password_hash
            FROM app_users
            WHERE email = ?
            """,
            (normalized_email,),
        ).fetchone()
    if row is None:
        return None

    salt = bytes.fromhex(row[0])
    expected_hash = bytes.fromhex(row[1])
    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    if hmac.compare_digest(actual_hash, expected_hash):
        return normalized_email
    return None


def load_student_notes(owner_email, search_text=""):
    normalized_email = normalize_email(owner_email)
    with closing(connect_app_database()) as connection:
        if search_text.strip():
            search_pattern = f"%{search_text.strip()}%"
            rows = connection.execute(
                """
                SELECT id, title, content, created_at, updated_at
                FROM student_notes
                WHERE owner_email = ?
                  AND (title LIKE ? OR content LIKE ?)
                ORDER BY updated_at DESC, id DESC
                """,
                (normalized_email, search_pattern, search_pattern),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, title, content, created_at, updated_at
                FROM student_notes
                WHERE owner_email = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (normalized_email,),
            ).fetchall()
    return [
        {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }
        for row in rows
    ]


def save_student_note(owner_email, title, content, note_id=None):
    normalized_email = normalize_email(owner_email)
    clean_title = title.strip()
    clean_content = content.strip()
    if not clean_title:
        raise ValueError("Add a title for your note.")
    if not clean_content:
        raise ValueError("Write something before saving your note.")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with closing(connect_app_database()) as connection:
        if note_id is None:
            cursor = connection.execute(
                """
                INSERT INTO student_notes
                    (owner_email, title, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_email, clean_title, clean_content, now, now),
            )
            saved_note_id = cursor.lastrowid
        else:
            cursor = connection.execute(
                """
                UPDATE student_notes
                SET title = ?, content = ?, updated_at = ?
                WHERE id = ? AND owner_email = ?
                """,
                (clean_title, clean_content, now, note_id, normalized_email),
            )
            if cursor.rowcount == 0:
                raise ValueError("That note is no longer available in your account.")
            saved_note_id = note_id
        connection.commit()
    return saved_note_id


def delete_student_note(owner_email, note_id):
    normalized_email = normalize_email(owner_email)
    with closing(connect_app_database()) as connection:
        cursor = connection.execute(
            """
            DELETE FROM student_notes
            WHERE id = ? AND owner_email = ?
            """,
            (note_id, normalized_email),
        )
        deleted = cursor.rowcount > 0
        connection.commit()
    return deleted


def open_note(note_id, title, content):
    st.session_state["selected_note_id"] = note_id
    st.session_state["pending_delete_note_id"] = None
    st.session_state["note_editor_title"] = title
    st.session_state["note_editor_content"] = content


def start_new_note():
    st.session_state["selected_note_id"] = None
    st.session_state["pending_delete_note_id"] = None
    st.session_state["note_editor_title"] = ""
    st.session_state["note_editor_content"] = ""


def request_note_deletion(note_id):
    st.session_state["pending_delete_note_id"] = note_id


def format_note_datetime(timestamp):
    return datetime.fromisoformat(timestamp).astimezone().strftime(
        "%b %d, %Y · %I:%M %p"
    )


def normalize_answer_value(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return " ".join(normalized.split())


def build_pdf_question_bank(pdf_text, question_limit):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", pdf_text)
        if 40 <= len(sentence.strip()) <= 500
    ]
    if not sentences:
        return []

    candidates = []
    for sentence in sentences:
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sentence)
        if len(words) < 6:
            continue
        strong_terms = [
            word
            for word in words
            if word.lower() not in STOP_WORDS and len(word) >= 5
        ]
        if not strong_terms:
            continue
        answer = max(set(strong_terms), key=lambda word: (strong_terms.count(word), len(word)))
        if not answer:
            continue
        candidates.append({"sentence": sentence, "answer": answer})

    if not candidates:
        return []

    questions = []
    used_answers = set()
    for index, candidate in enumerate(candidates):
        if len(questions) >= question_limit:
            break
        sentence = candidate["sentence"]
        answer = candidate["answer"]
        answer_key = answer.lower()
        if answer_key in used_answers:
            continue

        sentence_without_answer = re.sub(
            rf"\b{re.escape(answer)}\b",
            "____",
            sentence,
            count=1,
            flags=re.IGNORECASE,
        )
        if sentence_without_answer == sentence:
            sentence_without_answer = sentence

        distractor_pool = [
            item["answer"]
            for item in candidates
            if item["answer"].lower() != answer_key
            and item["answer"].lower() not in used_answers
        ]
        distractors = []
        for word in distractor_pool:
            if word not in distractors:
                distractors.append(word)
            if len(distractors) == 3:
                break

        if len(distractors) < 3:
            fallback_terms = [
                word for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sentence)
                if word.lower() not in STOP_WORDS and word.lower() != answer_key.lower()
            ]
            for word in fallback_terms:
                if word not in distractors and word.lower() != answer_key.lower():
                    distractors.append(word)
                if len(distractors) == 3:
                    break

        if len(distractors) < 3:
            continue

        options = [answer, *distractors[:3]]
        option_order = sorted(range(len(options)), key=lambda idx: (idx * 17 + len(answer) * 7) % 29)
        shuffled_options = [options[idx] for idx in option_order]
        correct_option = chr(ord("A") + shuffled_options.index(answer))
        question_type = "mcq" if index % 3 == 0 else "fill_blank"
        if index % 3 == 2:
            question_type = "short_answer"

        if question_type == "mcq":
            question = {
                "id": len(questions) + 1,
                "type": "mcq",
                "question": f"Choose the best word to complete the sentence: {sentence_without_answer}",
                "options": {
                    chr(ord("A") + option_index): option
                    for option_index, option in enumerate(shuffled_options)
                },
                "correct_option": correct_option,
                "answer": answer,
                "explanation": f"The sentence is about '{answer}', which is the most accurate term in the original text.",
            }
        elif question_type == "fill_blank":
            question = {
                "id": len(questions) + 1,
                "type": "fill_blank",
                "question": f"Fill in the blank from the PDF: {sentence_without_answer}",
                "answer": answer,
                "acceptable_answers": [answer, answer.title()],
                "explanation": f"The missing idea is '{answer}' because it matches the original passage.",
            }
        else:
            question = {
                "id": len(questions) + 1,
                "type": "short_answer",
                "question": f"In one short sentence, explain the idea described in this passage: {sentence}",
                "answer": answer,
                "acceptable_answers": [answer, answer.title(), answer.lower()],
                "explanation": f"A strong answer focuses on '{answer}' as the key idea in the passage.",
            }

        questions.append(question)
        used_answers.add(answer_key)

    return questions


def generate_pdf_quiz_questions(pdf_text, question_limit):
    return build_pdf_question_bank(pdf_text, question_limit)


def format_pdf_size(size_bytes):
    if size_bytes is None:
        return "Unknown"
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def get_pdf_page_count(pdf_bytes):
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            return len(document)
    except Exception:
        return None

st.set_page_config(
    page_title="Student Support & Dropout Risk",
    page_icon="🎓",
    layout="wide",
)

st.session_state.setdefault("authenticated_email", "")
st.session_state.setdefault("exam_questions", [])
st.session_state.setdefault("exam_source_name", "")
st.session_state.setdefault("quiz_last_result", None)
initialize_faq_database()

if not st.session_state["authenticated_email"]:
    center = st.columns([1, 2, 1])[1]

    with center:
        st.space("medium")
        with st.container(border=True):
            st.markdown(":material/health_and_safety:", text_alignment="center")
            st.title("Student support", text_alignment="center")
            st.markdown(
                "A thoughtful check-in can help students find the support "
                "they need to thrive.",
                text_alignment="center",
            )
            st.caption(
                "Sign in to continue to the student support tools.",
                text_alignment="center",
            )

            sign_in_tab, create_account_tab = st.tabs(
                ["Sign in", "Create account"]
            )
            with sign_in_tab:
                with st.form("account_login_form"):
                    login_email = st.text_input(
                        "Email address",
                        placeholder="you@example.com",
                        key="login_email",
                    )
                    login_password = st.text_input(
                        "Password",
                        type="password",
                        key="login_password",
                    )
                    login_submitted = st.form_submit_button(
                        "Sign in",
                        type="primary",
                        width="stretch",
                        icon=":material/arrow_forward:",
                    )
                if login_submitted:
                    authenticated_email = verify_account(
                        login_email,
                        login_password,
                    )
                    if authenticated_email:
                        st.session_state["authenticated_email"] = authenticated_email
                        st.rerun()
                    else:
                        st.error("The email or password is incorrect.")

            with create_account_tab:
                with st.form("create_account_form"):
                    new_account_email = st.text_input(
                        "Email address",
                        placeholder="you@example.com",
                        key="new_account_email",
                    )
                    new_account_password = st.text_input(
                        "Password",
                        type="password",
                        help="Use at least 8 characters.",
                        key="new_account_password",
                    )
                    confirm_password = st.text_input(
                        "Confirm password",
                        type="password",
                        key="confirm_account_password",
                    )
                    account_submitted = st.form_submit_button(
                        "Create account",
                        type="primary",
                        width="stretch",
                    )
                if account_submitted:
                    if new_account_password != confirm_password:
                        st.error("The passwords do not match.")
                    else:
                        try:
                            created_email = create_account(
                                new_account_email,
                                new_account_password,
                            )
                        except ValueError as error:
                            st.error(str(error))
                        else:
                            st.session_state["authenticated_email"] = created_email
                            st.rerun()

    st.stop()

with st.sidebar:
    st.caption(f"Signed in as: {st.session_state['authenticated_email']}")
    if st.button("Log out", width="stretch"):
        st.session_state["authenticated_email"] = ""
        st.session_state["selected_note_id"] = None
        st.session_state["pending_delete_note_id"] = None
        st.session_state["note_editor_title"] = ""
        st.session_state["note_editor_content"] = ""
        st.session_state["notes_search"] = ""
        st.rerun()

st.title("Student support hub", icon=":material/school:")
selected_feature = st.segmented_control(
    "Choose a tool",
    ["Dropout risk", "Exam quiz", "My Notes"],
    default="Dropout risk",
    key="selected_feature",
    width="stretch",
)

if selected_feature == "Dropout risk":

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    model_bundle = load_model()
    model = model_bundle["model"]

    st.subheader("Student dropout risk check")
    st.write(
        "Share a few details to see an estimated risk and supportive ideas "
        "for what to do next."
    )
    st.caption(
        "The estimate is a starting point for a caring conversation, not a "
        "diagnosis or a decision about a student's future."
    )

    with st.form("student_form"):
        st.subheader("Student information")
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", min_value=15, max_value=30, value=21, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            cgpa = st.number_input(
                "CGPA",
                min_value=1.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Enter CGPA on a 1 to 10 scale.",
            )
            attendance_rate = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=76.0,
                step=1.0,
            )
            mental_health_support = st.selectbox(
                "Currently receiving mental health support?",
                ["No", "Yes"],
            )

        with right:
            sleep_hours = st.number_input(
                "Sleep hours per night",
                min_value=1,
                max_value=24,
                value=7,
                step=1,
            )
            stress_level = st.number_input(
                "Stress level",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="1 means very low stress and 10 means very high stress.",
            )
            extracurricular_activities = st.selectbox(
                "Participates in extracurricular activities?",
                ["No", "Yes"],
            )
            study_mode = st.selectbox(
                "Study mode",
                ["Full-time", "Online", "Part-time"],
            )

        prediction_submitted = st.form_submit_button(
            "Check dropout risk",
            type="primary",
            width="stretch",
        )

    if prediction_submitted:
        student_data = {
            "age": age,
            "gender": gender,
            "cgpa": cgpa,
            "attendance_rate": attendance_rate,
            "mental_health_support": int(mental_health_support == "Yes"),
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "extracurricular_activities": int(
                extracurricular_activities == "Yes"
            ),
            "study_mode": study_mode,
        }
        input_df = pd.DataFrame(
            [student_data],
            columns=model_bundle["feature_columns"],
        )
        prediction = model.predict(input_df)[0]
        positive_class_index = list(model.classes_).index(1)
        dropout_risk = float(
            model.predict_proba(input_df)[0][positive_class_index]
        )

        st.subheader("Your result")
        st.metric("Estimated dropout risk", f"{dropout_risk * 100:.1f}%")
        risk_chart = pd.DataFrame(
            {
                "Outcome": ["Higher dropout risk", "Lower dropout risk"],
                "Likelihood (%)": [
                    dropout_risk * 100,
                    (1 - dropout_risk) * 100,
                ],
            }
        )
        risk_chart_plot = (
            alt.Chart(risk_chart)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                x=alt.X(
                    "Likelihood (%):Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Likelihood (%)",
                ),
                y=alt.Y(
                    "Outcome:N",
                    sort=["Higher dropout risk", "Lower dropout risk"],
                    title=None,
                ),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(
                        domain=["Higher dropout risk", "Lower dropout risk"],
                        range=["#FFC928", "#292A2E"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Outcome:N", title="Outcome"),
                    alt.Tooltip(
                        "Likelihood (%):Q",
                        format=".1f",
                        title="Likelihood",
                    ),
                ],
            )
            .properties(height=120)
        )
        st.altair_chart(
            risk_chart_plot,
            alt="Estimated likelihood of higher and lower dropout risk",
            width="stretch",
        )
        st.caption(
            "This estimate is based on patterns in the project dataset and is "
            "not a certainty. Discuss it with the student before deciding on support."
        )

        if prediction == 1:
            st.warning(
                "The estimate suggests extra support may be helpful. "
                "Approach the student with care and ask what would help them most."
            )
        else:
            st.success(
                "The estimate suggests lower risk right now. Keep encouraging "
                "the student and make support easy to reach."
            )

        st.markdown("#### A few ideas for what to improve next")
        suggestions = []
        if cgpa < 6.25:
            suggestions.append(
                "Offer a friendly academic check-in and ask whether tutoring, "
                "study groups, or a study plan would be useful."
            )
        if attendance_rate < 75:
            suggestions.append(
                "Explore what is making attendance difficult and agree on a "
                "realistic plan for reconnecting with classes."
            )
        if stress_level >= 7:
            suggestions.append(
                "Ask what is contributing to stress and share wellbeing or "
                "counselling resources if the student would like them."
            )
        if sleep_hours < 7:
            suggestions.append(
                "Check whether workload or daily routines are affecting rest, "
                "and offer help finding a more manageable schedule."
            )
        if mental_health_support == "No":
            suggestions.append(
                "Let the student know confidential support is available, "
                "without assuming they need it."
            )
        if extracurricular_activities == "No":
            suggestions.append(
                "If the student is interested, suggest a welcoming club or "
                "campus activity as one optional way to build connection."
            )
        if study_mode == "Online":
            suggestions.append(
                "Offer regular online check-ins and help the student connect "
                "with classmates and instructors."
            )
        elif study_mode == "Part-time":
            suggestions.append(
                "Review course and work commitments together and discuss a "
                "pace that feels sustainable."
            )
        if not suggestions:
            suggestions.append(
                "Ask what is working well, celebrate progress, and agree on a "
                "light-touch check-in to keep support available."
            )

        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        st.success(
            "Thank you for caring about student success. Every thoughtful "
            "check-in can make a difference. 💛",
            icon=":material/favorite:",
        )

elif selected_feature == "Exam quiz":
    st.subheader("Create an exam quiz from a PDF")
    st.write(
        "Upload study material and the app will generate multiple-choice, "
        "fill-in-the-blank and short-answer questions from the document text."
    )
    st.caption(
        "PDF files up to 500 MB are accepted. Password-protected PDFs are "
        "supported when you provide the password. Scanned pages use local "
        "English OCR when Tesseract is installed on the app server. Very large "
        "files need enough server memory and temporary disk space."
    )

    pdf_file = st.file_uploader(
        "Browse for a study PDF",
        type=["pdf"],
        key="exam_pdf_upload",
        max_upload_size=MAX_PDF_UPLOAD_MB,
        help="Choose a PDF up to 500 MB. Scanned PDFs require Tesseract OCR on the app server.",
    )
    if pdf_file is not None:
        pdf_page_count = get_pdf_page_count(pdf_file.getvalue())
        st.info(
            f"Selected PDF: {pdf_file.name} ? "
            f"Size: {format_pdf_size(pdf_file.size)} ? "
            f"Pages: {pdf_page_count if pdf_page_count is not None else 'unknown'} ? "
            f"Status: ready for upload"
        )

    pdf_password = st.text_input(
        "PDF password (if protected)",
        type="password",
        help="The password is used only while reading this uploaded PDF.",
    )
    use_ocr = st.checkbox(
        "Use OCR for scanned pages",
        value=True,
        help="OCR is performed on this server and requires Tesseract with English language data.",
    )
    question_limit = st.slider(
        "Number of quiz questions",
        min_value=3,
        max_value=10,
        value=5,
    )

    generate_quiz = st.button(
        "Generate quiz from PDF",
        type="primary",
        width="stretch",
        disabled=pdf_file is None,
    )
    if generate_quiz and pdf_file is not None:
        if pdf_file.size > MAX_PDF_UPLOAD_MB * 1024 * 1024:
            st.error(
                f"The PDF is larger than the {MAX_PDF_UPLOAD_MB} MB upload limit."
            )
        else:
            try:
                with st.spinner("Reading the PDF and preparing quiz questions..."):
                    extracted_text, ocr_pages, ocr_unavailable = extract_pdf_text(
                        pdf_file,
                        password=pdf_password,
                        use_ocr=use_ocr,
                    )
                if ocr_unavailable and ocr_pages:
                    st.warning(
                        "Tesseract OCR is not installed on the app server. "
                        "Scanned pages were skipped; install Tesseract with "
                        "English language data to read scanned PDFs."
                    )
                if not extracted_text.strip():
                    st.error(
                        "No readable text was found. For scanned PDFs, install "
                        "Tesseract OCR on the app server and try again."
                    )
                else:
                    generated_questions = generate_pdf_quiz_questions(
                        extracted_text,
                        question_limit,
                    )
                    if not generated_questions:
                        st.error(
                            "There was not enough readable text to make a quiz. "
                            "Try a PDF with more selectable text."
                        )
                    else:
                        st.session_state["exam_questions"] = generated_questions
                        st.session_state["exam_source_name"] = pdf_file.name
                        st.session_state["exam_page_count"] = get_pdf_page_count(pdf_file.getvalue())
                        st.session_state["quiz_last_result"] = None
                        st.success(
                            f"Created {len(generated_questions)} questions from {pdf_file.name}."
                        )
            except PDFPasswordError as error:
                st.error(str(error))
            except PDFProcessingError as error:
                st.error(str(error))

    if st.session_state["exam_questions"]:
        st.caption(
            f"Quiz source: {st.session_state['exam_source_name']} ? "
            f"{len(st.session_state['exam_questions'])} questions ? "
            f"{st.session_state.get('exam_page_count', 'unknown')} pages"
        )
        st.markdown(
            f"#### Quiz round ? {len(st.session_state['exam_questions'])} questions"
        )
        with st.form("exam_quiz_form"):
            quiz_answers = {}
            for question in st.session_state["exam_questions"]:
                question_type = question["type"]
                if question_type == "mcq":
                    option_labels = [
                        f"{letter}. {answer}"
                        for letter, answer in question["options"].items()
                    ]
                    response = st.radio(
                        question["question"],
                        option_labels,
                        key=f"exam_answer_{question['id']}",
                    )
                    quiz_answers[question["id"]] = response.split(".", maxsplit=1)[0]
                elif question_type == "fill_blank":
                    response = st.text_input(
                        question["question"],
                        key=f"exam_answer_{question['id']}",
                    )
                    quiz_answers[question["id"]] = response
                else:
                    response = st.text_area(
                        question["question"],
                        key=f"exam_answer_{question['id']}",
                        height=140,
                    )
                    quiz_answers[question["id"]] = response
            quiz_submitted = st.form_submit_button(
                "Submit quiz",
                type="primary",
                width="stretch",
            )

        if quiz_submitted:
            correct_count = 0
            reviewed_questions = []
            for question in st.session_state["exam_questions"]:
                response = quiz_answers.get(question["id"], "")
                question_type = question["type"]
                if question_type == "mcq":
                    correct = response == question["correct_option"]
                elif question_type == "fill_blank":
                    normalized_response = normalize_answer_value(response)
                    acceptable = {
                        normalize_answer_value(value)
                        for value in question.get("acceptable_answers", [question["answer"]])
                    }
                    correct = normalized_response in acceptable
                else:
                    normalized_response = normalize_answer_value(response)
                    acceptable = {
                        normalize_answer_value(value)
                        for value in question.get("acceptable_answers", [question["answer"]])
                    }
                    correct = normalized_response in acceptable

                if correct:
                    correct_count += 1
                reviewed_questions.append({
                    "question": question,
                    "response": response,
                    "correct": correct,
                })

            total_questions = len(st.session_state["exam_questions"])
            st.session_state["quiz_last_result"] = {
                "correct": correct_count,
                "total": total_questions,
                "answers": reviewed_questions,
            }

        quiz_result = st.session_state["quiz_last_result"]
        if quiz_result:
            correct_count = quiz_result["correct"]
            total_questions = quiz_result["total"]
            missed_count = total_questions - correct_count
            score_percent = correct_count / total_questions * 100
            st.markdown("#### Answer review")
            for result in quiz_result["answers"]:
                question = result["question"]
                correct = result["correct"]
                if correct:
                    st.success(f"{question['question']} ? Correct")
                else:
                    if question["type"] == "mcq":
                        correct_answer = question["options"][question["correct_option"]]
                        st.info(
                            f"{question['question']} ? Correct answer: "
                            f"{question['correct_option']}. {correct_answer}"
                        )
                    else:
                        st.info(
                            f"{question['question']} ? Correct answer: {question['answer']}"
                        )
                    if question.get("explanation"):
                        st.caption(question["explanation"])

            st.markdown("#### Your final score")
            st.metric(
                "Correct answers",
                f"{correct_count} / {total_questions}",
                delta=f"{score_percent:.0f}%",
            )
            score_data = pd.DataFrame(
                {
                    "Answer result": ["Correct", "Needs review"],
                    "Questions": [correct_count, missed_count],
                }
            )
            score_chart = (
                alt.Chart(score_data)
                .mark_bar(cornerRadiusEnd=6)
                .encode(
                    x=alt.X("Questions:Q", title="Number of questions"),
                    y=alt.Y(
                        "Answer result:N",
                        sort=["Correct", "Needs review"],
                        title=None,
                    ),
                    color=alt.Color(
                        "Answer result:N",
                        scale=alt.Scale(
                            domain=["Correct", "Needs review"],
                            range=["#FFC928", "#292A2E"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Answer result:N", "Questions:Q"],
                )
                .properties(height=100)
            )
            st.altair_chart(
                score_chart,
                alt="Number of correct answers and questions to review",
                width="stretch",
            )

elif selected_feature == "My Notes":
    st.subheader("My Notes")
    st.write("YOUR THOUGHT → PAGE → SAVE")
    st.caption(
        "A personal space for reflections, ideas and everyday moments. "
        "Your notes are available only when signed in to your account."
    )
    st.caption(
        "Notes and account records are stored in the project's local SQLite "
        "database. For a hosted deployment, use HTTPS and protect access to "
        "the app server and database file."
    )

    owner_email = st.session_state["authenticated_email"]
    all_notes = load_student_notes(owner_email)
    search_text = st.text_input(
        "Search your notes",
        placeholder="Search titles or note text...",
        key="notes_search",
    )
    search_term = search_text.strip().casefold()
    visible_notes = [
        note
        for note in all_notes
        if not search_term
        or search_term in note["title"].casefold()
        or search_term in note["content"].casefold()
    ]

    selected_note_id = st.session_state.get("selected_note_id")
    selected_note = next(
        (note for note in all_notes if note["id"] == selected_note_id),
        None,
    )
    if selected_note_id is not None and selected_note is None:
        start_new_note()
        selected_note_id = None

    notes_column, editor_column = st.columns([1, 2])
    with notes_column:
        st.markdown("#### Your pages")
        st.button(
            "New note",
            key="new_note_button",
            width="stretch",
            icon=":material/add:",
            on_click=start_new_note,
        )

        if not all_notes:
            st.info("No notes yet. Start writing your first thought!")
        elif not visible_notes:
            st.info("No notes match your search. Try another word.")
        else:
            for note in visible_notes:
                with st.container(border=True):
                    st.write(note["title"])
                    st.caption(format_note_datetime(note["updated_at"]))
                    preview = note["content"].strip().replace("\n", " ")
                    st.caption(
                        preview[:100] + ("…" if len(preview) > 100 else "")
                    )
                    open_column, delete_column = st.columns(2)
                    with open_column:
                        st.button(
                            "Open",
                            key=f"open_note_{note['id']}",
                            width="stretch",
                            on_click=open_note,
                            args=(note["id"], note["title"], note["content"]),
                        )
                    with delete_column:
                        st.button(
                            "Delete",
                            key=f"request_delete_note_{note['id']}",
                            width="stretch",
                            on_click=request_note_deletion,
                            args=(note["id"],),
                        )

        pending_delete_id = st.session_state.get("pending_delete_note_id")
        pending_note = next(
            (note for note in all_notes if note["id"] == pending_delete_id),
            None,
        )
        if pending_note is not None:
            st.warning(f"Delete “{pending_note['title']}”? This cannot be undone.")
            confirm_column, cancel_column = st.columns(2)
            with confirm_column:
                if st.button(
                    "Confirm delete",
                    key="confirm_delete_note",
                    type="primary",
                    width="stretch",
                ):
                    if delete_student_note(owner_email, pending_delete_id):
                        if st.session_state.get("selected_note_id") == pending_delete_id:
                            start_new_note()
                        else:
                            st.session_state["pending_delete_note_id"] = None
                        st.rerun()
                    else:
                        st.error("That note could not be found in your account.")
            with cancel_column:
                st.button(
                    "Cancel",
                    key="cancel_delete_note",
                    width="stretch",
                    on_click=lambda: st.session_state.__setitem__(
                        "pending_delete_note_id", None
                    ),
                )

    with editor_column:
        st.markdown("#### Write your thought")
        if selected_note is not None:
            st.caption(
                f"Created {format_note_datetime(selected_note['created_at'])} · "
                f"Updated {format_note_datetime(selected_note['updated_at'])}"
            )
        with st.form("note_editor_form", clear_on_submit=False):
            note_title = st.text_input(
                "Title",
                placeholder="Give this note a title",
                key="note_editor_title",
            )
            note_content = st.text_area(
                "Your thought",
                placeholder="Write anything on your mind...",
                height=320,
                key="note_editor_content",
            )
            note_submitted = st.form_submit_button(
                "SAVE NOTE" if selected_note_id is None else "UPDATE NOTE",
                type="primary",
                width="stretch",
            )

        if note_submitted:
            try:
                saved_note_id = save_student_note(
                    owner_email,
                    note_title,
                    note_content,
                    note_id=selected_note_id,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state["selected_note_id"] = saved_note_id
                st.session_state["pending_delete_note_id"] = None
                st.success("Your note was saved.")
                st.rerun()
