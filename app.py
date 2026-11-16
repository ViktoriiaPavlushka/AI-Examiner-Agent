import json

import streamlit as st
from groq import APIError, AuthenticationError, Groq, RateLimitError

from agent import ExaminerAgent

MODEL = "openai/gpt-oss-120b"
BOT_AVATAR = ":material/school:"
USER_AVATAR = ":material/person:"

st.set_page_config(page_title="AI Examiner", page_icon=":material/school:", layout="centered")

st.markdown(
    """
    <style>
    :root {
        --pink: #f3c3d3;
        --pink-soft: #fcedf2;
        --pink-line: #ecc9d6;
        --pink-dark: #a2546f;
        --yellow: #f6e3a1;
        --yellow-soft: #fdf8e7;
        --green: #cfe5cc;
        --green-soft: #f1f8ef;
        --green-dark: #4f7d52;
        --line: #ebe1e4;
        --ink: #3f3539;
        --muted: #7a6c72;
    }

    .stApp { background: #fcfaf8; color: var(--ink); }

    /* Світлий вигляд незалежно від теми браузера/системи (темний режим) */
    header[data-testid="stHeader"] { background: #fcfaf8 !important; }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] span { color: var(--muted) !important; }
    div[data-testid="stBottom"], div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] { background: #fcfaf8 !important; }
    section[data-testid="stSidebar"] * { color: var(--ink); }
    div[data-testid="stTextInputRootElement"] {
        background: #ffffff !important; border: 1px solid var(--pink-line) !important;
        border-radius: 6px !important;
    }
    div[data-testid="stTextInputRootElement"]:focus-within { border-color: #e48aac !important; }
    div[data-testid="stTextInput"] input { border: 0 !important; }
    div[data-testid="stTextInput"] button { background: #ffffff !important; color: var(--muted) !important; }
    input::placeholder, textarea::placeholder { color: #a8999f !important; opacity: 1; }
    textarea, input { -webkit-text-fill-color: var(--ink) !important; }
    textarea::placeholder, input::placeholder { -webkit-text-fill-color: #a8999f !important; }
    div[data-testid="stChatInput"] > div { background: #ffffff !important; }
    div[data-testid="stChatInput"] button { background: var(--pink) !important; color: var(--ink) !important; }
    div[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"],
    div[data-testid="stChatMessage"] > div:first-child {
        background: var(--pink-soft) !important; color: var(--pink-dark) !important;
    }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * { color: var(--ink) !important; }
    div[data-testid="stTooltipIcon"] button,
    span[data-testid="stTooltipIcon"] button { background: transparent !important; border: 0 !important; }
    [data-testid="stTooltipIcon"] svg { color: var(--muted) !important; stroke: var(--muted) !important; }
    div[data-testid="stChatMessage"] [data-testid="stIconMaterial"] {
        color: var(--pink-dark) !important; opacity: 1 !important;
    }
    div[data-testid="stChatMessage"] > div:first-child { border: 1px solid var(--pink-line) !important; }
    .block-container { max-width: 860px; padding-top: 3.2rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label, li, .stMarkdown { color: var(--ink); }

    section[data-testid="stSidebar"] {
        background: var(--pink-soft);
        border-right: 1px solid var(--pink-line);
    }
    section[data-testid="stSidebar"] h3 {
        font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
        color: var(--muted); font-weight: 600;
    }

    .header {
        padding: 0 0 1rem 1.1rem;
        border-left: 4px solid var(--pink);
        margin-bottom: 1.4rem;
    }
    .header h1 { margin: 0; padding: 0; font-size: 1.9rem; font-weight: 700; color: var(--ink); }
    .header p  { margin: .4rem 0 0 0; color: var(--muted); font-size: .98rem; line-height: 1.5; }
    .tags { margin-top: .75rem; display: flex; gap: .4rem; flex-wrap: wrap; }
    .tag {
        padding: .12rem .55rem; border-radius: 4px; font-size: .78rem;
        font-weight: 600; color: var(--ink); border: 1px solid transparent;
    }
    .tag.pink   { background: var(--pink-soft);   border-color: var(--pink-line); }
    .tag.yellow { background: var(--yellow-soft); border-color: #eedfa8; }
    .tag.green  { background: var(--green-soft);  border-color: #cfe2cb; }

    div[data-testid="stTextInput"] input,
    div[data-testid="stChatInput"] textarea {
        background: #ffffff !important; border-radius: 6px; color: var(--ink) !important;
    }
    div[data-testid="stChatInput"] {
        border: 1px solid var(--line); border-radius: 8px; background: #ffffff;
    }
    div[data-testid="stExpander"] details {
        border-radius: 6px; border-color: var(--line); background: #ffffff;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 6px;
        padding: .8rem 1rem;
        margin-bottom: .5rem;
        background: #ffffff;
        border: 1px solid var(--line);
    }
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
        border-left: 4px solid var(--pink);
    }
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        background: var(--yellow-soft); border-left: 4px solid var(--yellow);
    }

    div.stButton > button, div.stDownloadButton > button {
        width: 100%; min-height: 2.6rem; border-radius: 6px;
        border: 1px solid #e6b3c5; color: var(--ink); font-weight: 600;
        background: var(--pink);
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #d99ab1; color: var(--ink); background: #eeb2c7;
    }

    .card {
        padding: .8rem 1rem; border-radius: 6px; margin-bottom: .8rem;
        background: #ffffff; border: 1px solid var(--pink-line);
    }
    .card h4 {
        margin: 0 0 .45rem 0; font-size: .78rem; text-transform: uppercase;
        letter-spacing: .08em; color: var(--muted); font-weight: 600;
    }
    .card .name { font-weight: 600; }
    .card .meta { font-size: .85rem; color: var(--muted); }
    .topic { display: flex; gap: .55rem; align-items: flex-start;
             padding: .35rem 0; font-size: .9rem; line-height: 1.35;
             border-top: 1px solid #f3ecee; }
    .card h4 + .topic { border-top: 0; }
    .topic .badge {
        flex-shrink: 0; min-width: 2.7rem; text-align: center;
        padding: .08rem .35rem; border-radius: 4px; font-size: .75rem; font-weight: 700;
    }
    .badge.done    { background: var(--green);  color: #3d6640; }
    .badge.current { background: var(--pink);   color: #7f3f57; }
    .badge.wait    { background: var(--yellow); color: #73602a; }

    .result {
        margin: 1rem 0 1.2rem 0; padding: 1.2rem 1.4rem; border-radius: 6px;
        background: var(--green-soft); border: 1px solid #cfe2cb;
        border-left: 4px solid var(--green-dark);
    }
    .result .label {
        font-size: .78rem; text-transform: uppercase; letter-spacing: .08em;
        color: var(--muted); font-weight: 600; margin-bottom: .3rem;
    }
    .result .score { font-size: 2.6rem; font-weight: 700; color: var(--green-dark); line-height: 1; }
    .result .score span { font-size: 1.1rem; color: var(--muted); font-weight: 500; }
    .result .topic { border-top-color: #dcebd9; }
    .result .feedback { margin-top: .9rem; white-space: pre-wrap; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_agent(model: str) -> ExaminerAgent:
    return ExaminerAgent(model=model)


def esc(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Налаштування")
    api_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Ключ не зберігається в коді — лише в пам'яті поточної сесії. "
        "Отримати безкоштовно: https://console.groq.com/keys",
    )

    if "agent" not in st.session_state:
        st.session_state.agent = new_agent(MODEL)
    agent: ExaminerAgent = st.session_state.agent
    session = agent.session

    if st.button("Почати іспит заново"):
        st.session_state.agent = new_agent(MODEL)
        st.rerun()

    # --- прогрес іспиту ---
    if session.student:
        rows = []
        for i, topic in enumerate(session.topics):
            grade = session.topic_grades.get(topic)
            if grade:
                badge = f'<span class="badge done">{grade["score"]:g}</span>'
            elif topic == session.current_topic:
                badge = '<span class="badge current">зараз</span>'
            else:
                badge = '<span class="badge wait">далі</span>'
            rows.append(f'<div class="topic">{badge}<div>{esc(topic)}</div></div>')
        st.markdown(
            f"""<div class="card"><h4>Студент</h4>
            <div class="name">{esc(session.student['name'])}</div>
            <div class="meta">{esc(session.student['email'])}
            · {esc(session.student.get('group', ''))}</div></div>
            <div class="card"><h4>Теми іспиту</h4>{''.join(rows)}</div>""",
            unsafe_allow_html=True,
        )


# ---------------- Main ----------------
st.markdown(
    """
    <div class="header">
        <h1>AI Examiner</h1>
        <p>LLM-агент, який приймає усний іспит з курсу «Обробка мови методами
        штучного інтелекту»: перевіряє студента у списку, обирає випадкові теми,
        веде діалог з уточнювальними запитаннями та виставляє оцінку з відгуком.</p>
        <div class="tags">
            <span class="tag pink">LLM-агент</span>
            <span class="tag yellow">Groq API</span>
            <span class="tag green">Оцінка 0–10 і відгук</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Як це працює?"):
    st.markdown(
        """
1. Введіть **Groq API key** на бічній панелі.
2. Напишіть боту своє **ім'я, прізвище та email** — після перевірки ви отримаєте 2–3 випадкові теми.
3. Відповідайте на запитання. По кожній темі може бути кілька уточнювальних запитань.
4. Після останньої теми ви побачите оцінку (0–10) і відгук: що вдалося добре і над чим варто попрацювати.
        """
    )

chat_box = st.container()
with chat_box:
    for msg in agent.llm_messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant" and msg.get("content") and not msg.get("tool_calls"):
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(msg["content"])

    if session.finished:
        grades = "".join(
            f'<div class="topic"><span class="badge done">{g["score"]:g}</span>'
            f"<div><b>{esc(t)}</b><br><span style='color:#7d6b74'>{esc(g['comment'])}</span></div></div>"
            for t, g in session.topic_grades.items()
        )
        st.markdown(
            f"""<div class="result">
                <div class="label">Підсумкова оцінка</div>
                <div class="score">{session.final_score:g}<span> / 10</span></div>
                <div style="margin-top:1rem">{grades}</div>
                <div class="feedback">{esc(session.feedback)}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Завантажити протокол іспиту (JSON)",
            data=json.dumps(agent.history, ensure_ascii=False, indent=2),
            file_name=f"exam_{session.exam_id}.json",
            mime="application/json",
        )

placeholder = (
    "Іспит завершено" if session.finished
    else "Введіть відповідь…" if api_key
    else "Спочатку введіть Groq API key на бічній панелі"
)
prompt = st.chat_input(placeholder, disabled=session.finished or not api_key)

if prompt:
    with chat_box:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            try:
                with st.spinner("Екзаменатор думає…"):
                    reply = agent.chat(Groq(api_key=api_key), prompt)
                st.markdown(reply)
            except AuthenticationError:
                st.error("Неправильний API-ключ. Перевірте ключ Groq і спробуйте ще раз.")
            except RateLimitError:
                st.warning("Перевищено ліміт запитів Groq. Зачекайте хвилину й повторіть.")
            except (APIError, RuntimeError) as exc:
                st.error(f"Помилка під час звернення до моделі: {exc}")
            else:
                st.rerun()
